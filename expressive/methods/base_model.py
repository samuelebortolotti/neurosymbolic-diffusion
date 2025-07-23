from typing import Dict, Optional, Tuple
import numpy as np
from torch import Tensor, nn
import math
from abc import ABC, abstractmethod

import torch

from expressive.args import AbsArguments
from expressive.methods.logger import PRED_TYPES_W, PRED_TYPES_Y, BOIATestLog, TestLog, TrainingLog
from expressive.models.diffusion_model import WY_DATA, ForwardAbsorbing, UnmaskingModel
from torch.distributions import Categorical

from expressive.util import compute_ece, get_models, marginal_mode, safe_reward, safe_sample_categorical, true_mode, int_to_digit_tensor
from torch.nn import functional as F

class Problem(ABC):
    @abstractmethod
    def shape_w(self) -> torch.Size:
        pass

    @abstractmethod
    def shape_y(self) -> torch.Size:
        pass

    @abstractmethod
    def y_from_w(self, w_SBW: Tensor) -> Tensor:
        pass

    def eval_y(self, y_0_SBY: Tensor, y_0_BY: Tensor, w_0_BW: Tensor) -> Tensor:
        return torch.all(y_0_SBY == y_0_BY, dim=-1)


class BaseNeSyDiffusion(nn.Module, ABC):
    """
    Base class for NeSy diffusion models. Mostly implements the sampling algorithms
    """

    def __init__(self, p: UnmaskingModel, problem: Problem, args: AbsArguments):
        super().__init__()
        self.p = p
        self.q_w = ForwardAbsorbing(problem.shape_w()[-1])
        self.problem = problem
        self.inference_layer = nn.Linear(problem.shape_w()[1] ** problem.shape_w()[0], problem.out_dim)
        self.args = args
        if args.entropy_variant == "exact_conditional" and (not hasattr(args, "dataset") or args.dataset != "boia"):
            all_assignments_MW, all_y_outs_MY = get_models(problem)
            self.register_buffer("all_assignments_MW", all_assignments_MW)
            self.register_buffer("all_y_outs_MY", all_y_outs_MY)

    def first_hitting_sampler(
        self,
        encoding_SBWE: Tensor,
        w_n_SBW: Tensor,
        y_n_SBY: Tensor,
        L: int,
        S: int,
        only_w: bool = False,
    ) -> WY_DATA:
        """Parallel first-hitting sampler (See Zhang et al 2024) for NeSy masked diffusion."""
        t = 1
        for n in range(L, 0, -1):
            # Compute timestep s to jump to. Assumes linear schedule
            u = torch.rand(w_n_SBW.shape[:-1], device=w_n_SBW.device)
            s = u ** (1 / n) * t

            # Compute distribution at timestep s
            input_nn = (w_n_SBW, y_n_SBY) if not only_w else w_n_SBW
            p_w_SBWD = self.p.distribution(input_nn, encoding_SBWE, s)

            # Sample tw_0 and ty_0 from p(tw_0, ty_0|y_t) using (eg) rejection sampling / resampling
            tw_0_SBW, ty_0_SBY = self.reject_sample_w_0(p_w_SBWD, y_n_SBY, S, only_w)

            # Indices to (possibly) accept in w
            is_masked_SBW = w_n_SBW == self.mask_dim_w()
            # Create a mask for the accepted positions for w
            i_w_SB = self.sample_masked_indices(is_masked_SBW)

            # Decide whether to accept in w or y
            accept_w_SB = None
            is_masked_SBY = None
            i_y_SB = None

            if only_w:
                mask = torch.zeros_like(w_n_SBW, dtype=torch.bool)
                mask[
                    torch.arange(w_n_SBW.shape[0])[:, None, None],
                    torch.arange(w_n_SBW.shape[1])[None, :, None],
                    i_w_SB[:, :, None],
                ] = 1
                w_n_SBW = torch.where(mask, tw_0_SBW, w_n_SBW)
                num_masked_SB = torch.sum(w_n_SBW == self.mask_dim_w(), dim=-1)
            else:
                num_w_masked_SB = torch.sum(is_masked_SBW, dim=-1)
                p_accept_w_SB = num_w_masked_SB / n
                accept_w_SB = (
                    torch.rand(num_w_masked_SB.shape, device=w_n_SBW.device)
                    < p_accept_w_SB
                )

                # Create a mask for the accepted positions for w
                mask = torch.zeros_like(w_n_SBW, dtype=torch.bool)
                mask[
                    torch.arange(w_n_SBW.shape[0])[:, None, None],
                    torch.arange(w_n_SBW.shape[1])[None, :, None],
                    i_w_SB[:, :, None],
                ] = accept_w_SB[:, :, None]

                # Update w_n_SBW using the mask
                w_n_SBW = torch.where(mask, tw_0_SBW, w_n_SBW)

                # Same for y
                # Indices to (possibly) accept in y
                is_masked_SBY = y_n_SBY == self.mask_dim_y()
                i_y_SB = self.sample_masked_indices(is_masked_SBY)

                mask = torch.zeros_like(y_n_SBY, dtype=torch.bool)
                mask[
                    torch.arange(y_n_SBY.shape[0])[:, None, None],
                    torch.arange(y_n_SBY.shape[1])[None, :, None],
                    i_y_SB[:, :, None],
                ] = ~accept_w_SB[:, :, None]
                y_n_SBY = torch.where(mask, ty_0_SBY, y_n_SBY)

                num_masked_SB = torch.sum(
                    w_n_SBW == self.mask_dim_w(), dim=-1
                ) + torch.sum(y_n_SBY == self.mask_dim_y(), dim=-1)
            if not torch.all(num_masked_SB == n - 1):
                print(f"num_masked_SB: {num_masked_SB}")
                print(f"n-1: {n-1}")
                print(f"accept_w_SB: {accept_w_SB}")
                print(f"is_masked_SBW: {is_masked_SBW}")
                print(f"i_w_SB: {i_w_SB}")
                print(f"is_masked_SBY: {is_masked_SBY}")
                print(f"i_y_SB: {i_y_SB}")
                raise ValueError("One of the samples was not updated")

            t = s

        return w_n_SBW, y_n_SBY

    def discretised_sampler(
        self,
        encoding_SBWE: Tensor,
        w_n_SBW: Tensor,
        y_n_SBY: Tensor,
        T: int,
        S: int,
        only_w: bool = False,
    ) -> WY_DATA:
        """Traditional discrete diffusion sampler with fixed number of timesteps."""
        for step in range(T):
            # Compute timestep (linear schedule from 1 to 0)
            t = 1.0 - (step / T)

            # Get distribution at current timestep
            input_nn = (w_n_SBW, y_n_SBY) if not only_w else w_n_SBW
            p_w_SBWD = self.p.distribution(input_nn, encoding_SBWE, torch.tensor(t))

            # Sample using rejection sampling
            tw_0_SBW, ty_0_SBY = self.reject_sample_w_0(p_w_SBWD, y_n_SBY, S, only_w)

            # Find currently masked dimensions for w and y
            masked_w = w_n_SBW == self.mask_dim_w()

            # Compute unmasking probability based on linear schedule
            # At t=1, prob=0; at t=0, prob=1
            unmask_prob = 1 / (t * T)

            # Generate random values for each masked position
            rand_w = torch.rand_like(masked_w.float())

            # Determine which positions to unmask
            unmask_w = (rand_w < unmask_prob) & masked_w

            # Update values
            w_n_SBW = torch.where(unmask_w, tw_0_SBW, w_n_SBW)
            if not only_w:
                # Repeat for y
                masked_y = y_n_SBY == self.mask_dim_y()
                rand_y = torch.rand_like(masked_y.float())
                unmask_y = (rand_y < unmask_prob) & masked_y
                y_n_SBY = torch.where(unmask_y, ty_0_SBY, y_n_SBY)

        return w_n_SBW, y_n_SBY

    def sample(
        self,
        x_BX: Tensor,
        w_T_BW: Tensor,
        y_T_BY: Optional[Tensor],
        num_samples: int,
        T: Optional[int],
        S: int,
        encoding_BWE: Optional[Tensor] = None,
        only_w: bool = False,
    ) -> WY_DATA:
        """Initialize common components for sampling methods.

        Args:
            x_BX: Input tensor
            w_T_BW: Initial w tensor (can be masked)
            y_T_BY: Initial y tensor (can be masked)
            num_samples: Number of samples to draw
            T: Number of timesteps to use. If None, uses the first-hitting sampler.
            S: Number of samples to draw for rejection sampling
            only_w: If True, only w is sampled, otherwise w and y are both sampled (for the linked model)
        Returns:
            Tuple containing:
            - w_n_SBW: Final w tensor
            - y_n_SBY: Final y tensor
        """
        if not only_w and y_T_BY is None:
            raise ValueError("y_T_BY must be provided if only_w is False")
        if encoding_BWE is None:
            encoding_BWE = self.p.encode_x(x_BX)
        # TODO: Is it really needed to expand like this?
        encoding_SBWE = encoding_BWE[None, :, :].expand(
            (num_samples,) + encoding_BWE.shape
        )

        # Clone to avoid in-place operations
        w_n_SBW = torch.clone(w_T_BW[None, :, :]).expand((num_samples,) + w_T_BW.shape)
        if y_T_BY is not None:
            y_n_SBY = torch.clone(y_T_BY[None, :, :]).expand(
                (num_samples,) + y_T_BY.shape
            )
        else:
            y_n_SBY = None

        L = torch.sum(w_T_BW == self.mask_dim_w(), dim=1)
        if not only_w:
            L += torch.sum(y_T_BY == self.mask_dim_y(), dim=1)
        assert torch.all(L == L[0])
        # If T is not provided, or is greater than the number of masked dimensions, use first-hitting (exact) sampler
        if T is None or L[0] <= T:
            # Use first-hitting sampler. Calculate number of masked dimensions to unmask
            w_0_SBW, y_0_SBY = self.first_hitting_sampler(
                encoding_SBWE, w_n_SBW, y_n_SBY, L[0].cpu().item(), S, only_w
            )
        else:
            # Use discretised sampler for T timesteps
            w_0_SBW, y_0_SBY = self.discretised_sampler(
                encoding_SBWE, w_n_SBW, y_n_SBY, T, S, only_w
            )

        assert torch.all(
            w_0_SBW != self.mask_dim_w()
        ), "Some w dimensions remain masked"
        if not only_w:
            assert torch.all(
                y_0_SBY != self.mask_dim_y()
            ), "Some y dimensions remain masked"

            return w_0_SBW, y_0_SBY
        return w_0_SBW

    def sample_masked_indices(self, is_masked_SBD: Tensor) -> Tensor:
        # Count how many dimensions are masked
        num_masked_SB = torch.sum(is_masked_SBD, dim=-1)

        # Compute probability of unmasking each dimension.
        # Initialise uniformly in case no dimensions are masked, to prevent division by zero
        prob_masked_SB = torch.ones_like(is_masked_SBD) / is_masked_SBD.shape[0]
        has_masked_SB = num_masked_SB > 0
        prob_masked_SB[has_masked_SB] = (
            is_masked_SBD.float() / num_masked_SB[:, :, None]
        )[has_masked_SB]

        # Sample indices
        distr_masked = Categorical(probs=prob_masked_SB)
        i_SB = safe_sample_categorical(distr_masked)
        return i_SB

    def reject_sample_w_0(
        self, p_w_SBWD: Tensor, y_n_SBY: Tensor, S_samples: int, only_w: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Performs rejection sampling to get valid samples of w_0 that optimise for satisfying constraints.

        Args:
            p_w_SBWD: Probability distribution over w values, shape (S,B,W,D):
            y_n_SBY: Target y values to match, shape (S,B,Y):
            strict: If True, requires all unmasked dimensions of y to match.
                   If False, returns the sample that matches the most unmasked dimensions of y.

        Returns:
            tw_0_SBW: Accepted samples of w_0 that satisfy constraints, shape (S,B,W)

        Raises:
            ValueError: If strict=True, and no valid samples are found.

        The sampling process:
        1. Draws K sets of samples from p_w distribution
        2. Computes corresponding y values using varphi function
        3. Checks which samples match y_n in unmasked dimensions
        4. Takes first valid sample for each (S,B) position
        """

        p_w_dist = torch.distributions.Categorical(probs=p_w_SBWD[..., :-1])

        # In the first step of unconditional sampling, all dimensions are masked.
        # This saves an unnecessary call to the symbolic function
        if y_n_SBY is None or torch.all(y_n_SBY == self.mask_dim_y()):
            tw_0_SBW = safe_sample_categorical(p_w_dist)
            if only_w:
                return tw_0_SBW, None
            else:
                ty_0_SBW = self.problem.y_from_w(tw_0_SBW)
                return tw_0_SBW, ty_0_SBW
        tw_0_KSBW = safe_sample_categorical(p_w_dist, (S_samples,))

        # Compute output ty_0
        # Note: We do _not_ use carry-over unmasking here, since we want to calculate the violation on the unmasked values
        ty_0_KSBY = self.problem.y_from_w(tw_0_KSBW)

        # Compute violations and rewards
        violations_KSBY = (ty_0_KSBY != y_n_SBY.unsqueeze(0)) * (
            y_n_SBY != self.mask_dim_y()
        ).unsqueeze(0)
        rewards_KSB, _ = safe_reward(violations_KSBY, beta=self.args.beta)

        # Sample in proportion to rewards with self-normalised importance sampling
        probs_KSB = rewards_KSB / rewards_KSB.sum(dim=0, keepdim=True)
        if len(probs_KSB.shape) == 3:
            probs_SBK = probs_KSB.permute(1, 2, 0)
        else:
            probs_SBK = probs_KSB.permute(1, 0)
        accepted_idx_SB = safe_sample_categorical(Categorical(probs=probs_SBK))

        # Use advanced indexing to select the first accepted sample
        # TODO: Maybe a better way to handle shapes is possible
        if len(tw_0_KSBW.shape) == 4:
            _, S, B, _ = tw_0_KSBW.shape
            tw_0_SBW = tw_0_KSBW[
                accepted_idx_SB, torch.arange(S)[:, None], torch.arange(B)
            ]
            ty_0_SBY = ty_0_KSBY[
                accepted_idx_SB, torch.arange(S)[:, None], torch.arange(B)
            ]
        else:
            _, B, _ = tw_0_KSBW.shape
            tw_0_SBW = tw_0_KSBW[accepted_idx_SB, torch.arange(B)]
            ty_0_SBY = ty_0_KSBY[accepted_idx_SB, torch.arange(B)]

        return tw_0_SBW, ty_0_SBY

    def entropy_loss(self, y_0_BY: Tensor, q_w_0_BWD: Tensor) -> Tensor:
        dist = Categorical(probs=q_w_0_BWD)
        if self.args.entropy_variant == "exact_conditional":
            if hasattr(self.args, "dataset") and self.args.dataset == "boia":
                # If boia, use special implementation 
                dpl_model = self.problem.dataset.dpl_model
                q_w_0_BWDt2 = self.problem.dataset.dpl_model.normalize_concepts(q_w_0_BWD)
                _, entropy_B = dpl_model.problog_inference(q_w_0_BWDt2, query=y_0_BY, compute_entropies=True)
                return entropy_B
            # Get the entropy of the distribution over all models 
            # Base log probabilities of all models
            log_probs_MB = dist.log_prob(self.all_assignments_MW.unsqueeze(1).expand(-1, y_0_BY.shape[0], -1)).sum(-1)
            mask_MB = torch.all(y_0_BY == self.all_y_outs_MY[:, None, :], dim=-1)
            # Filter assignments that match target y
            filtered_log_probs_MB = torch.where(mask_MB, log_probs_MB, -float('inf'))
            # Normalise
            log_WMC_B = torch.logsumexp(filtered_log_probs_MB, dim=0)
            cond_log_probs_MB = filtered_log_probs_MB - log_WMC_B[None, :]
            # Compute entropy terms
            entropy_terms_MB = torch.zeros_like(log_probs_MB)
            entropy_terms_MB[mask_MB] = cond_log_probs_MB.exp()[mask_MB] * cond_log_probs_MB[mask_MB]
            # Normalise by number of dimensions of W for scaling consistency
            return -entropy_terms_MB.sum(dim=0) / q_w_0_BWD.shape[-2]
        elif self.args.entropy_variant == "unconditional":
            return dist.entropy().mean(dim=-1)
        raise NotImplementedError(f"Entropy variant {self.args.entropy_variant} not implemented")

    @abstractmethod
    def loss(
        self,
        x_BX: Tensor,
        y_0_BY: Tensor,
        log: TrainingLog,
        w_0_BW: Optional[Tensor] = None,
    ) -> Tensor:
        pass

    def tilde_y0(self, w_0_BW: Tensor, y_t_BY: Tensor) -> Tensor:
        tilde_y_0_BY = self.problem.y_from_w(w_0_BW)

        # For unmasked dimensions in y_t_BY, copy over those values from tilde_y_0_BY
        unmasked_dims = y_t_BY != self.mask_dim_y()
        tilde_y_0_BY[..., unmasked_dims] = y_t_BY[unmasked_dims]
        return tilde_y_0_BY

    def loss_weight(self, losses: Tensor, time: Tensor) -> Tensor:
        # Implements the noising schedule.
        return -losses / time

    def evaluate(
        self,
        x_BX: torch.Tensor,
        y_0_BY: torch.Tensor,
        w_0_BW: Optional[torch.Tensor],
        log: TestLog,
    ) -> torch.Tensor:
        # Returns all predictions of w
        self.eval()
        # Initialize w and y with mask vectors
        w_t_BW = torch.full(
            (x_BX.shape[0],) + self.problem.shape_w()[:-1],
            self.mask_dim_w(),
            device=x_BX.device,
        )
        y_t_BY = torch.full(
            (x_BX.shape[0],) + self.problem.shape_y()[:-1],
            self.mask_dim_y(),
            device=x_BX.device,
        )

        with torch.no_grad():
            # Sample w_0 and y_0
            res = self.sample(
                x_BX,
                w_t_BW,
                y_t_BY,
                self.args.test_L,
                self.args.test_T,
                self.args.test_K,
                only_w=self.args.simple_model,
            )
        if self.args.simple_model:
            hat_w_0_SBW = res
        else:
            hat_w_0_SBW, hat_y_0_SBY = res
            # Compare sampled y prediction to gt
            log.y_acc_avg += (
                self.problem.eval_y(hat_y_0_SBY, y_0_BY, w_0_BW).float().mean().item()
            )
            # Compute majority voting on y (this has to take all dimensions of y into account)
            # TODO: Note that this uses the marginal mode to be backwards compatible. Probably incorrect. 
            hat_y_0_BY = marginal_mode(hat_y_0_SBY, dim=0)
            log.y_acc_top += (
                self.problem.eval_y(hat_y_0_BY, y_0_BY, w_0_BW).float().mean().item()
            )

        result_dict = {}
        result_dict["LABELS"] = y_0_BY
        result_dict["CONCEPTS"] = w_0_BW

        # Compare sampled w to ground truth
        w_accuracy_SBW = (hat_w_0_SBW == w_0_BW).float()
        log.w_acc_avg += w_accuracy_SBW.mean().item()

        pred_options_w, pred_options_y, hat_y_0_SBY = self.all_pred_options(hat_w_0_SBW)

        for pred_type in PRED_TYPES_W:
            w_BW = pred_options_w[pred_type]
            log.pred_types[pred_type] += (w_BW == w_0_BW).float().mean().item()
            result_dict[f"{pred_type}"] = w_BW

        # Evaluate accuracy on y for different prediction options
        for pred_type in PRED_TYPES_Y:
            y_BY = pred_options_y[pred_type]
            log.pred_types[pred_type] += (
                self.problem.eval_y(y_BY, y_0_BY, w_0_BW).float().mean().item()
            )
            result_dict[f"{pred_type}"] = y_BY

        result_dict["W_SAMPLES"] = hat_w_0_SBW
        result_dict["Y_SAMPLES"] = hat_y_0_SBY

        if isinstance(log, BOIATestLog):
            for pred_type in PRED_TYPES_W:
                w_BW = pred_options_w[pred_type]
                log.w_preds[pred_type] = np.concatenate([log.w_preds[pred_type], w_BW.detach().cpu().int().numpy()], axis=0)
            for pred_type in PRED_TYPES_Y:
                y_BY = pred_options_y[pred_type]
                log.y_preds[pred_type] = np.concatenate([log.y_preds[pred_type], y_BY.detach().cpu().int().numpy()], axis=0)
            log.w_targets_B21 = np.concatenate([log.w_targets_B21, w_0_BW.detach().cpu().int().numpy()], axis=0)
            log.y_targets_B3 = np.concatenate([log.y_targets_B3, y_0_BY.detach().cpu().int().numpy()], axis=0)
        
        return result_dict

    def all_pred_options(self, hat_w_0_SBW: Tensor) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor]:
        """
        Returns all possible methods for predicting a y from samples of w.
        """

        # TODO

        # # Marginal mode then f
        # hat_w_0_MM_BW = marginal_mode(hat_w_0_SBW)
        # print(hat_w_0_MM_BW.shape)
        # hat_y_0_MMf_BY = self.problem.y_from_w(hat_w_0_MM_BW)

        hat_w_0_MM_BW = marginal_mode(hat_w_0_SBW) # [16, 2]
        hat_w_0_MM_BW_one_hot = torch.nn.functional.one_hot(hat_w_0_MM_BW, num_classes=self.problem.shape_w()[-1]).float() # [16, 2, 10]
        # hat_w_0_MM_BW_one_hot = hat_w_0_MM_BW_one_hot.view(hat_w_0_MM_BW_one_hot.shape[0], -1).float() # [16, 20]
        hat_w_0_MM_BW_one_hot = (hat_w_0_MM_BW_one_hot[:, 0, :].unsqueeze(-1) @ hat_w_0_MM_BW_one_hot[:, 1, :].unsqueeze(-2)).view(hat_w_0_MM_BW_one_hot.shape[0], -1) # [16, 100]
        pty_0_SBY = torch.nn.functional.softmax(self.inference_layer(hat_w_0_MM_BW_one_hot), dim=-1) # [16, 19]
        ty_0_SBY = torch.argmax(pty_0_SBY, dim=-1) # [16]
        hat_y_0_MMf_BY = int_to_digit_tensor(ty_0_SBY, self.problem.out_digits)

        # True mode then f
        hat_w_0_TM_BW = true_mode(hat_w_0_SBW)
        hat_w_0_TM_BW_one_hot = torch.nn.functional.one_hot(hat_w_0_TM_BW, num_classes=self.problem.shape_w()[-1]).float() # [16, 2, 10]
        # hat_w_0_TM_BW_one_hot = hat_w_0_TM_BW_one_hot.view(hat_w_0_TM_BW_one_hot.shape[0], -1).float() # [16, 20]
        hat_w_0_TM_BW_one_hot = (hat_w_0_TM_BW_one_hot[:, 0, :].unsqueeze(-1) @ hat_w_0_TM_BW_one_hot[:, 1, :].unsqueeze(-2)).view(hat_w_0_TM_BW_one_hot.shape[0], -1) # [16, 100]
        pty_0_SBY1 = torch.nn.functional.softmax(self.inference_layer(hat_w_0_TM_BW_one_hot), dim=-1) # [16, 19]
        ty_0_SBY1 = torch.argmax(pty_0_SBY1, dim=-1) # [16]
        hat_y_0_TMf_BY = int_to_digit_tensor(ty_0_SBY1, self.problem.out_digits)

        # print(hat_w_0_MM_BW.shape)
        # hat_y_0_TMf_BY = self.problem.y_from_w(hat_w_0_TM_BW)

        hat_w_0_SBW_one_hot = torch.nn.functional.one_hot(hat_w_0_SBW, num_classes=self.problem.shape_w()[-1]).float() # [8, 16, 2, 10]
        # hat_w_0_SBW_one_hot = hat_w_0_SBW_one_hot.view(hat_w_0_SBW_one_hot.shape[0], hat_w_0_SBW_one_hot.shape[1], -1).float() # [8, 16, 20]
        hat_w_0_SBW_one_hot = (hat_w_0_SBW_one_hot[:, :, 0, :].unsqueeze(-1) @ hat_w_0_SBW_one_hot[:, :, 1, :].unsqueeze(-2)).view(hat_w_0_SBW_one_hot.shape[0], hat_w_0_SBW_one_hot.shape[1], -1) # [8, 16, 100]
        pty_0_SBY2 = torch.nn.functional.softmax(self.inference_layer(hat_w_0_SBW_one_hot), dim=-1) # [8, 16, 19]
        ty_0_SBY2 = torch.argmax(pty_0_SBY2, dim=-1) # [8, 16]
        hat_y_0_SBY = int_to_digit_tensor(ty_0_SBY2, self.problem.out_digits)

        # hat_y_0_SBY = self.problem.y_from_w(hat_w_0_SBW)
        # f then marginal mode (This was the default in most eval)
        hat_y_0_fMM_BY = marginal_mode(hat_y_0_SBY, dim=0)

        # f then true mode
        hat_y_0_fTM_BY = true_mode(hat_y_0_SBY)

        return {
            "w_MM": hat_w_0_MM_BW,
            "w_TM": hat_w_0_TM_BW,
        }, {
            "y_MMf": hat_y_0_MMf_BY,
            "y_TMf": hat_y_0_TMf_BY,
            "y_fMM": hat_y_0_fMM_BY,
            "y_fTM": hat_y_0_fTM_BY,
        }, hat_y_0_SBY


    def mask_dim_w(self) -> int:
        return self.problem.shape_w()[-1]

    def mask_dim_y(self) -> int:
        return self.problem.shape_y()[-1]
