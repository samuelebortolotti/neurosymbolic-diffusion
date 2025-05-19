from abc import ABC
from typing import Callable, Optional, Tuple

import numpy as np

from expressive.args import AbsArguments
from expressive.methods.base_model import BaseNeSyDiffusion, Problem
from expressive.models.diffusion_model import UnmaskingModel, ForwardAbsorbing
from expressive.util import safe_reward, safe_sample_categorical
import torch
from torch import Tensor
from torch.nn.functional import one_hot

from torch.distributions import Categorical

from expressive.methods.logger import TrainingLog


class CondNeSyDiffusion(BaseNeSyDiffusion):
    """This model adds conditioning to y when generating w, essentially interleaving their computation. 
    """

    def __init__(
        self,
        p: UnmaskingModel,
        problem: Problem,
        args: AbsArguments,
    ):
        super().__init__(p, problem, args)
        self.q_y = ForwardAbsorbing(self.problem.shape_y()[-1])

    def rloo_loss(
        self,
        log_probs_SpB: Tensor,
        reward_SpBD: Tensor,
        unnorm_reward_SpBD: Tensor,
        condition_non_w0_SB: Tensor,
        masked_dims_BD: Tensor,
        U=50.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Conditional reinforce with leave-one-out (rloo) loss. Always adds w^0 to the set of samples, renormalising.
        log_probs: predicted log probabilities for each action taken
        rewards: resulting rewards (negative loss)
        Condition: Needs to take into account both the y^t constraint and the tw^(0, s) != w^0 constraint
        """

        # Combine condition and constraint
        reward_SBD = reward_SpBD[:-1]
        unnorm_reward_SBD = unnorm_reward_SpBD[:-1]
        filtered_reward_SBD = reward_SBD * condition_non_w0_SB[..., None]
        filtered_unnorm_reward_SBD = unnorm_reward_SBD * condition_non_w0_SB[..., None]
        log_probs_SB = log_probs_SpB[:-1]

        # Compute number of samples different from w0
        n_non_w0_B = condition_non_w0_SB.sum(0)
        n_non_w0_BD = n_non_w0_B[:, None].expand(-1, masked_dims_BD.shape[1])

        any_meet_constraints_BD = torch.any(filtered_reward_SBD > 0, dim=0)

        # Compute loo baselines, but only for dimensions where at least two samples are different from w0 (otherwise the baseline is not defined)
        mask_SBD = (
            torch.logical_and(n_non_w0_BD >= 2, any_meet_constraints_BD)
            .unsqueeze(0)
            .expand(reward_SBD.shape)
        )
        numerator_b_SBD = torch.sum(filtered_reward_SBD, dim=0) - reward_SBD
        denominator_b_BD = (n_non_w0_BD - 1).unsqueeze(0).expand(reward_SBD.shape)
        b_SBD = torch.zeros_like(reward_SBD)
        b_SBD[mask_SBD] = numerator_b_SBD[mask_SBD] / denominator_b_BD[mask_SBD]

        # Compute RLOO gradient. If all samples are equal to w0, just set this term to 0.
        # Filter contributions when tw^(0, s)_i != w^(0)_i and when no samples meet the constraints
        filter_BD = torch.logical_and(n_non_w0_BD >= 1, any_meet_constraints_BD)
        numerator_rho_BD = torch.sum(
            (condition_non_w0_SB * log_probs_SB)[..., None]
            * (reward_SBD - b_SBD).detach(),
            dim=0,
        )
        rho_samples_BD = torch.zeros_like(masked_dims_BD, dtype=torch.float32)
        rho_samples_BD[filter_BD] = numerator_rho_BD[filter_BD] / n_non_w0_BD[filter_BD]

        # Compute average rewards. Initialize with 1 to avoid division by zero when conditions is not met for any sample
        numerator_mu_BD = filtered_reward_SBD.sum(dim=0)
        mu_samples_BD = torch.ones_like(n_non_w0_BD, dtype=torch.float32)
        mu_samples_BD[filter_BD] = numerator_mu_BD[filter_BD] / n_non_w0_BD[filter_BD]

        # Compute unbiased log-expectation for monitoring purposes:
        numerator_E_BD = filtered_unnorm_reward_SBD.sum(dim=0)
        E_BD = torch.zeros_like(n_non_w0_BD, dtype=torch.float32)
        E_BD[filter_BD] = numerator_E_BD[filter_BD] / n_non_w0_BD[filter_BD]

        # Combine with probabilities of w0
        # To ensure gradients propagate when the exp of log probabilities underflows, we use a trick described in the
        #  appendix where exp(log(x)) is replaced by exp((-50 - log(x).detach() + log(x)) that passes exp(-U) in the
        #  forward pass while propagating gradients into log(x).
        probs_w0_B1 = torch.exp(
            torch.maximum(
                log_probs_SpB[-1], (-U - log_probs_SpB[-1]).detach() + log_probs_SpB[-1]
            )
        )[:, None]
        reward_w0_BD = reward_SpBD[-1]

        # Add contributions from w0. Remove contributions from unmasked dimensions
        rho_BD = torch.where(
            masked_dims_BD,
            rho_samples_BD + probs_w0_B1 * (reward_w0_BD - rho_samples_BD),
            torch.zeros_like(rho_samples_BD),
        )
        mu_BD = torch.where(
            masked_dims_BD,
            mu_samples_BD + probs_w0_B1 * (reward_w0_BD - mu_samples_BD),
            torch.ones_like(mu_samples_BD),
        )

        E_BD = torch.where(
            masked_dims_BD,
            E_BD + probs_w0_B1 * (unnorm_reward_SpBD[-1] - E_BD),
            torch.ones_like(E_BD),
        )

        # Hacky way to get 1 over average as extra multiplicative term in the loss.
        #  I got this from https://github.com/ML-KULeuven/catlog/blob/main/ADDITION/addition.py#L6
        #  The idea here is that you want to get an estimate of the gradient of - log WMC, which (when differentiated)
        #  should be 1 / WMC * dWMC/dtheta. This is a way to get that 1 / WMC term (or rather an estimate of it)
        #  And also means that the loss will be an interpretable term (namely an estimate of semantic loss).
        epsilon = torch.tensor(1e-8, device=mu_BD.device)
        L_BD = torch.log((mu_BD - rho_BD).detach() + rho_BD + epsilon)

        log_E_BD = torch.log(E_BD + epsilon).detach()

        # TODO: Are these assertions meaningful?
        # They interact realy annoyingly with the w0 contribution term, so I've commented them out for now
        # Compute log(detach(exp(log_mu_BD) - exp(log_rho_BD))
        # Assertion: If a dimension is unmasked, the loss should be zero (we filter this case out)
        assert torch.allclose(
            L_BD[~masked_dims_BD], torch.tensor(0.0, device=L_BD.device)
        )

        # # Assertion: On masked dimensions, if all samples are equal to w0, the loss should just consider the w0 contribution
        # assert torch.allclose(L_BD[torch.logical_and(masked_dims_BD, n_non_w0_BD == 0)], torch.log(epsilon + probs_w0_B1 * reward_w0_BD))

        # # Assertion: If a dimension is masked and some samples are accepted, but no samples meet the constraint, the loss should be log epsilon + w0 contribution
        # any_samples_meet_constraints = torch.any(filter_SBD, dim=0)
        # assert torch.allclose(L_BD[
        #                       torch.logical_and(~any_samples_meet_constraints, n_non_w0_BD > 0)],
        #                       torch.log(epsilon + probs_w0_B1 * reward_w0_BD))

        return L_BD, log_E_BD

    def loss(
        self, x_BX: Tensor, y_0_BY: Tensor, log: TrainingLog, eval_w_0_BW: Optional[Tensor] = None
    ) -> Tensor:
        """
        Shapes legend:
        - B: Batch size
        - X: Input size
        - Y: number of RVs for y
        - W: number of RVs for w
        - d: number of values for y or w
        - D: number of values for y or w + 1 (mask dim)
        - S: number of samples of \tilde{w_0} (=S)
        """
        self.train()
        # initialize embedding of x
        encoding_BWE = self.p.encode_x(x_BX)

        # Create mask matrix bm (a matrix full of the number of values possible for w (with 0 indexing))
        s_w = y_0_BY.shape[:-1] + self.problem.shape_w()
        bm_BW = torch.ones(s_w[:-1], device=x_BX.device, dtype=torch.long) * s_w[-1]

        # initialize q(w_0|x, y_0)
        q_w_0_BWD = self.p.distribution(
            (bm_BW, y_0_BY),
            encoding_BWE,
            torch.zeros_like(y_0_BY[..., 0], device=bm_BW.device),
        )

        # TODO: H (entropy)
        # Sample w_0
        w_1_BW = torch.full(
            (x_BX.shape[0],) + self.problem.shape_w()[:-1],
            self.mask_dim_w(),
            device=x_BX.device,
        )
        var_w_0_BW = self.sample(
            x_BX,
            w_1_BW,
            y_0_BY,
            self.args.variational_K,
            self.args.variational_T,
            self.args.variational_K,
            encoding_BWE,
        )[0][0]
        w_0_BWD = one_hot(var_w_0_BW, s_w[-1] + 1).float()

        # Sample timesteps
        t = torch.rand((x_BX.shape[0],), device=x_BX.device)

        # Compute q(y_t | y_0) and q(w_t | w_0)
        y_0_BYD = one_hot(y_0_BY, self.problem.shape_y()[-1] + 1).float()
        q_y_t_BYD: Tensor = self.q_y.t_step(y_0_BYD, t)
        q_w_t_BWD: Tensor = self.q_w.t_step(w_0_BWD, t)

        # Sample from both
        y_t_BY = safe_sample_categorical(Categorical(probs=q_y_t_BYD))
        w_t_BW = safe_sample_categorical(Categorical(probs=q_w_t_BWD))

        # Compute p(\tilde{w}_0|y_t, w_t x)
        p_w_0_BWD = self.p.distribution((w_t_BW, y_t_BY), encoding_BWE, t)

        # Sample S_2 values for \tilde{w}_0
        tw_0 = Categorical(probs=p_w_0_BWD[..., :-1])
        tw_0_SBW = safe_sample_categorical(tw_0, (self.args.loss_S,))

        tw_0_SpBW = torch.cat([tw_0_SBW, var_w_0_BW[None, :, :]], dim=0)

        # Compute deterministic function for tw_0, with carry-over unmasking on y_t_BY
        ty_0_SpBY = self.problem.y_from_w(tw_0_SpBW)

        #####################
        # LOSS FUNCTIONS
        #####################
        # Compute log probs of tw_0
        log_probs_SpB = tw_0.log_prob(tw_0_SpBW).sum(-1)

        # Check if y^(t)_i is unmasked
        masked_dims_BY = y_t_BY == self.problem.shape_y()[-1]

        # Compute y^(t) violations on the unmasked values
        violations_y_t_SpBY = (1 - masked_dims_BY.float()) * (
            ty_0_SpBY != y_t_BY[None, :, :]
        )
        # Compute exponentiated reward for reward in the RLOO loss
        reward_y_t_SpB, norm_reward_y_t_SpB = safe_reward(
            violations_y_t_SpBY, beta=self.args.beta
        )

        # Condition that samples need to be different from w0
        constraint_w0_SB = ~torch.all(tw_0_SBW == var_w_0_BW, dim=-1)

        #####################
        # w RLOO denoising loss
        #####################
        # Compute conditional RLOOs on each w_0_i dimension
        # Check if tw^{0, s}_i == w^0_i
        # If w^(t)_i is unmasked, then by carry-over unmasking, tw^{0, s}_i == w^0_i
        constraint_w_0_SpBW = (var_w_0_BW[None, :, :] == tw_0_SpBW).float()
        reward_w_0_SpBW = constraint_w_0_SpBW * reward_y_t_SpB[:, :, None]
        norm_reward_w_0_SpBW = constraint_w_0_SpBW * norm_reward_y_t_SpB[:, :, None]
        # Check if w^(t)_i is unmasked
        masked_dims_BW = w_t_BW == self.mask_dim_w()

        #####################
        # y RLOO denoising loss
        #####################
        # Compute y^(t) constraint. No carry-over unmasking here, so only measure this on the unmasked values
        constraint_y0_SpBY = torch.logical_or(
            y_0_BY[None, :, :] == ty_0_SpBY, ~masked_dims_BY[None, :, :]
        ).float()
        reward_y_0_SpBY = constraint_y0_SpBY * reward_y_t_SpB[:, :, None]
        norm_reward_y_0_SpBY = constraint_y0_SpBY * norm_reward_y_t_SpB[:, :, None]

        # Concatenate constraints
        reward_cat_SpBDp1 = torch.cat(
            [reward_w_0_SpBW, reward_y_0_SpBY, reward_y_t_SpB.unsqueeze(-1)], dim=2
        )
        norm_reward_cat_SpBDp1 = torch.cat(
            [
                norm_reward_w_0_SpBW,
                norm_reward_y_0_SpBY,
                norm_reward_y_t_SpB.unsqueeze(-1),
            ],
            dim=2,
        )
        masked_dims_cat_BDp1 = torch.cat(
            [
                masked_dims_BW,
                masked_dims_BY,
                torch.ones(
                    device=x_BX.device, dtype=torch.bool, size=(x_BX.shape[0], 1)
                ),
            ],
            dim=1,
        )

        # Compute the RLOO loss for all constraints. Uses LOO with w0 injection
        L_denoising_BDp1, log_E_BDp1 = self.rloo_loss(
            log_probs_SpB,
            reward_cat_SpBDp1,
            norm_reward_cat_SpBDp1,
            constraint_w0_SB,
            masked_dims_cat_BDp1,
        )
        assert (log_E_BDp1 <= 0).all()
        assert torch.logical_and(
            0 <= norm_reward_cat_SpBDp1, norm_reward_cat_SpBDp1 <= 1
        ).all()

        # Take the mean over the dimensions of w and y
        dim_W = masked_dims_BW.shape[1]
        L_w_denoising = self.loss_weight(L_denoising_BDp1[:, :dim_W].mean(-1), t).mean()
        L_y_denoising = self.loss_weight(
            L_denoising_BDp1[:, dim_W:-1].mean(-1), t
        ).mean()

        # For the Z loss, it should technically be the mean over all dimensions. Note that this one is _positive_
        L_Z = (
            -self.loss_weight(L_denoising_BDp1[:, -1], t)
            * (masked_dims_BW.sum(-1) + masked_dims_BY.sum(-1))
            / (L_denoising_BDp1.shape[-1] - 1)
        ).mean()

        log_E_w_denoising = self.loss_weight(log_E_BDp1[:, :dim_W].mean(-1), t).mean()
        log_E_y_denoising = self.loss_weight(log_E_BDp1[:, dim_W:-1].mean(-1), t).mean()
        log_Z = (
            self.loss_weight(log_E_BDp1[:, -1], t)
            * (masked_dims_BW.sum(-1) + masked_dims_BY.sum(-1))
            / (log_E_BDp1.shape[-1] - 1)
        ).mean()

        # Negative entropy on variational distribution
        q_entropy: Tensor = self.entropy_loss(y_0_BY, q_w_0_BWD).mean()

        var_y_0_BY = self.problem.y_from_w(var_w_0_BW)
        var_violations_y_0_BY = var_y_0_BY != y_0_BY

        if var_w_0_BW is not None:
            log.var_accuracy_w += (var_w_0_BW == eval_w_0_BW).float().mean().item()
            log.w_preds = np.concatenate([log.w_preds, var_w_0_BW.flatten().detach().cpu().int().numpy()])
            log.w_targets = np.concatenate([log.w_targets, eval_w_0_BW.flatten().detach().cpu().int().numpy()])

        log.var_entropy += q_entropy.item()
        log.unmasking_entropy += tw_0.entropy().mean().item()
        log.w_denoise += log_E_w_denoising.item()
        log.y_denoise += log_E_y_denoising.item()
        log.Z_loss += log_Z.item()
        log.avg_violation += violations_y_t_SpBY.mean().item()
        log.avg_constraints += constraint_y0_SpBY.float().mean().item()
        log.avg_var_violations += var_violations_y_0_BY.float().mean().item()
        log.var_accuracy_y += torch.min(~var_violations_y_0_BY, dim=-1)[0].float().mean().item()
        
        # TODO: Do we need to readd these?
        # These conditions can currently be violated due to the use of log-expectations.
        # if not (torch.all(L_w_denoising_B >= 0) and torch.all(L_y_denoising_B >= 0) and q_entropy >= 0):
        #     print(f"L_w_denoising_B: {L_w_denoising_B}")
        #     print(f"L_y_denoising_B: {L_y_denoising_B}")
        #     print(f"q_entropy: {q_entropy}")
        #     print("WARNING: Loss is negative")

        return (
            L_y_denoising
            + self.args.w_denoise_weight * L_w_denoising
            + self.args.Z_weight * L_Z
            - self.args.entropy_weight * q_entropy
        )