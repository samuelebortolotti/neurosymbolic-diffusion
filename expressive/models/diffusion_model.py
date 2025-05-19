from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, List, Generic, TypeVar, Callable, Optional, Union
import torch
from torch import nn, Tensor
from torch.nn.functional import one_hot
from torch.nn.modules import Linear

from expressive.args import AbsArguments, Arguments
import math

from expressive.models.dit import create_DiT, hidden_size

DATA = TypeVar("DATA")
WY_DATA = Tuple[Tensor, Tensor]
TimeSteps = Optional[Union[List[int], int]]


class ForwardModel(nn.Module, Generic[DATA], ABC):
    T: int

    # Note: The t given here represents the list of timesteps, indexing the 2nd dimension in x
    @abstractmethod
    def one_step(self, x_prev: DATA, t: TimeSteps = None) -> DATA:
        pass

    @abstractmethod
    def t_step(self, x_0: DATA, t: TimeSteps = None) -> DATA:
        pass

    @abstractmethod
    def cond_jump(
        self, x_0: DATA, x_t: DATA, t: TimeSteps = None, s: TimeSteps = None
    ) -> DATA:
        pass


class ForwardAbsorbing(nn.Module):
    def __init__(self, dimension: int):
        # Assuming linear absorbing model for now
        # Assuming continuous-time
        super().__init__()
        self.dimension = dimension
        # TODO: If vocab sizes get large, there's probably a much more memory efficient way to implement this
        #  Instead of storing the full matrix, just store the probability of transitioning to the masking state

    def t_step(self, x_0_SBXD: Tensor, t_SB: Tensor) -> Tensor:
        if (t_SB <= 0).any() or (t_SB > 1).any():
            print('--------------------------------')
            print("WARNING: t_SB out of range")
            print("value of t_SB:", t_SB)
            print('--------------------------------')
        assert t_SB.ndim == x_0_SBXD.ndim - 2

        if x_0_SBXD.shape[-1] == self.dimension:
            # Add a column of zeros to the end if not already added
            x_0_SBXD = torch.cat(
                [x_0_SBXD, torch.zeros(x_0_SBXD.shape[:-1] + (1,))], dim=-1
            )
        x_t_SBXD = x_0_SBXD.clone()

        # Probability of transitioning to mask
        x_t_SBXD[..., -1] = t_SB[..., None]
        # Probability of keeping the current value
        x_t_SBXD[..., :-1] = x_t_SBXD[..., :-1] * (1 - t_SB[..., None, None])
        return x_t_SBXD

    def cond_jump(
        self, x_0: Tensor, x_t: Tensor, t: Union[float, Tensor], s: Union[float, Tensor]
    ) -> Tensor:
        if isinstance(t, float):
            assert 0 < s < t <= 1
        else:
            assert isinstance(t, Tensor)
            assert isinstance(s, Tensor)
            assert (0 < s).all() and (s < t).all() and (t <= 1).all()

        x_s = x_t.clone()

        # Set all values where x_t is masked to a mix of the value in x_0 and the mask
        mask = [x_t == self.dimension]
        x_s[mask] = (s * x_t[mask] + (t - s) * x_0[mask]) / t
        return x_s


class ForwardDiscrete(ForwardModel[Tensor]):
    def __init__(self, Q: List[Tensor]):
        # Q is a list of transition matrices. Also provide a matrix for t=0 (which is the identity matrix)
        # So the length of both should be T+1
        super().__init__()
        self.T = len(Q) - 1
        self.register_buffer("Q", torch.stack(Q, dim=0))
        self.Kw = Q[0].shape[-1]
        lineQ = [Q[0]]

        # Cache \overline{Q}_t = Q_0 Q_1 ... Q_t
        for i in range(1, len(Q)):
            lineQ.append(torch.matmul(lineQ[-1], Q[i]))
        self.register_buffer("lineQ", torch.stack(lineQ, dim=0))

    def _matvecmul(self, Q_TKK: Tensor, x_BTNK: Tensor) -> torch.Tensor:
        x_BTN1K = x_BTNK.unsqueeze(-2)
        Q_1T1KK = Q_TKK.unsqueeze(0).unsqueeze(2)
        return torch.matmul(x_BTN1K, Q_1T1KK).squeeze(-2)

    def _getQ_lineQ(self, t: TimeSteps) -> Tuple[Tensor, Tensor]:
        if t is None:
            return self.Q, self.lineQ
        if isinstance(t, int):
            t = [t]
        assert all([self.T >= _t >= 0 for _t in t])
        return self.Q[t], self.lineQ[t]

    def one_step(self, x_prev: Tensor, t: TimeSteps = None) -> Tensor:
        """
        :param t: The current time step
        :param x: The current x
        :return: The next x
        """
        Q, _ = self._getQ_lineQ(t)
        return self._matvecmul(Q, x_prev)

    def t_step(self, x_0: Tensor, t: TimeSteps = None) -> Tensor:
        """
        :param x_0: The w at timestep 0
        :param t: The current time step(s)
        :return: The distribution over x at timestep t
        """
        _, lineQ = self._getQ_lineQ(t)
        return self._matvecmul(lineQ, x_0)

    def cond_jump(
        self,
        x_0_BTNK: Tensor,
        x_t_BTNK: Tensor,
        t: TimeSteps = None,
        s: TimeSteps = None,
    ) -> Tensor:
        """
        :param t: The current time step
        :param s: The timestep to jump to
        :param x_0: The x at timestep 0
        :param x_t: The x at timstep t
        :return: The distribution over x at timestep s

        For this equation, consult Eq 3 in the D3PM paper.
        """
        for i, _t in enumerate(t):
            assert _t > s[i]

        Qt, lineQt = self._getQ_lineQ(t)
        _, lineQs = self._getQ_lineQ(s)
        # Adjust for 0-indexing of matrices

        nom_BTXK = self._matvecmul(Qt.mT, x_t_BTNK) * self._matvecmul(lineQs, x_0_BTNK)
        pre1_BTXK = self._matvecmul(lineQt, x_0_BTNK).unsqueeze(-2)
        denom_BTX1 = torch.matmul(pre1_BTXK, x_t_BTNK.unsqueeze(-1)).squeeze(-1)
        x_s_BTNK = nom_BTXK / denom_BTX1

        # Ensure timestep 0 is kept unchanged
        mask = [_s == 0 for _s in s]
        x_s_BTNK[..., mask, :, :] = x_0_BTNK[..., mask, :, :]
        return x_s_BTNK


class DoubleForwardModel(ForwardModel[WY_DATA]):
    def __init__(self, fw: ForwardModel[Tensor], fy: ForwardModel[Tensor]):
        assert fw.T == fy.T
        super().__init__()
        self.T = fw.T
        self.fw = fw
        self.fy = fy

    def one_step(self, x_prev: WY_DATA, t: TimeSteps = None) -> WY_DATA:
        return self.fw.one_step(x_prev[0], t), self.fy.one_step(x_prev[1], t)

    def t_step(self, x_0: WY_DATA, t: TimeSteps = None) -> WY_DATA:
        return self.fw.t_step(x_0[0], t), self.fy.t_step(x_0[1], t)

    def cond_jump(
        self, x_0: WY_DATA, x_t: WY_DATA, t: TimeSteps = None, s: TimeSteps = None
    ) -> WY_DATA:
        return self.fw.cond_jump(x_0[0], x_t[0], t, s), self.fy.cond_jump(
            x_0[1], x_t[1], t, s
        )


class ForwardUniform(ForwardDiscrete):
    # Uniform noising using schedule from Hoogeboom 2021 https://arxiv.org/pdf/2102.05379.pdf , Appendix B
    def __init__(self, K: int, args: Arguments):
        schedule = []

        def _f_cos(t: int) -> float:
            return math.cos(
                math.pi
                / 2
                * (t / args.T + args.cosine_schedule_s)
                / (1 + args.cosine_schedule_s)
            )

        for i in range(args.T + 1):
            schedule.append(_f_cos(i) / _f_cos(0))

        Q = [
            torch.eye(K) * schedule[i] + torch.ones(K, K) * (1 - schedule[i]) / K
            for i in range(args.T + 1)
        ]

        super().__init__(Q)


class ForwardUniformVariational(ForwardUniform):
    def __init__(self, t0_model, K: int, args: Arguments):
        super().__init__(K, args)
        self.K = K
        self.args = args
        self.t0_model = t0_model

    def predict_t0(self, x: Tensor, y_0: Tensor, encoding: torch.Tensor) -> Tensor:
        return self.t0_model(x, y_0, encoding)


class DoubleForwardUniform(DoubleForwardModel):
    def __init__(self, w_0_model: nn.Module, K_W: int, K_Y: int, args: Arguments):
        super().__init__(ForwardUniform(K_W, args), ForwardUniform(K_Y, args))
        self.w_0_model = w_0_model

    def predict_w_0(self, x: Tensor, y_0: Tensor, encoding: Tensor) -> Tensor:
        return self.w_0_model(x, y_0, encoding)


class ReverseModel(ABC, nn.Module, Generic[DATA]):
    @abstractmethod
    def distribution(
        self,
        x_t: DATA,
        encoding: Tensor,
        t: TimeSteps = None,
        s: TimeSteps = None,
    ) -> DATA:
        pass

    @abstractmethod
    # Just encodes the input
    def encode(self, x: Tensor) -> Tensor:
        pass


class JumpReverseModel(ReverseModel, ABC, Generic[DATA]):
    def __init__(self, q: ForwardModel[DATA]):
        super().__init__()
        self.q: ForwardModel[DATA] = q

    def distribution(
        self,
        x_t: DATA,
        encoding: Tensor,
        t: TimeSteps = None,
        s: TimeSteps = None,
    ) -> DATA:
        _px0 = self._distribution(x_t, encoding, t)
        return self.q.cond_jump(_px0, x_t, t, s)

    @abstractmethod
    # p_t(w_0, y_0 | x, y_t, w_t)
    def _distribution(self, x: DATA, encoding: Tensor, t: TimeSteps = None) -> DATA:
        pass


class EmbeddingLayer(nn.Module):
    """
    Taken from https://github.com/kuleshov-group/mdlm/blob/master/models/dit.py#L292
    """

    def __init__(self, dim: int, vocab_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class LayerNorm(nn.Module):
    """
    https://github.com/kuleshov-group/mdlm/blob/master/models/dit.py#L126
    """

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.nn.functional.layer_norm(x.float(), [self.dim])
        return x * self.weight


@torch.jit.script
def modulate_fused(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale) + shift


class DDitFinalLayer(nn.Module):
    # From https://github.com/kuleshov-group/mdlm/blob/master/models/dit.py#L302
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c)[..., None, :].chunk(2, dim=-1)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class UnmaskingModel(nn.Module, ABC):
    def __init__(
        self, vocab_dim: int, w_dims: int, seq_length: int, args: AbsArguments
    ) -> None:
        """
        :param vocab_dim: Number of possible values for W (and Y, if using), **excluding** mask dimension
        :param w_dims: The first dimensions in the vocabulary are the possible predictions for w. This should _not_ include the mask dim
        :param seq_length: Length of seq expected
        """
        super().__init__()
        self.args = args
        self.vocab_dim = vocab_dim
        self.w_dims = w_dims
        self.seq_length = seq_length

        if args.model.startswith("DiT") or args.model.startswith("mlp"):
            embed_size = hidden_size(args.model)
            self.vocab_embed = EmbeddingLayer(embed_size, vocab_dim + 1)
            self.encoding_combiner = Linear(embed_size * 2, embed_size)

            if args.model.startswith("DiT"):
                self.model = create_DiT(seq_length, args.model, **args.as_dict())
                self.output_layer = DDitFinalLayer(embed_size, vocab_dim, embed_size)

                # Final layers
                _ = self.output_layer.adaLN_modulation.weight.data.zero_()
                _ = self.output_layer.adaLN_modulation.bias.data.zero_()
                _ = self.output_layer.linear.weight.data.zero_()
                _ = self.output_layer.linear.bias.data.zero_()
            elif args.model.startswith("mlp"):
                self.seq_encoder = Linear(seq_length * embed_size, embed_size)
                self.hidden_layer = Linear(2 * embed_size, embed_size)
                self.output_layer = Linear(embed_size, vocab_dim)

    # @torch.compile
    def distribution(self, wy_t: WY_DATA, x_encoding: Tensor, t: Tensor) -> Tensor:
        if self.args.simple_model:
            w_SBW = wy_t
        else:
            w_SBW = wy_t[0]

        # Compute the logits for the distribution over w_0
        l_w0_SBWd = self.logits_t0(wy_t, x_encoding, t)

        # Softmax to get distribution
        p_w0_SBWd = torch.softmax(l_w0_SBWd, dim=-1)

        # Zero masking probabilities https://arxiv.org/pdf/2406.07524 sets masking dimension probability to 0
        mask_zeros_SBW1 = torch.zeros_like(p_w0_SBWd[..., :1])
        p_w0_SBWD = torch.cat([p_w0_SBWd, mask_zeros_SBW1], dim=-1)

        # Carry-over unmasking https://arxiv.org/pdf/2406.07524 sets probabilities of unmasked w_t to 1
        p_w0_SBWD[w_SBW != self.vocab_dim] = one_hot(
            w_SBW[w_SBW != self.vocab_dim], self.vocab_dim + 1
        ).float()
        return p_w0_SBWD

    def encode_x(self, x: Tensor) -> Tensor:
        # Can be used if the encoding of x remains the same throughout the decoding process, saves computation
        return x

    def logits_t0(self, wy_t: WY_DATA, x_encoding: Tensor, t: Tensor) -> Tensor:
        # Encode the sequence together with the embedding of x
        wy = self.encode_seq(wy_t, x_encoding)
        if self.args.model.startswith("DiT"):
            # Run the diffusion transformer over the sequence
            x_SBXE, t_SBE = self.model(wy, t)

            # Predict distribution over output symbols from embeddings
            return self.output_layer(x_SBXE[..., : self.w_dims, :], t_SBE)
        if self.args.model.startswith("mlp"):
            # Run MLP over sequence
            x_SBE = self.seq_encoder(wy.view(wy.shape[:-2] + (-1,)))

            # Combine the encoding of w at each timestep with the encoding of the sequence
            w_SBWE = wy[..., : self.w_dims, :]
            w_SBWE = torch.cat([w_SBWE, x_SBE.unsqueeze(-2).expand_as(w_SBWE)], dim=-1)
            w_SBWE = self.hidden_layer(w_SBWE)
            w_SBWE = torch.relu(w_SBWE)
            return self.output_layer(w_SBWE)
        raise ValueError(f"Model {self.args.model} not supported")

    def encode_seq(self, seq_BX: WY_DATA, e_BWE: Tensor) -> Tensor:
        e_SBWE = e_BWE
        if self.args.simple_model:
            for _ in range(seq_BX.dim() - 3):
                e_SBWE = e_SBWE.unsqueeze(0)
            e_SBWE = e_SBWE.expand(seq_BX.shape[:-2] + (-1, -1, -1))

            w_SBWE: Tensor = self.vocab_embed(seq_BX)

            # Combine the encoding of x at each timestep with the encoding of the sequence
            w_SBWE[..., : self.w_dims, :] = self.encoding_combiner(
                torch.cat([e_SBWE, w_SBWE], dim=-1)
            )

            return w_SBWE
        else:
            for _ in range(seq_BX[0].dim() - 3):
                e_SBWE = e_SBWE.unsqueeze(0)
            e_SBWE = e_SBWE.expand(seq_BX[0].shape[:-2] + (-1, -1, -1))

            # This assumes W and Y have the same dimensions, would need an override for the general case or sth
            wy_SBXK = torch.cat([seq_BX[0], seq_BX[1]], dim=-1)
            wy_SBXE: Tensor = self.vocab_embed(wy_SBXK)

            # Combine the encoding of x at each timestep with the encoding of the sequence
            wy_SBXE[..., : self.w_dims, :] = self.encoding_combiner(
                torch.cat([e_SBWE, wy_SBXE[..., : self.w_dims, :]], dim=-1)
            )

            return wy_SBXE
