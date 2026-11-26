"""DuETT: Dual Event Time Transformer for electronic health records.

Authors:
    Hritish Mahajan (NetID: hritish2, hritish2@illinois.edu)
    Alex Jung (NetID: jhjung2, jhjung2@illinois.edu)

Paper:
    Alex Labach, Aslesha Pokhrel, Xiao Shi Huang, Saba Zuberi, Seung Eun Yi,
    Maksims Volkovs, Tomi Poutanen and Rahul G. Krishnan. "DuETT: Dual Event
    Time Transformer for Electronic Health Records." Machine Learning for
    Healthcare (MLHC) 2023, PMLR 219:403-422.
    https://proceedings.mlr.press/v219/labach23a.html
    (arXiv: https://arxiv.org/abs/2304.13017)

Description:
    DuETT bins a patient's multivariate time series into a fixed grid of
    time bins x event types, where each cell holds the last observed value
    and the number of observations. Transformer layers alternate between
    attending over the event axis and over the time axis, so the cost grows
    with ``n_time_bins ** 2 + n_events ** 2`` instead of with the number of
    measurements. The model is pre-trained by masking one whole time bin and
    one whole event type per sample and predicting their values and whether
    each value was observed, then fine-tuned on the downstream label.

    This is a plain PyTorch reimplementation. The authors' code
    (https://github.com/layer6ai-labs/DuETT) was used as the specification;
    where it differs from the paper, this module follows the code and says
    so on :class:`DuETT`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..datasets import SampleDataset
from .base_model import BaseModel

MAX_COUNT = 15
"""Observation counts above this share one count embedding (as in the code)."""


def bin_time_series(
    x: torch.Tensor,
    x_time: torch.Tensor,
    n_bins: int,
    window: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bins irregular observations into DuETT's time x event grid.

    Each cell keeps the last observed value in its bin (0 if none) and the
    number of observations. With ``window=None`` the bins split
    ``[0, last observation time]`` of each sample evenly, and an observation
    at the last time goes to the last bin, as in the authors' PhysioNet code.
    With a ``window``, the bins split ``[0, window)`` for every sample and
    later observations go to the last bin.

    Args:
        x: ``[B, L, V]`` values, NaN where a variable was not observed.
        x_time: ``[B, L]`` observation times in hours, ascending within a
            sample, NaN for padding.
        n_bins: Number of time bins.
        window: Length of the binned window in hours, or ``None`` for each
            sample's own last observation time.

    Returns:
        ``(values, counts, bin_ends)``: ``values`` and ``counts`` are
        ``[B, n_bins, V]``; ``bin_ends`` is ``[B, n_bins]``, the end time of
        each bin in days (the time unit of the authors' code).

    Examples:
        >>> import torch
        >>> x = torch.tensor([[[1.0], [2.0], [float("nan")], [4.0]]])
        >>> t = torch.tensor([[0.0, 1.0, 2.0, 4.0]])
        >>> values, counts, ends = bin_time_series(x, t, n_bins=2)
        >>> values[0, :, 0], counts[0, :, 0]
        (tensor([2., 4.]), tensor([2., 1.]))
    """
    batch, length, n_vars = x.shape
    valid_time = torch.isfinite(x_time)
    observed = torch.isfinite(x) & valid_time.unsqueeze(-1)
    times = torch.where(valid_time, x_time, torch.zeros_like(x_time))

    if window is None:
        span = times.amax(dim=1, keepdim=True)
        at_end = times >= span
        span = span.clamp(min=1e-6)
    else:
        span = times.new_full((batch, 1), float(window))
        at_end = times >= span
    bins = torch.floor(times / span * n_bins).long().clamp(0, n_bins - 1)
    bins = torch.where(at_end, torch.full_like(bins, n_bins - 1), bins)

    index = bins.unsqueeze(-1).expand(batch, length, n_vars)
    obs = observed.to(x.dtype)
    counts = x.new_zeros(batch, n_bins, n_vars).scatter_add_(1, index, obs)

    # The last observation of a cell is the one with the largest position.
    # Float positions: MPS has no int64 atomic max (exact below 2**24 steps).
    position = torch.arange(length, device=x.device, dtype=torch.float32)
    position = position.view(1, length, 1).expand(batch, length, n_vars)
    position = torch.where(observed, position, torch.full_like(position, -1))
    last = torch.full(
        (batch, n_bins, n_vars), -1.0, device=x.device
    ).scatter_reduce_(1, index, position, reduce="amax")
    gather_index = last.clamp(min=0).long()
    values = torch.gather(torch.nan_to_num(x, nan=0.0), 1, gather_index)
    values = torch.where(last >= 0, values, torch.zeros_like(values))

    steps = torch.arange(1, n_bins + 1, device=x.device, dtype=x.dtype)
    bin_ends = steps.view(1, n_bins) / n_bins * span.to(x.dtype) / 24.0
    return values, counts, bin_ends


class ScaleNorm(nn.Module):
    """Scale normalization (Nguyen & Salazar, 2019): ``g * sqrt(d) * x / |x|``.

    The authors' code uses the ``x-transformers`` ScaleNorm, which has one
    learned scalar ``g`` per norm.

    Args:
        dim: Size of the normalized (last) dimension.
        eps: Lower bound of the norm.

    Examples:
        >>> import torch
        >>> norm = ScaleNorm(4)
        >>> norm(torch.ones(2, 4)).norm(dim=-1).tolist()
        [2.0, 2.0]
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes ``x`` over its last dimension."""
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return x / norm * self.scale * self.g


class DuETTEncoderLayer(nn.Module):
    """One pre-norm transformer layer over wide tokens.

    DuETT's tokens are whole rows or columns of the grid, so they are wide
    (``d_embedding`` times the length of the other axis), while attention
    projects them down to ``n_heads * d_head`` dimensions: "skinny"
    attention on "fat" tokens, as in the authors' code. A final ScaleNorm
    follows the residual blocks, like a depth-1 ``x_transformers.Encoder``.

    Args:
        dim: Token width.
        n_heads: Number of attention heads.
        d_head: Width of each head.
        d_feedforward: Hidden width of the feed-forward block.

    Examples:
        >>> import torch
        >>> layer = DuETTEncoderLayer(dim=16, n_heads=2, d_head=4, d_feedforward=8)
        >>> layer(torch.randn(3, 5, 16)).shape
        torch.Size([3, 5, 16])
    """

    def __init__(self, dim: int, n_heads: int, d_head: int, d_feedforward: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.attn_norm = ScaleNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.to_out = nn.Linear(inner, dim)
        self.ff_norm = ScaleNorm(dim)
        self.ff_in = nn.Linear(dim, d_feedforward)
        self.ff_act = nn.GELU()
        self.ff_out = nn.Linear(d_feedforward, dim)
        self.final_norm = ScaleNorm(dim)

    def forward(self, x: torch.Tensor, dropout: float = 0.0) -> torch.Tensor:
        """Applies attention and feed-forward blocks.

        Args:
            x: ``[B, N, dim]`` tokens.
            dropout: Dropout on the attention weights and in the
                feed-forward block (only applied in training mode).

        Returns:
            ``[B, N, dim]`` transformed tokens.
        """
        batch, n_tokens, _ = x.shape
        p = dropout if self.training else 0.0
        q, k, v = self.to_qkv(self.attn_norm(x)).chunk(3, dim=-1)
        q, k, v = (
            t.view(batch, n_tokens, self.n_heads, self.d_head).transpose(1, 2)
            for t in (q, k, v)
        )
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        attended = attended.transpose(1, 2).reshape(batch, n_tokens, -1)
        x = x + self.to_out(attended)
        hidden = F.dropout(self.ff_act(self.ff_in(self.ff_norm(x))), p, self.training)
        x = x + self.ff_out(hidden)
        return self.final_norm(x)


class PerVariableEmbedding(nn.Module):
    """One small MLP per event type: ``[value, count] -> d_embedding``.

    Equivalent to the authors' list of ``n_vars`` separate
    ``Linear(2, h) -> ReLU -> BatchNorm -> Linear(h, d)`` MLPs, but batched
    over the variables with grouped weights.

    Args:
        n_vars: Number of event types.
        d_hidden: Hidden width of each MLP.
        d_embedding: Output width.

    Examples:
        >>> import torch
        >>> emb = PerVariableEmbedding(n_vars=3, d_hidden=8, d_embedding=4)
        >>> emb(torch.randn(2, 5, 3, 2)).shape
        torch.Size([2, 5, 3, 4])
    """

    def __init__(self, n_vars: int, d_hidden: int, d_embedding: int) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(n_vars, 2, d_hidden))
        self.b1 = nn.Parameter(torch.empty(n_vars, d_hidden))
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm1d(n_vars * d_hidden)
        self.w2 = nn.Parameter(torch.empty(n_vars, d_hidden, d_embedding))
        self.b2 = nn.Parameter(torch.empty(n_vars, d_embedding))
        for weight, bias, fan_in in (
            (self.w1, self.b1, 2),
            (self.w2, self.b2, d_hidden),
        ):
            bound = 1 / math.sqrt(fan_in)  # nn.Linear's default init
            nn.init.uniform_(weight, -bound, bound)
            nn.init.uniform_(bias, -bound, bound)

    def forward(self, cells: torch.Tensor) -> torch.Tensor:
        """Embeds ``[B, T, V, 2]`` cells into ``[B, T, V, d_embedding]``."""
        batch, n_time, n_vars, _ = cells.shape
        hidden = torch.einsum("btvi,vih->btvh", cells, self.w1) + self.b1
        hidden = self.act(hidden).reshape(batch * n_time, -1)
        hidden = self.norm(hidden).view(batch, n_time, n_vars, -1)
        return torch.einsum("btvh,vhd->btvd", hidden, self.w2) + self.b2


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    """``Linear -> ReLU -> BatchNorm -> Linear``, the code's ``simple_mlp``."""
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.BatchNorm1d(d_hidden),
        nn.Linear(d_hidden, d_out),
    )


class DuETT(BaseModel):
    """Dual Event Time Transformer (Labach et al., MLHC 2023).

    The input of a sample is a grid of ``n_time_bins`` time bins by
    ``n_vars`` event types. Each cell's last value and observation count
    (counts above 15 share an embedding) go through a per-event-type MLP to
    a ``d_embedding`` vector. The static features, encoded by an MLP, form
    one more event row, repeated in every bin (early fusion), and a learned
    [REP] token forms one more time column, so the tensor is
    ``[B, n_time_bins + 1, n_vars + 1, d_embedding]``. Each of the
    ``n_layers`` DuETT layers runs an event transformer, whose tokens are
    the event rows (all bins of one event type, flattened), then a time
    transformer, whose tokens are the time columns. A learned embedding per
    event type, and a learned embedding of each bin's end time (continuous
    value embedding, CVE), are added before every event and time
    transformer respectively. The classifier reads the flattened [REP]
    column.

    **Inputs.** By default the model reads the irregular ``x`` (``[B, L, V]``,
    NaN where unobserved, ideally z-scored) and ``x_time`` (``[B, L]`` hours,
    NaN padding) that :class:`~pyhealth.tasks.PhysioNet2012Mortality`
    produces, and bins them with :func:`bin_time_series`. With ``count_key``
    it instead reads a task's pre-binned grid: ``value_key`` (``[B, T, V]``
    last values, 0 where unobserved), ``count_key`` (``[B, T, V]``) and
    ``time_key`` (``[B, T]`` bin end times, in days in the authors' code).
    ``static_key`` optionally names a ``[B, S]`` static feature vector;
    without one, the static row is a learned constant.

    **Training.** :meth:`pretrain_forward` gives the self-supervised loss:
    per sample, one random time bin and one random event type are replaced
    by the [MASK] embedding; their values are predicted with MSE and their
    presence with BCE, ``loss = MSE + presence_loss_weight * BCE`` for each
    of the two. :meth:`forward` gives the supervised loss with the dataset's
    label processor, so :class:`~pyhealth.trainer.Trainer` can fine-tune.
    Transformer dropout is ``pretrain_dropout`` in :meth:`pretrain_forward`
    and ``dropout`` in :meth:`forward`, so one model can be pre-trained
    and then fine-tuned, as in the authors' recipe.

    **Paper vs. code.** Where they differ, this follows the authors' code:

    - The cell embedding is an MLP per event type (the paper says a linear
      layer) and the count is embedded as one learned scalar per count.
    - ``d_embedding=24``, 2 heads, feed-forward width 512 and ScaleNorm
      pre-norm are the code's defaults; the paper does not state them.
    - Pre-training drops each event type with probability
      ``pretrain_input_dropout`` (0.5 in the code), except the ones observed
      in the masked bin; fine-tuning replaces each time bin with [MASK] with
      probability ``aug_mask`` (0.5 in the code). Neither is in the paper.
    - The paper never gives the presence-loss weight; the code uses 0.2.
    - The paper's value loss averages over observed cells, the code's over
      all cells of the masked bin/event, with unobserved ones zeroed. The
      code's is the default; ``pretrain_loss_observed_only=True`` gives the
      paper's.
    - The code weights the classes by ``1/(2p)`` and ``1/(2(1-p))``. Here
      the loss is the label processor's; ``"binary"`` labels with
      ``pos_weight="balanced"`` give the same weighting up to a constant.

    Args:
        dataset: The samples; used for the field names, the number of event
            types and static features, and the label type.
        n_time_bins: Number of time bins ``T`` (32 in the code). For
            pre-binned input it must match the task's grid.
        d_embedding: Width of each cell embedding.
        n_layers: Number of DuETT layers (event + time transformer pairs).
        n_heads: Attention heads per transformer.
        d_feedforward: Hidden width of each transformer's feed-forward block.
        d_hidden_embedding: Hidden width of the per-event-type MLPs.
        d_hidden_static: Hidden width of the static-feature MLP.
        d_hidden_head: Hidden width of the classifier MLP.
        dropout: Transformer dropout in :meth:`forward` (0.5 when
            fine-tuning in the code).
        aug_mask: Probability of masking each time bin in :meth:`forward`
            during training (0.5 when fine-tuning in the code).
        pretrain_dropout: Transformer dropout in :meth:`pretrain_forward`.
        pretrain_input_dropout: Probability of dropping an event type in
            :meth:`pretrain_forward`.
        presence_loss_weight: Weight ``alpha`` of the presence BCE losses.
        pretrain_loss_observed_only: Average the value MSE over observed
            cells only (the paper) instead of over all cells (the code).
        bin_window: Window of :func:`bin_time_series` in hours, ``None`` for
            each sample's last observation (irregular input only).
        value_key: Input field with the values.
        time_key: Input field with the observation or bin end times.
        count_key: Input field with the counts of a pre-binned grid, or
            ``None`` to bin irregular input.
        static_key: Input field with the static features, or ``None``.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> rng = np.random.default_rng(0)
        >>> def sample(i):
        ...     x = rng.normal(size=(6, 3)).astype("float32")
        ...     x[rng.random((6, 3)) < 0.5] = np.nan
        ...     t = np.arange(6, dtype="float32") * 8
        ...     return {"patient_id": f"p{i}", "x": x, "x_time": t, "label": i % 2}
        >>> dataset = create_sample_dataset(
        ...     [sample(i) for i in range(4)],
        ...     {"x": "tensor", "x_time": "tensor"}, {"label": "binary"},
        ... )
        >>> model = DuETT(dataset, n_time_bins=4, d_embedding=8)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=4)))
        >>> model.pretrain_forward(**batch)["loss"].shape
        torch.Size([])
        >>> model(**batch)["y_prob"].shape
        torch.Size([4, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        n_time_bins: int = 32,
        d_embedding: int = 24,
        n_layers: int = 2,
        n_heads: int = 2,
        d_feedforward: int = 512,
        d_hidden_embedding: int = 64,
        d_hidden_static: int = 128,
        d_hidden_head: int = 64,
        dropout: float = 0.5,
        aug_mask: float = 0.5,
        pretrain_dropout: float = 0.0,
        pretrain_input_dropout: float = 0.5,
        presence_loss_weight: float = 0.2,
        pretrain_loss_observed_only: bool = False,
        bin_window: float | None = None,
        value_key: str = "x",
        time_key: str = "x_time",
        count_key: str | None = None,
        static_key: str | None = None,
    ) -> None:
        super().__init__(dataset)
        if len(self.label_keys) != 1:
            raise ValueError("DuETT expects exactly one label field.")
        if d_embedding % n_heads:
            raise ValueError("d_embedding must be divisible by n_heads.")
        for name, p in (
            ("dropout", dropout),
            ("aug_mask", aug_mask),
            ("pretrain_dropout", pretrain_dropout),
            ("pretrain_input_dropout", pretrain_input_dropout),
        ):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        self.label_key = self.label_keys[0]
        self.value_key, self.time_key = value_key, time_key
        self.count_key, self.static_key = count_key, static_key

        sample = dataset[0]
        values = torch.as_tensor(sample[value_key])
        self.n_vars = int(values.shape[-1])
        if count_key is not None and int(values.shape[0]) != n_time_bins:
            raise ValueError(
                f"n_time_bins={n_time_bins} but {value_key!r} has "
                f"{int(values.shape[0])} bins."
            )
        self.d_static = (
            0
            if static_key is None
            else int(torch.as_tensor(sample[static_key]).numel())
        )
        self.n_time_bins = n_time_bins
        self.d_embedding = d_embedding
        self.dropout = dropout
        self.aug_mask = aug_mask
        self.pretrain_dropout = pretrain_dropout
        self.pretrain_input_dropout = pretrain_input_dropout
        self.presence_loss_weight = presence_loss_weight
        self.pretrain_loss_observed_only = pretrain_loss_observed_only
        self.bin_window = bin_window

        event_dim = d_embedding * (n_time_bins + 1)  # one event row
        time_dim = d_embedding * (self.n_vars + 1)  # one time column
        self.count_embedding = nn.Embedding(MAX_COUNT + 1, 1)
        self.cell_embedding = PerVariableEmbedding(
            self.n_vars, d_hidden_embedding, d_embedding
        )
        if self.d_static:
            self.static_encoder = _mlp(self.d_static, d_hidden_static, d_embedding)
        else:
            self.static_row = nn.Parameter(torch.randn(d_embedding))
        self.mask_embedding = nn.Parameter(torch.randn(d_embedding))
        self.rep_embedding = nn.Parameter(torch.randn(d_embedding))
        self.event_embedding = nn.Parameter(torch.randn(self.n_vars + 1, event_dim))
        d_cve = int(math.sqrt(time_dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(1, d_cve),
            nn.Tanh(),
            nn.BatchNorm1d(d_cve),
            nn.Linear(d_cve, time_dim),
        )
        self.rep_time_embedding = nn.Parameter(torch.randn(time_dim))
        d_head = d_embedding // n_heads
        self.event_transformers = nn.ModuleList(
            DuETTEncoderLayer(event_dim, n_heads, d_head, d_feedforward)
            for _ in range(n_layers)
        )
        self.time_transformers = nn.ModuleList(
            DuETTEncoderLayer(time_dim, n_heads, d_head, d_feedforward)
            for _ in range(n_layers)
        )
        self.head = _mlp(time_dim, d_hidden_head, self.label_processor.output_size())
        self.value_proj = nn.Linear(time_dim, self.n_vars)
        self.presence_proj = nn.Linear(time_dim, self.n_vars)
        self.event_value_proj = nn.Linear(event_dim, n_time_bins)
        self.event_presence_proj = nn.Linear(event_dim, n_time_bins)

    def grid(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """The batch's binned grid.

        Args:
            **kwargs: A batch from :func:`~pyhealth.datasets.get_dataloader`.

        Returns:
            ``(values, counts, times, static)``: ``[B, T, V]``, ``[B, T, V]``,
            ``[B, T]`` and ``[B, S]`` (``None`` without ``static_key``).
        """
        values = kwargs[self.value_key].to(self.device).float()
        times = kwargs[self.time_key].to(self.device).float()
        if self.count_key is None:
            values, counts, times = bin_time_series(
                values, times, self.n_time_bins, self.bin_window
            )
        else:
            counts = kwargs[self.count_key].to(self.device).float()
        static = None
        if self.static_key is not None:
            static = kwargs[self.static_key].to(self.device).float()
            static = static.reshape(static.shape[0], -1)
        return values, counts, times, static

    def encode(
        self,
        values: torch.Tensor,
        counts: torch.Tensor,
        times: torch.Tensor,
        static: torch.Tensor | None = None,
        time_mask: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
        dropout: float = 0.0,
    ) -> torch.Tensor:
        """Runs the DuETT layers on a binned grid.

        Args:
            values: ``[B, T, V]`` last values, 0 where unobserved.
            counts: ``[B, T, V]`` observation counts.
            times: ``[B, T]`` bin end times.
            static: ``[B, S]`` static features, or ``None``.
            time_mask: ``[B, T]``, true for bins to replace with [MASK].
            event_mask: ``[B, V]``, true for event types to replace with
                [MASK] (in every bin, the [REP] column included).
            dropout: Transformer dropout.

        Returns:
            ``[B, T + 1, V + 1, d_embedding]``: the last time index is the
            [REP] column and the last event index the static row.
        """
        batch, n_time, n_vars = values.shape
        if n_time != self.n_time_bins:
            raise ValueError(f"Expected {self.n_time_bins} time bins, got {n_time}.")
        count_index = counts.long().clamp(0, MAX_COUNT)
        count_scalar = self.count_embedding(count_index).squeeze(-1)
        cells = self.cell_embedding(torch.stack([values, count_scalar], dim=-1))
        if static is not None:
            static_row = self.static_encoder(static)
        else:
            static_row = self.static_row.expand(batch, -1)
        static_row = static_row.unsqueeze(1).expand(batch, n_time, -1).unsqueeze(2)
        psi = torch.cat([cells, static_row], dim=2)  # [B, T, V + 1, d]
        rep = self.rep_embedding.expand(batch, 1, n_vars + 1, -1)
        psi = torch.cat([psi, rep], dim=1)  # [B, T + 1, V + 1, d]

        mask = self.mask_embedding.view(1, 1, 1, -1)
        if time_mask is not None:
            time_mask = F.pad(time_mask.bool(), (0, 1), value=False)
            psi = torch.where(time_mask.view(batch, n_time + 1, 1, 1), mask, psi)
        if event_mask is not None:
            event_mask = F.pad(event_mask.bool(), (0, 1), value=False)
            psi = torch.where(event_mask.view(batch, 1, n_vars + 1, 1), mask, psi)

        time_emb = self.time_embedding(times.reshape(-1, 1)).view(batch, n_time, -1)
        rep_time = self.rep_time_embedding.expand(batch, 1, -1)
        time_emb = torch.cat([time_emb, rep_time], dim=1)  # [B, T + 1, time_dim]
        shape = psi.shape
        for event_layer, time_layer in zip(
            self.event_transformers, self.time_transformers
        ):
            # Event tokens: one per event row, [B, V + 1, (T + 1) * d].
            tokens = psi.transpose(1, 2).flatten(2) + self.event_embedding
            out = event_layer(tokens, dropout)
            psi = out.view(batch, shape[2], shape[1], shape[3]).transpose(1, 2)
            # Time tokens: one per time column, [B, T + 1, (V + 1) * d].
            tokens = psi.flatten(2) + time_emb
            psi = time_layer(tokens, dropout).view(shape)
        return psi

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Classifies the batch from the [REP] column.

        In training mode, each time bin is masked with probability
        ``aug_mask`` first.

        Args:
            **kwargs: A batch from :func:`~pyhealth.datasets.get_dataloader`;
                with the label field, the loss is computed too. ``embed=True``
                also returns the [REP] representation.

        Returns:
            A dict with ``logit`` and ``y_prob``; with labels also ``loss`` and
            ``y_true``; with ``embed=True`` also ``embed``
            (``[B, (V + 1) * d_embedding]``).
        """
        values, counts, times, static = self.grid(**kwargs)
        time_mask = None
        if self.training and self.aug_mask > 0:
            time_mask = torch.rand(values.shape[:2], device=self.device) < self.aug_mask
        psi = self.encode(
            values, counts, times, static, time_mask=time_mask, dropout=self.dropout
        )
        rep = psi[:, -1].flatten(1)
        logit = self.head(rep)
        out = {"logit": logit, "y_prob": self.label_processor.activation(logit)}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            out["loss"] = self.label_processor.loss(logit, y_true)
            out["y_true"] = y_true
        if kwargs.get("embed", False):
            out["embed"] = rep
        return out

    def pretrain_forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Self-supervised loss: predict one masked time bin and event type.

        For each sample, one random time bin and one random event type are
        replaced by [MASK] and every event type is dropped (values and counts
        zeroed in all bins) with probability ``pretrain_input_dropout``,
        except the ones observed in the masked bin. The masked bin's values
        and presence are predicted from its output column, and the masked
        event type's from its output row. Labels are not used. The masking
        and input dropout are random in evaluation mode too, as in the
        authors' validation loss.

        Args:
            **kwargs: A batch from :func:`~pyhealth.datasets.get_dataloader`.

        Returns:
            A dict with the total ``loss`` and its parts ``time_value_loss``,
            ``time_presence_loss``, ``event_value_loss`` and
            ``event_presence_loss`` (unweighted).
        """
        values, counts, times, static = self.grid(**kwargs)
        batch, n_time, n_vars = values.shape
        rows = torch.arange(batch, device=self.device)
        time_index = torch.randint(n_time, (batch,), device=self.device)
        event_index = torch.randint(n_vars, (batch,), device=self.device)
        observed = (counts > 0).float()
        time_target = values[rows, time_index]  # [B, V]
        time_presence = observed[rows, time_index]
        event_target = values[rows, :, event_index]  # [B, T]
        event_presence = observed[rows, :, event_index]

        if self.pretrain_input_dropout > 0:
            keep = torch.rand(batch, n_vars, device=self.device)
            keep = (keep >= self.pretrain_input_dropout) | time_presence.bool()
            keep = keep.unsqueeze(1).to(values.dtype)
            values, counts = values * keep, counts * keep

        time_mask = F.one_hot(time_index, n_time).bool()
        event_mask = F.one_hot(event_index, n_vars).bool()
        psi = self.encode(
            values,
            counts,
            times,
            static,
            time_mask=time_mask,
            event_mask=event_mask,
            dropout=self.pretrain_dropout,
        )
        time_out = psi[rows, time_index].flatten(1)  # [B, (V + 1) * d]
        event_out = psi[rows, :, event_index].flatten(1)  # [B, (T + 1) * d]

        losses = {
            "time_value_loss": self._value_loss(
                self.value_proj(time_out), time_target, time_presence
            ),
            "time_presence_loss": F.binary_cross_entropy_with_logits(
                self.presence_proj(time_out), time_presence
            ),
            "event_value_loss": self._value_loss(
                self.event_value_proj(event_out), event_target, event_presence
            ),
            "event_presence_loss": F.binary_cross_entropy_with_logits(
                self.event_presence_proj(event_out), event_presence
            ),
        }
        alpha = self.presence_loss_weight
        losses["loss"] = (
            losses["time_value_loss"]
            + losses["event_value_loss"]
            + alpha * (losses["time_presence_loss"] + losses["event_presence_loss"])
        )
        return losses

    def _value_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, presence: torch.Tensor
    ) -> torch.Tensor:
        """MSE on observed cells: averaged over all cells, or observed ones."""
        squared = (prediction * presence - target * presence) ** 2
        if self.pretrain_loss_observed_only:
            return squared.sum() / presence.sum().clamp(min=1.0)
        return squared.mean()
