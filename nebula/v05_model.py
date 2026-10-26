"""v0.5-alpha 分层音画时序模型（冻结编码器特征上的轻量 Transformer）。

结构（见 docs/plans/v0.5-alpha-temporal-model.md 第 3 节）：

1. 局部段（每个 8s 段内）：
   - PE-AV joint/audio/video 3 个 1024 维段级向量 -> 3 个 context token；
   - 稀疏 8fps CLIP/DINOv2 帧向量按 bin 聚合成 16 个 visual token；
   - 100Hz 音频事件 mean/std（24 维）-> 16 个 audio token；
   - 30fps 视频事件 mean/std（12 维）-> 16 个 visual-event token；
   - [CLS] + 以上 token，加模态嵌入与段内位置编码，过 local_layers 层
     TransformerEncoder；CLS + masked mean -> segment token。
2. 段间：segment token + 段位置编码 + 有效 mask，过 temporal_layers 层
   TransformerEncoder；有效段 masked mean -> work vector。
3. 预测头：masked segment 的 PE-AV joint 重建（256 -> 1024）、AV 配对二分类、
   3 段排列恢复（6 类）。

mask 语义约定（与 tools/build_v05_tokens 产物一致）：

- segment_mask (B,W) bool：True 表示该段有效（非 padding）；
- mask_visual / mask_audio / mask_visual_event (B,W,16) bool：True 表示该 bin 有内容；
- masked_segments (B,W) bool：True 表示该段被选为 mask 预测目标，段 token 在进入
  段间编码器前替换为 learned mask token（目标段的 PE-AV 内容不会泄漏给模型）；
- padding 段的所有 token 均被 key padding mask 完全屏蔽，且其内容被替换为常量，
  因此修改 padding 内容不会改变任何输出（测试断言）。

模型只做 CPU 可跑的小 config 前向/损失验证；不加载真实数据、不训练。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_CONTEXT_TOKENS = 3
"""PE-AV 段级 context token 数（joint/audio/video）。"""

_BIN_MODALITIES = 3
"""局部 bin token 模态数（visual/audio/visual-event）。"""

_MASK_SOURCE_DEFAULT = "paav"


@dataclass(frozen=True)
class V05AlphaConfig:
    """v0.5-alpha 模型超参（默认 small 配置，参数量 < 5M）。"""

    hidden: int = 256
    heads: int = 4
    local_layers: int = 2
    temporal_layers: int = 2
    dropout: float = 0.1
    max_segments: int = 128
    bins: int = 16
    paav_dim: int = 1024
    clip_dim: int = 512
    dino_dim: int = 384
    audio_dim: int = 24
    visual_event_dim: int = 12
    order_k: int = 3
    order_classes: int = 6

    def __post_init__(self) -> None:
        if int(self.hidden) <= 0:
            raise ValueError(f"hidden 必须 > 0，收到 {self.hidden!r}")
        if int(self.heads) <= 0:
            raise ValueError(f"heads 必须 > 0，收到 {self.heads!r}")
        if int(self.hidden) % int(self.heads) != 0:
            raise ValueError(
                f"hidden({self.hidden}) 必须能被 heads({self.heads}) 整除"
            )
        if int(self.local_layers) < 1:
            raise ValueError("local_layers 必须 >= 1")
        if int(self.temporal_layers) < 1:
            raise ValueError("temporal_layers 必须 >= 1")
        dropout = float(self.dropout)
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout 必须在 [0,1)，收到 {self.dropout!r}")
        if int(self.max_segments) < 1:
            raise ValueError("max_segments 必须 >= 1")
        if int(self.bins) < 1:
            raise ValueError("bins 必须 >= 1")
        for name in (
            "paav_dim",
            "clip_dim",
            "dino_dim",
            "audio_dim",
            "visual_event_dim",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} 必须 >= 1，收到 {getattr(self, name)!r}")
        order_k = int(self.order_k)
        if order_k < 2:
            raise ValueError("order_k 必须 >= 2")
        if int(self.order_classes) != math.factorial(order_k):
            raise ValueError(
                f"order_classes 必须等于 order_k! = {math.factorial(order_k)}，"
                f"收到 {self.order_classes!r}"
            )


def _require_tensor(batch: Mapping[str, Tensor], key: str) -> Tensor:
    value = batch.get(key)
    if not isinstance(value, Tensor):
        raise ValueError(f"batch[{key!r}] 必须是 torch.Tensor")
    return value


def _expect_shape(tensor: Tensor, shape: tuple[int, ...], name: str) -> None:
    actual = tuple(int(v) for v in tensor.shape)
    if actual != shape:
        raise ValueError(f"{name} 形状应为 {shape}，收到 {actual}")


def _first_label(
    batch: Mapping[str, Tensor], names: tuple[str, ...]
) -> Tensor | None:
    for name in names:
        value = batch.get(name)
        if isinstance(value, Tensor):
            return value
    return None


class V05AlphaModel(nn.Module):
    """v0.5-alpha：局部跨模态编码 + 全曲时序聚合的轻量 Transformer。

    forward(batch) 返回 dict：
    - work_vector (B, hidden)
    - segment_tokens (B, W, hidden)（进入段间编码器前的 token；masked/padding 段为
      learned mask token）
    - mask_pred (B, W, paav_dim)（对被 mask 段的 PE-AV joint 重建）
    - av_logits (B,)（视觉/音频池化后的 AV 配对 logits，未过 sigmoid）
    - order_logits (B, order_classes)（3 个段 token 的排列分类）

    batch 必需键：paav (B,W,3,paav_dim)、visual_clip (B,W,bins,clip_dim)、
    visual_dino (B,W,bins,dino_dim)、audio (B,W,bins,audio_dim)、
    visual_event (B,W,bins,visual_event_dim)、mask_visual/mask_audio/
    mask_visual_event (B,W,bins bool)、segment_mask (B,W bool)、
    masked_segments (B,W bool)。可选 order_indices (B,order_k long) 指定 order
    头使用的段下标；缺省按有效且非 masked 的段优先选择。
    """

    def __init__(self, config: V05AlphaConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else V05AlphaConfig()
        if not isinstance(self.config, V05AlphaConfig):
            raise ValueError(
                f"config 必须是 V05AlphaConfig，收到 {type(self.config).__name__}"
            )
        cfg = self.config
        hidden = int(cfg.hidden)
        self.hidden = hidden
        self.bins = int(cfg.bins)
        self.max_segments = int(cfg.max_segments)
        self.order_k = int(cfg.order_k)

        # ---------------------------------------------------------- 模态投影
        self.paav_proj = nn.Linear(int(cfg.paav_dim), hidden)
        self.clip_proj = nn.Linear(int(cfg.clip_dim), hidden)
        self.dino_proj = nn.Linear(int(cfg.dino_dim), hidden)
        self.audio_proj = nn.Linear(int(cfg.audio_dim), hidden)
        self.visual_event_proj = nn.Linear(int(cfg.visual_event_dim), hidden)
        # 视觉 token = CLIP + DINOv2 投影后相加，再乘/加模态嵌入
        self.context_modality = nn.Parameter(torch.zeros(_CONTEXT_TOKENS, hidden))
        self.visual_modality = nn.Parameter(torch.zeros(hidden))
        self.audio_modality = nn.Parameter(torch.zeros(hidden))
        self.event_modality = nn.Parameter(torch.zeros(hidden))
        # 缺失 bin 的 learned missing token（visual/audio/event 各一）
        self.missing_tokens = nn.Parameter(torch.zeros(_BIN_MODALITIES, hidden))

        # ---------------------------------------------------------- 位置/特殊 token
        local_length = 1 + _CONTEXT_TOKENS + _BIN_MODALITIES * self.bins
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.local_position = nn.Parameter(torch.zeros(local_length, hidden))
        self.segment_position = nn.Parameter(torch.zeros(self.max_segments, hidden))
        self.mask_token = nn.Parameter(torch.zeros(hidden))
        # 数据集 PE-AV joint 均值（由 trainer 写入）：mask head 只学习“相对全局均值的残差”，
        # 使 mask 重建至少不差于常量均值基线；零初始化时等价于普通 head。
        self.register_buffer("mask_mean", torch.zeros(int(cfg.paav_dim)))

        # ---------------------------------------------------------- 编码器
        local_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(cfg.heads),
            dim_feedforward=4 * hidden,
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.local_encoder = nn.TransformerEncoder(
            local_layer, num_layers=int(cfg.local_layers), enable_nested_tensor=False
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(cfg.heads),
            dim_feedforward=4 * hidden,
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=int(cfg.temporal_layers),
            enable_nested_tensor=False,
        )
        self.local_norm = nn.LayerNorm(hidden)
        self.segment_norm = nn.LayerNorm(hidden)
        self.temporal_norm = nn.LayerNorm(hidden)
        self.work_norm = nn.LayerNorm(hidden)

        # ---------------------------------------------------------- 预测头
        self.mask_head = nn.Linear(hidden, int(cfg.paav_dim))
        # 小初始化：初始 mask_pred 接近 mask_mean（常量均值基线），同时保留对 mask token/
        # 段内容的非零敏感性，训练主要学习相对均值的残差。
        nn.init.normal_(self.mask_head.weight, std=0.01)
        nn.init.zeros_(self.mask_head.bias)
        self.av_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(hidden, 1),
        )
        self.order_head = nn.Sequential(
            nn.Linear(self.order_k * hidden, hidden),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(hidden, int(cfg.order_classes)),
        )

    # ---------------------------------------------------------------- 局部 token
    def _local_token_layout(self) -> tuple[int, int, int, int]:
        """返回 context/visual/audio/event 在局部序列中的切片边界。"""

        ctx_start = 1
        ctx_end = ctx_start + _CONTEXT_TOKENS
        vis_start = ctx_end
        vis_end = vis_start + self.bins
        aud_end = vis_end + self.bins
        ev_end = aud_end + self.bins
        return vis_start, vis_end, aud_end, ev_end

    def _build_local_tokens(
        self,
        *,
        paav: Tensor,
        visual_clip: Tensor,
        visual_dino: Tensor,
        audio: Tensor,
        visual_event: Tensor,
        mask_visual: Tensor,
        mask_audio: Tensor,
        mask_visual_event: Tensor,
        segment_mask: Tensor,
        visible_segment: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """构造局部 token 与 key padding mask；返回 (tokens, pad, 3 个 bin valid)。"""

        batch_size, segments = segment_mask.shape
        visual_valid = mask_visual & segment_mask.unsqueeze(-1)
        audio_valid = mask_audio & segment_mask.unsqueeze(-1)
        event_valid = mask_visual_event & segment_mask.unsqueeze(-1)

        context = self.paav_proj(paav)
        context = torch.where(
            visible_segment.unsqueeze(-1).unsqueeze(-1),
            context,
            torch.zeros_like(context),
        )
        context = context + self.context_modality.view(1, 1, _CONTEXT_TOKENS, -1)

        visual = self.clip_proj(visual_clip) + self.dino_proj(visual_dino)
        visual = torch.where(
            visual_valid.unsqueeze(-1),
            visual,
            self.missing_tokens[0].view(1, 1, 1, -1).expand_as(visual),
        )
        visual = visual + self.visual_modality

        audio_tokens = self.audio_proj(audio)
        audio_tokens = torch.where(
            audio_valid.unsqueeze(-1),
            audio_tokens,
            self.missing_tokens[1].view(1, 1, 1, -1).expand_as(audio_tokens),
        )
        audio_tokens = audio_tokens + self.audio_modality

        event_tokens = self.visual_event_proj(visual_event)
        event_tokens = torch.where(
            event_valid.unsqueeze(-1),
            event_tokens,
            self.missing_tokens[2].view(1, 1, 1, -1).expand_as(event_tokens),
        )
        event_tokens = event_tokens + self.event_modality

        cls = self.cls_token.view(1, 1, 1, -1).expand(batch_size, segments, 1, -1)
        tokens = torch.cat([cls, context, visual, audio_tokens, event_tokens], dim=2)
        tokens = tokens + self.local_position.view(
            1, 1, tokens.shape[2], self.hidden
        )

        cls_valid = torch.ones(
            batch_size, segments, 1, dtype=torch.bool, device=segment_mask.device
        )
        context_valid = visible_segment.unsqueeze(-1).expand(-1, -1, _CONTEXT_TOKENS)
        valid = torch.cat(
            [cls_valid, context_valid, visual_valid, audio_valid, event_valid], dim=2
        )
        pad = ~valid
        return tokens, pad, visual_valid, audio_valid, event_valid

    # ---------------------------------------------------------------- 池化工具
    @staticmethod
    def _masked_pool_tokens(
        values: Tensor, valid: Tensor, segment_weights: Tensor
    ) -> Tensor:
        """values (B,W,T,H), valid (B,W,T) -> 按 token/段双层 masked mean (B,H)。"""

        weights = valid.to(values.dtype)
        counts = weights.sum(dim=2, keepdim=True)
        token_pool = (values * weights.unsqueeze(-1)).sum(dim=2) / counts.clamp_min(1.0)
        has_token = (counts.squeeze(-1) > 0).to(values.dtype)
        segment_weights = segment_weights.to(values.dtype) * has_token
        denominator = segment_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (token_pool * segment_weights.unsqueeze(-1)).sum(dim=1) / denominator

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor, *, dim: int) -> Tensor:
        weights = mask.to(values.dtype)
        denominator = weights.sum(dim=dim, keepdim=True).clamp_min(1.0)
        return (values * weights.unsqueeze(-1)).sum(dim=dim) / denominator

    def _default_order_indices(
        self,
        preferred: Tensor,
        segment_mask: Tensor,
    ) -> Tensor:
        """按行选择 order 头使用的段下标（优先非 masked 有效段）。"""

        batch_size, segments = segment_mask.shape
        picks: list[list[int]] = []
        for row in range(batch_size):
            chosen = [index for index in range(segments) if bool(preferred[row, index])]
            if len(chosen) < self.order_k:
                extra = [
                    index
                    for index in range(segments)
                    if bool(segment_mask[row, index]) and index not in chosen
                ]
                chosen.extend(extra)
            while len(chosen) < self.order_k:
                chosen.append(chosen[-1] if chosen else 0)
            picks.append(chosen[: self.order_k])
        return torch.tensor(picks, dtype=torch.long, device=segment_mask.device)

    # ---------------------------------------------------------------- forward
    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        cfg = self.config
        paav = _require_tensor(batch, "paav")
        visual_clip = _require_tensor(batch, "visual_clip")
        visual_dino = _require_tensor(batch, "visual_dino")
        audio = _require_tensor(batch, "audio")
        visual_event = _require_tensor(batch, "visual_event")
        mask_visual = _require_tensor(batch, "mask_visual").bool()
        mask_audio = _require_tensor(batch, "mask_audio").bool()
        mask_visual_event = _require_tensor(batch, "mask_visual_event").bool()
        segment_mask = _require_tensor(batch, "segment_mask").bool()
        masked_segments_raw = batch.get("masked_segments")
        if isinstance(masked_segments_raw, Tensor):
            masked_segments = masked_segments_raw.bool()
        else:
            masked_segments = torch.zeros_like(segment_mask)

        if paav.ndim != 4 or paav.shape[2] != _CONTEXT_TOKENS:
            raise ValueError(
                f"paav 形状应为 (B,W,{_CONTEXT_TOKENS},paav_dim)，收到 {tuple(paav.shape)}"
            )
        batch_size, segments = int(paav.shape[0]), int(paav.shape[1])
        if segments < 1:
            raise ValueError("W 必须 >= 1")
        if segments > self.max_segments:
            raise ValueError(
                f"段数 W={segments} 超过 max_segments={self.max_segments}"
            )
        bins = self.bins
        _expect_shape(paav, (batch_size, segments, _CONTEXT_TOKENS, int(cfg.paav_dim)), "paav")
        _expect_shape(visual_clip, (batch_size, segments, bins, int(cfg.clip_dim)), "visual_clip")
        _expect_shape(visual_dino, (batch_size, segments, bins, int(cfg.dino_dim)), "visual_dino")
        _expect_shape(audio, (batch_size, segments, bins, int(cfg.audio_dim)), "audio")
        _expect_shape(
            visual_event,
            (batch_size, segments, bins, int(cfg.visual_event_dim)),
            "visual_event",
        )
        _expect_shape(segment_mask, (batch_size, segments), "segment_mask")
        for name, value in (
            ("mask_visual", mask_visual),
            ("mask_audio", mask_audio),
            ("mask_visual_event", mask_visual_event),
        ):
            _expect_shape(value, (batch_size, segments, bins), name)
        _expect_shape(
            masked_segments, (batch_size, segments), "masked_segments"
        )

        masked_segments = masked_segments & segment_mask
        visible_segment = segment_mask & ~masked_segments
        tokens, local_pad, visual_valid, audio_valid, event_valid = (
            self._build_local_tokens(
                paav=paav,
                visual_clip=visual_clip,
                visual_dino=visual_dino,
                audio=audio,
                visual_event=visual_event,
                mask_visual=mask_visual,
                mask_audio=mask_audio,
                mask_visual_event=mask_visual_event,
                segment_mask=segment_mask,
                visible_segment=visible_segment,
            )
        )

        local_length = tokens.shape[2]
        local_out = self.local_encoder(
            tokens.reshape(batch_size * segments, local_length, self.hidden),
            src_key_padding_mask=local_pad.reshape(batch_size * segments, local_length),
        )
        local_out = self.local_norm(local_out).view(
            batch_size, segments, local_length, self.hidden
        )

        # CLS + 非 CLS 有效 token 的 masked mean -> segment token
        token_valid = ~local_pad  # (B,W,L)
        mean_valid = token_valid[:, :, 1:].to(local_out.dtype)
        counts = mean_valid.sum(dim=2, keepdim=True).clamp_min(1.0)
        segment_mean = (
            local_out[:, :, 1:] * mean_valid.unsqueeze(-1)
        ).sum(dim=2) / counts
        segment_tokens = self.segment_norm(local_out[:, :, 0] + segment_mean)

        mask_vector = self.mask_token.view(1, 1, self.hidden)
        segment_tokens = torch.where(
            segment_mask.unsqueeze(-1),
            segment_tokens,
            mask_vector.expand_as(segment_tokens),
        )
        segment_tokens = torch.where(
            masked_segments.unsqueeze(-1),
            mask_vector.expand_as(segment_tokens),
            segment_tokens,
        )

        # 段间时序编码
        temporal_input = segment_tokens + self.segment_position[:segments].view(
            1, segments, self.hidden
        )
        temporal_out = self.temporal_encoder(
            temporal_input, src_key_padding_mask=~segment_mask
        )
        temporal_out = self.temporal_norm(temporal_out)
        work_vector = self.work_norm(self._masked_mean(temporal_out, segment_mask, dim=1))

        # ------------------------------------------------------------ 预测头
        mask_pred = self.mask_head(temporal_out) + self.mask_mean.view(1, 1, -1)

        vis_start, vis_end, aud_end, ev_end = self._local_token_layout()
        visual_pool = self._masked_pool_tokens(
            local_out[:, :, vis_start:vis_end], visual_valid, visible_segment
        )
        audio_pool = self._masked_pool_tokens(
            local_out[:, :, vis_end:aud_end], audio_valid, visible_segment
        )
        av_feature = torch.cat([visual_pool, audio_pool], dim=-1)
        av_logits = self.av_head(av_feature).squeeze(-1)

        order_indices = batch.get("order_indices")
        if isinstance(order_indices, Tensor):
            indices = order_indices.long()
            if tuple(indices.shape) != (batch_size, self.order_k):
                raise ValueError(
                    f"order_indices 形状应为 {(batch_size, self.order_k)}，"
                    f"收到 {tuple(indices.shape)}"
                )
            indices = indices.clamp(0, segments - 1)
        else:
            indices = self._default_order_indices(visible_segment, segment_mask)
        gather_index = indices.unsqueeze(-1).expand(
            batch_size, self.order_k, self.hidden
        )
        triplet = torch.gather(temporal_out, 1, gather_index)
        order_logits = self.order_head(triplet.flatten(1))

        return {
            "work_vector": work_vector,
            "segment_tokens": segment_tokens,
            "mask_pred": mask_pred,
            "av_logits": av_logits,
            "order_logits": order_logits,
        }


def v05_losses(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    *,
    w_mask: float = 1.0,
    w_av: float = 0.5,
    w_order: float = 0.5,
) -> dict[str, Tensor]:
    """v0.5-alpha 自监督损失（只做数值/接口验证，不训练真实模型）。

    - mask_loss：被 mask 且有效的段上，预测 PE-AV joint（batch.paav[...,0,:]，
      可用 batch.mask_target 覆盖）的 L2 归一化 MSE + 1-cosine 平均；
    - av_loss：batch.av_label（或 av_labels/av_target）(B,) 的 BCEWithLogits；
    - order_loss：batch.order_label（或 order_labels/order_target）(B,) 的交叉熵；
    - total = w_mask*mask_loss + w_av*av_loss + w_order*order_loss。

    缺失标签时对应损失返回 0（仍保留计算图），便于只验证 mask 分支。
    """

    mask_pred = outputs.get("mask_pred")
    if not isinstance(mask_pred, Tensor):
        raise ValueError("outputs['mask_pred'] 必须是 torch.Tensor")
    zero = mask_pred.sum() * 0.0
    mask_loss = zero
    av_loss = zero
    order_loss = zero

    segment_mask = batch.get("segment_mask")
    masked_segments = batch.get("masked_segments")
    target = batch.get("mask_target")
    if target is None:
        paav = batch.get("paav")
        if isinstance(paav, Tensor) and paav.ndim == 4:
            target = paav[..., 0, :]
    if (
        isinstance(segment_mask, Tensor)
        and isinstance(masked_segments, Tensor)
        and isinstance(target, Tensor)
    ):
        valid = segment_mask.bool() & masked_segments.bool()
        if bool(valid.any()):
            prediction = mask_pred[valid].float()
            expected = target.to(device=prediction.device, dtype=torch.float32)[valid]
            prediction = F.normalize(prediction, dim=-1, eps=1e-8)
            expected = F.normalize(expected, dim=-1, eps=1e-8)
            mse = F.mse_loss(prediction, expected)
            cosine = (prediction * expected).sum(dim=-1)
            mask_loss = mse + (1.0 - cosine).mean()

    av_label = _first_label(batch, ("av_label", "av_labels", "av_target"))
    if av_label is not None:
        logits = outputs["av_logits"].reshape(-1).float()
        labels = av_label.to(device=logits.device, dtype=torch.float32).reshape(-1)
        if labels.numel() != logits.numel():
            raise ValueError(
                f"AV 标签数量 {labels.numel()} 与 av_logits {logits.numel()} 不一致"
            )
        av_loss = F.binary_cross_entropy_with_logits(logits, labels)

    order_label = _first_label(batch, ("order_label", "order_labels", "order_target"))
    if order_label is not None:
        logits = outputs["order_logits"].float()
        labels = order_label.to(device=logits.device, dtype=torch.long).reshape(-1)
        if labels.numel() != logits.shape[0]:
            raise ValueError(
                f"order 标签数量 {labels.numel()} 与 batch 数 {logits.shape[0]} 不一致"
            )
        order_loss = F.cross_entropy(logits, labels)

    total = float(w_mask) * mask_loss + float(w_av) * av_loss + float(w_order) * order_loss
    return {
        "mask_loss": mask_loss,
        "av_loss": av_loss,
        "order_loss": order_loss,
        "total": total,
        "loss": total,
    }


__all__ = ["V05AlphaConfig", "V05AlphaModel", "v05_losses"]
