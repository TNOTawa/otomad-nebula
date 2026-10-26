"""v0.5-alpha 分层时序模型（nebula.v05_model）的 CPU 单元测试。

全部使用 tiny config 与随机张量，不加载真实数据、不训练真实模型：
- 输出 shape / 有限性 / 默认参数量 < 5M；
- padding 完全屏蔽：修改 padded 段/无效 bin 内容不影响任何输出；
- masked 段内容不泄漏、masked/padded 段 token 为 learned mask token；
- eval 确定性、梯度非零、v05_losses 各项与权重、mask 损失只算有效 masked 段；
- 配置/输入形状/order_indices 校验。
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from nebula.v05_model import (  # noqa: E402
    V05AlphaConfig,
    V05AlphaModel,
    v05_losses,
)


def _tiny_config(**overrides) -> V05AlphaConfig:
    base: dict[str, object] = {
        "hidden": 32,
        "heads": 4,
        "local_layers": 2,
        "temporal_layers": 2,
        "dropout": 0.0,
        "max_segments": 8,
        "bins": 4,
        "paav_dim": 16,
        "clip_dim": 8,
        "dino_dim": 6,
        "audio_dim": 8,
        "visual_event_dim": 4,
    }
    base.update(overrides)
    return V05AlphaConfig(**base)  # type: ignore[arg-type]


def _rand(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator)


def _make_batch(
    config: V05AlphaConfig,
    *,
    batch_size: int = 2,
    segments: int = 5,
    valid_lengths: list[int] | None = None,
    seed: int = 7,
    masked: bool = True,
) -> dict[str, torch.Tensor]:
    """构造确定性随机 batch（默认第 2 行有 padding）。"""

    generator = torch.Generator().manual_seed(seed)
    if valid_lengths is None:
        valid_lengths = [segments] + [max(1, segments - 2)] * (batch_size - 1)
    if len(valid_lengths) != batch_size:
        raise AssertionError("valid_lengths 长度必须等于 batch_size")
    if any(not 1 <= length <= segments for length in valid_lengths):
        raise AssertionError(f"valid_lengths 非法: {valid_lengths}")

    bins = config.bins
    paav = _rand((batch_size, segments, 3, config.paav_dim), generator)
    visual_clip = _rand((batch_size, segments, bins, config.clip_dim), generator)
    visual_dino = _rand((batch_size, segments, bins, config.dino_dim), generator)
    audio = _rand((batch_size, segments, bins, config.audio_dim), generator)
    visual_event = _rand(
        (batch_size, segments, bins, config.visual_event_dim), generator
    )

    segment_mask = torch.zeros(batch_size, segments, dtype=torch.bool)
    for row, length in enumerate(valid_lengths):
        segment_mask[row, :length] = True
    mask_visual = torch.ones(batch_size, segments, bins, dtype=torch.bool)
    mask_audio = torch.ones(batch_size, segments, bins, dtype=torch.bool)
    mask_visual_event = torch.ones(batch_size, segments, bins, dtype=torch.bool)
    # 在有效段里制造一些缺失 bin（mask=False）
    mask_visual[0, 0, bins - 1] = False
    mask_audio[0, 0, 0] = False
    if batch_size > 1:
        mask_visual_event[1, 0, bins - 1] = False
    mask_visual = mask_visual & segment_mask.unsqueeze(-1)
    mask_audio = mask_audio & segment_mask.unsqueeze(-1)
    mask_visual_event = mask_visual_event & segment_mask.unsqueeze(-1)

    masked_segments = torch.zeros(batch_size, segments, dtype=torch.bool)
    if masked:
        for row, length in enumerate(valid_lengths):
            candidates = list(range(1, length)) or [0]
            masked_segments[row, candidates[0]] = True

    return {
        "paav": paav,
        "visual_clip": visual_clip,
        "visual_dino": visual_dino,
        "audio": audio,
        "visual_event": visual_event,
        "mask_visual": mask_visual,
        "mask_audio": mask_audio,
        "mask_visual_event": mask_visual_event,
        "segment_mask": segment_mask,
        "masked_segments": masked_segments,
    }


def _add_labels(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = dict(batch)
    batch_size = int(batch["segment_mask"].shape[0])
    result["av_label"] = torch.tensor(
        [1.0 if index % 2 == 0 else 0.0 for index in range(batch_size)]
    )
    result["order_label"] = torch.tensor(
        [index % 6 for index in range(batch_size)], dtype=torch.long
    )
    return result


def _forward(model: V05AlphaModel, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        return model(batch)


# ──────────────────────────────────────────────────────────────────────────────
# 基本 shape / 参数预算
# ──────────────────────────────────────────────────────────────────────────────


def test_tiny_model_output_shapes_and_finite():
    config = _tiny_config()
    model = V05AlphaModel(config).eval()
    batch = _make_batch(config)
    outputs = _forward(model, batch)
    batch_size = int(batch["segment_mask"].shape[0])
    segments = int(batch["segment_mask"].shape[1])
    assert set(outputs) == {
        "work_vector",
        "segment_tokens",
        "mask_pred",
        "av_logits",
        "order_logits",
    }
    assert tuple(outputs["work_vector"].shape) == (batch_size, config.hidden)
    assert tuple(outputs["segment_tokens"].shape) == (
        batch_size,
        segments,
        config.hidden,
    )
    assert tuple(outputs["mask_pred"].shape) == (
        batch_size,
        segments,
        config.paav_dim,
    )
    assert tuple(outputs["av_logits"].shape) == (batch_size,)
    assert tuple(outputs["order_logits"].shape) == (batch_size, config.order_classes)
    for value in outputs.values():
        assert bool(torch.isfinite(value).all())


def test_default_config_parameter_count_under_5m():
    model = V05AlphaModel()
    assert isinstance(model, torch.nn.Module)
    total = sum(parameter.numel() for parameter in model.parameters())
    assert 0 < total < 5_000_000, total
    assert model.config.hidden == 256
    assert model.config.local_layers == 2
    assert model.config.temporal_layers == 2


def test_single_segment_minimal_batch():
    config = _tiny_config()
    model = V05AlphaModel(config).eval()
    batch = _make_batch(config, batch_size=1, segments=1, masked=False)
    outputs = _forward(model, batch)
    assert tuple(outputs["work_vector"].shape) == (1, config.hidden)
    assert tuple(outputs["mask_pred"].shape) == (1, 1, config.paav_dim)
    # order_k=3 > W=1：内部应重复最后一个有效段而不是崩溃
    assert tuple(outputs["order_logits"].shape) == (1, config.order_classes)
    assert bool(torch.isfinite(outputs["order_logits"]).all())


def test_all_bins_masked_still_finite():
    config = _tiny_config()
    model = V05AlphaModel(config).eval()
    batch = _make_batch(config, batch_size=1, segments=2, masked=False)
    for key in ("mask_visual", "mask_audio", "mask_visual_event"):
        batch[key] = torch.zeros_like(batch[key])
    outputs = _forward(model, batch)
    assert bool(torch.isfinite(outputs["work_vector"]).all())
    assert bool(torch.isfinite(outputs["mask_pred"]).all())


# ──────────────────────────────────────────────────────────────────────────────
# padding / masked 内容不变性
# ──────────────────────────────────────────────────────────────────────────────


def test_padding_content_invariance():
    config = _tiny_config()
    model = V05AlphaModel(config).eval()
    batch = _make_batch(config, valid_lengths=[4, 2], segments=5)
    outputs = _forward(model, batch)

    segment_mask = batch["segment_mask"]
    padded = ~segment_mask
    perturbed = dict(batch)
    for key in ("paav", "visual_clip", "visual_dino", "audio", "visual_event"):
        tensor = batch[key].clone()
        # 修改所有 padded 段的内容（数值故意很大）
        tensor[padded] = torch.randn_like(tensor[padded]) * 100.0
        perturbed[key] = tensor
    # 修改无效 bin 的内容（mask=False 处）
    for key, mask_key in (
        ("visual_clip", "mask_visual"),
        ("visual_dino", "mask_visual"),
        ("audio", "mask_audio"),
        ("visual_event", "mask_visual_event"),
    ):
        tensor = perturbed[key].clone()
        invalid = ~batch[mask_key]
        tensor[invalid] = torch.randn_like(tensor[invalid]) * 100.0
        perturbed[key] = tensor

    outputs2 = _forward(model, perturbed)
    for key in outputs:
        assert torch.allclose(
            outputs[key], outputs2[key], atol=1e-6, rtol=0.0
        ), f"{key} 受 padding 内容影响"


def test_masked_segment_content_does_not_leak():
    config = _tiny_config()
    model = V05AlphaModel(config).eval()
    batch = _make_batch(config, valid_lengths=[4, 3], segments=5)
    outputs = _forward(model, batch)
    masked_segments = batch["masked_segments"]

    perturbed = dict(batch)
    for key in ("paav", "visual_clip", "visual_dino", "audio", "visual_event"):
        tensor = batch[key].clone()
        if key == "paav":
            tensor[masked_segments] = torch.randn_like(tensor[masked_segments]) * 50.0
        else:
            tensor[masked_segments] = (
                torch.randn_like(tensor[masked_segments]) * 50.0
            )
        perturbed[key] = tensor

    outputs2 = _forward(model, perturbed)
    for key in outputs:
        assert torch.allclose(
            outputs[key], outputs2[key], atol=1e-6, rtol=0.0
        ), f"masked 段内容泄漏到 {key}"


def test_mask_flag_changes_outputs_and_uses_learned_token():
    config = _tiny_config()
    model = V05AlphaModel(config).eval()
    batch = _make_batch(config, valid_lengths=[4, 3], segments=5)
    masked = _forward(model, batch)
    unmasked_batch = dict(batch)
    unmasked_batch["masked_segments"] = torch.zeros_like(batch["masked_segments"])
    unmasked = _forward(model, unmasked_batch)

    # 关掉 mask 后 work_vector / mask_pred 应该改变（模型能看到目标段）
    assert not torch.allclose(masked["work_vector"], unmasked["work_vector"])
    assert not torch.allclose(masked["mask_pred"], unmasked["mask_pred"])

    segment_mask = batch["segment_mask"]
    mask_token = model.mask_token.view(1, 1, -1)
    expected = mask_token.expand_as(masked["segment_tokens"])
    masked_positions = batch["masked_segments"]
    padded_positions = ~segment_mask
    assert torch.allclose(
        masked["segment_tokens"][masked_positions],
        expected[masked_positions],
        atol=1e-6,
    )
    assert torch.allclose(
        masked["segment_tokens"][padded_positions],
        expected[padded_positions],
        atol=1e-6,
    )
    # 未 masked 的有效段不应被替换
    visible = segment_mask & ~masked_positions
    assert not torch.allclose(
        masked["segment_tokens"][visible],
        expected[visible],
    )


def test_valid_content_perturbation_changes_outputs():
    """防止 padding 不变性测试变成“所有输入都不起作用”。"""

    config = _tiny_config()
    model = V05AlphaModel(config).eval()
    batch = _make_batch(config, valid_lengths=[4, 3], segments=5)
    outputs = _forward(model, batch)
    perturbed = dict(batch)
    perturbed["paav"] = batch["paav"] + 1.0
    outputs2 = _forward(model, perturbed)
    assert not torch.allclose(outputs["work_vector"], outputs2["work_vector"])


# ──────────────────────────────────────────────────────────────────────────────
# 确定性 / 梯度
# ──────────────────────────────────────────────────────────────────────────────


def test_eval_determinism():
    config = _tiny_config(dropout=0.3)
    model = V05AlphaModel(config).eval()
    batch = _make_batch(config)
    first = _forward(model, batch)
    second = _forward(model, batch)
    for key in first:
        assert torch.allclose(first[key], second[key], atol=0.0, rtol=0.0)


def test_gradients_nonzero_for_all_heads():
    config = _tiny_config()
    model = V05AlphaModel(config).train()
    batch = _add_labels(_make_batch(config))
    outputs = model(batch)
    losses = v05_losses(outputs, batch)
    losses["total"].backward()
    assert model.mask_head.weight.grad is not None
    assert float(model.mask_head.weight.grad.abs().sum()) > 0.0
    assert model.av_head[0].weight.grad is not None
    assert float(model.av_head[0].weight.grad.abs().sum()) > 0.0
    assert model.order_head[0].weight.grad is not None
    assert float(model.order_head[0].weight.grad.abs().sum()) > 0.0
    nonzero = sum(
        1
        for parameter in model.parameters()
        if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
    )
    total = sum(1 for _ in model.parameters())
    assert nonzero >= int(total * 0.5), (nonzero, total)


# ──────────────────────────────────────────────────────────────────────────────
# 损失
# ──────────────────────────────────────────────────────────────────────────────


def test_losses_keys_weighting_and_finiteness():
    config = _tiny_config()
    model = V05AlphaModel(config).eval()
    batch = _add_labels(_make_batch(config))
    with torch.no_grad():
        outputs = model(batch)
        losses = v05_losses(outputs, batch)
    assert set(losses) >= {"mask_loss", "av_loss", "order_loss", "total"}
    for value in losses.values():
        assert bool(torch.isfinite(value).all())
    expected_total = (
        losses["mask_loss"] + 0.5 * losses["av_loss"] + 0.5 * losses["order_loss"]
    )
    assert losses["total"] == pytest.approx(float(expected_total), rel=1e-6)
    assert losses["loss"] == pytest.approx(float(expected_total), rel=1e-6)


def test_mask_loss_only_valid_masked_segments():
    config = _tiny_config()
    model = V05AlphaModel(config).eval()
    batch = _add_labels(_make_batch(config, valid_lengths=[4, 3]))
    with torch.no_grad():
        outputs = model(batch)
    losses = v05_losses(outputs, batch)

    target = batch["paav"][..., 0, :]
    valid = batch["segment_mask"] & batch["masked_segments"]
    assert int(valid.sum()) > 0
    prediction = F.normalize(outputs["mask_pred"][valid], dim=-1, eps=1e-8)
    expected = F.normalize(target[valid], dim=-1, eps=1e-8)
    manual = F.mse_loss(prediction, expected) + (
        1.0 - (prediction * expected).sum(dim=-1)
    ).mean()
    assert float(losses["mask_loss"]) == pytest.approx(float(manual), rel=1e-5)

    # 修改非 masked 段的目标不应影响 mask_loss
    changed = dict(batch)
    changed_target = target.clone()
    changed_target[~valid] = changed_target[~valid] + 5.0
    changed["paav"] = batch["paav"].clone()
    changed["paav"][..., 0, :] = changed_target
    losses2 = v05_losses(outputs, changed)
    assert float(losses2["mask_loss"]) == pytest.approx(
        float(losses["mask_loss"]), rel=1e-6
    )

    # 修改 masked 段目标应改变 mask_loss
    changed_target[valid] = changed_target[valid] + 5.0
    changed["paav"][..., 0, :] = changed_target
    losses3 = v05_losses(outputs, changed)
    assert float(losses3["mask_loss"]) != pytest.approx(
        float(losses["mask_loss"]), rel=1e-6
    )


def test_av_and_order_loss_follow_labels():
    config = _tiny_config()
    model = V05AlphaModel(config).eval()
    batch = _add_labels(_make_batch(config, batch_size=4))
    with torch.no_grad():
        outputs = model(batch)
    baseline = v05_losses(outputs, batch)

    flipped = dict(batch)
    flipped["av_label"] = 1.0 - batch["av_label"]
    changed_av = v05_losses(outputs, flipped)
    assert float(changed_av["av_loss"]) != pytest.approx(
        float(baseline["av_loss"]), rel=1e-6
    )

    relabeled = dict(batch)
    relabeled["order_label"] = (batch["order_label"] + 1) % config.order_classes
    changed_order = v05_losses(outputs, relabeled)
    assert float(changed_order["order_loss"]) != pytest.approx(
        float(baseline["order_loss"]), rel=1e-6
    )


def test_losses_without_aux_labels_are_zero():
    config = _tiny_config()
    model = V05AlphaModel(config).eval()
    batch = _make_batch(config)
    with torch.no_grad():
        outputs = model(batch)
        losses = v05_losses(outputs, batch)
    assert float(losses["av_loss"]) == 0.0
    assert float(losses["order_loss"]) == 0.0
    assert float(losses["total"]) == pytest.approx(float(losses["mask_loss"]))


def test_losses_with_no_masked_segments_is_zero():
    config = _tiny_config()
    model = V05AlphaModel(config).eval()
    batch = _add_labels(_make_batch(config, masked=False))
    with torch.no_grad():
        outputs = model(batch)
        losses = v05_losses(outputs, batch)
    assert float(losses["mask_loss"]) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 配置 / 输入校验
# ──────────────────────────────────────────────────────────────────────────────


def test_config_validation_errors():
    with pytest.raises(ValueError, match="整除"):
        V05AlphaConfig(hidden=30, heads=4)
    with pytest.raises(ValueError, match="dropout"):
        V05AlphaConfig(dropout=1.0)
    with pytest.raises(ValueError, match="order_classes"):
        V05AlphaConfig(order_k=3, order_classes=5)
    with pytest.raises(ValueError, match="local_layers"):
        V05AlphaConfig(local_layers=0)
    with pytest.raises(ValueError, match="temporal_layers"):
        V05AlphaConfig(temporal_layers=0)
    with pytest.raises(ValueError, match="bins"):
        V05AlphaConfig(bins=0)
    with pytest.raises(ValueError, match="paav_dim"):
        V05AlphaConfig(paav_dim=0)
    assert math.factorial(3) == 6


def test_forward_rejects_bad_shapes_and_too_many_segments():
    config = _tiny_config(max_segments=4)
    model = V05AlphaModel(config).eval()
    batch = _make_batch(config, batch_size=1, segments=3, valid_lengths=[3])
    bad = dict(batch)
    bad["paav"] = batch["paav"][:, :, :2, :]
    with pytest.raises(ValueError, match="paav"):
        model(bad)
    bad = dict(batch)
    bad["segment_mask"] = batch["segment_mask"][:, :2]
    with pytest.raises(ValueError, match="segment_mask"):
        model(bad)
    bad = dict(batch)
    bad["mask_visual"] = batch["mask_visual"][:, :, :1]
    with pytest.raises(ValueError, match="mask_visual"):
        model(bad)
    too_many = _make_batch(config, batch_size=1, segments=5, valid_lengths=[5])
    with pytest.raises(ValueError, match="max_segments"):
        model(too_many)


def test_order_indices_override():
    config = _tiny_config()
    model = V05AlphaModel(config).eval()
    batch = _make_batch(config)
    batch_with_indices = dict(batch)
    batch_with_indices["order_indices"] = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    outputs = _forward(model, batch_with_indices)
    assert tuple(outputs["order_logits"].shape) == (2, config.order_classes)
    assert bool(torch.isfinite(outputs["order_logits"]).all())

    bad = dict(batch)
    bad["order_indices"] = torch.tensor([[0, 1]], dtype=torch.long)
    with pytest.raises(ValueError, match="order_indices"):
        model(bad)


def test_masked_segments_optional_defaults_to_none():
    config = _tiny_config()
    model = V05AlphaModel(config).eval()
    batch = _make_batch(config, masked=False)
    batch.pop("masked_segments")
    outputs = _forward(model, batch)
    assert tuple(outputs["work_vector"].shape) == (2, config.hidden)
