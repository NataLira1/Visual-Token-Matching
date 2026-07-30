import torch

from vtm.data import EpisodeSampler, build_manifest, make_synthetic_taskonomy
from vtm.engine import adapt_vtm
from vtm.model import VTM


def make_model() -> VTM:
    return VTM(
        image_size=32,
        train_tasks=("a", "b"),
        backbone="tiny_cnn",
        pretrained=False,
        feature_blocks=(0, 1, 2, 3),
        label_depth=4,
        num_heads=4,
        decoder_width=32,
        embed_dim=32,
        patch_size=4,
    )


def test_forward_shapes_for_all_shots() -> None:
    model = make_model().eval()
    for shots in (1, 5, 10):
        query = torch.randn(2, 3, 32, 32)
        support_images = torch.randn(2, shots, 3, 32, 32)
        support_masks = torch.randint(0, 2, (2, shots, 1, 32, 32)).float()
        logits, attention = model(
            query,
            support_images,
            support_masks,
            task="a",
            return_attention=True,
        )
        assert logits.shape == (2, 1, 32, 32)
        assert len(attention) == 4
        assert attention[-1].shape == (2, 4, 64, shots * 64)


def test_backbone_is_frozen_and_inference_is_deterministic() -> None:
    torch.manual_seed(0)
    model = make_model().eval()
    assert not any(parameter.requires_grad for parameter in model.image_encoder.backbone.parameters())
    query = torch.randn(1, 3, 32, 32)
    support_images = torch.randn(1, 1, 3, 32, 32)
    support_masks = torch.randint(0, 2, (1, 1, 1, 32, 32)).float()
    first = model(query, support_images, support_masks, task="a")
    second = model(query, support_images, support_masks, task="a")
    torch.testing.assert_close(first, second)


def test_adaptation_changes_only_external_task_bias(tmp_path) -> None:
    make_synthetic_taskonomy(tmp_path / "data", image_size=32)
    records = build_manifest(tmp_path / "data", tmp_path / "manifest.json", [6])
    sampler = EpisodeSampler(records, 32, 0.01, 0, seed=0)
    support = sampler.pool("train", 6)[:1]
    model = make_model().eval()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    bias = adapt_vtm(model, sampler, support, 6, 1, 0.01, torch.device("cpu"))
    assert bias.shape == model.bias_shape
    assert bias.abs().sum() > 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
