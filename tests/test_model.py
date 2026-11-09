"""Unit tests for ``src.model`` (Stage 4)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch import nn

from src.config import NUM_CLASSES, PATHOLOGIES, TrainConfig
from src.model import (
    ActivationCapture,
    activation_and_gradient,
    build_model,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture(scope="module")
def densenet() -> nn.Module:
    torch.manual_seed(0)
    return build_model("densenet121", pretrained=False).eval()


def test_densenet_head_and_output_shape(densenet: nn.Module) -> None:
    head = densenet.classifier
    assert isinstance(head[0], nn.Dropout) and head[0].p == 0.2
    assert isinstance(head[1], nn.Linear) and head[1].in_features == 1024
    assert head[1].out_features == NUM_CLASSES == len(PATHOLOGIES)
    with torch.no_grad():
        assert densenet(torch.randn(2, 3, 224, 224)).shape == (2, NUM_CLASSES)


def test_resnet50_baseline_has_same_head_design() -> None:
    model = build_model("resnet50", pretrained=False).eval()
    assert isinstance(model.fc[0], nn.Dropout) and isinstance(model.fc[1], nn.Linear)
    with torch.no_grad():
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, NUM_CLASSES)
    assert count_parameters(model) > 20_000_000


def test_unknown_backbone_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_model("vgg16", pretrained=False)


def test_densenet_activation_and_gradient_shapes(densenet: nn.Module) -> None:
    logits, activation, gradient = activation_and_gradient(
        densenet, torch.randn(1, 3, 224, 224), class_index=7, backbone="densenet121")
    assert logits.shape == (1, NUM_CLASSES)
    assert activation.shape == (1, 1024, 7, 7) and gradient.shape == activation.shape
    assert activation.min() >= 0  # ReLU is applied after norm5, as in DenseNet.forward
    assert gradient.abs().sum() > 0


def test_hooks_are_removed_after_use(densenet: nn.Module) -> None:
    activation_and_gradient(densenet, torch.randn(1, 3, 224, 224), 0, "densenet121")
    assert len(densenet.features.norm5._forward_hooks) == 0
    with ActivationCapture(densenet, "features.norm5", relu_after=True):
        assert len(densenet.features.norm5._forward_hooks) == 1
    assert len(densenet.features.norm5._forward_hooks) == 0


def test_hooks_are_removed_even_when_forward_fails(densenet: nn.Module) -> None:
    with pytest.raises(RuntimeError):
        with ActivationCapture(densenet, "features.norm5", relu_after=True):
            densenet(torch.randn(1, 1, 224, 224))  # wrong channel count
    assert len(densenet.features.norm5._forward_hooks) == 0


def test_gradients_differ_between_classes(densenet: nn.Module) -> None:
    image = torch.randn(1, 3, 224, 224)
    _, _, grad_a = activation_and_gradient(densenet, image, 0, "densenet121")
    _, _, grad_b = activation_and_gradient(densenet, image, 7, "densenet121")
    assert not torch.allclose(grad_a, grad_b)


def test_per_image_class_indices_match_single_calls(densenet: nn.Module) -> None:
    images = torch.randn(2, 3, 224, 224)
    _, _, batched = activation_and_gradient(densenet, images, torch.tensor([3, 9]), "densenet121")
    _, _, first = activation_and_gradient(densenet, images[:1], 3, "densenet121")
    _, _, second = activation_and_gradient(densenet, images[1:], 9, "densenet121")
    assert torch.allclose(batched[0], first[0], atol=1e-5)
    assert torch.allclose(batched[1], second[0], atol=1e-5)


def test_alternative_target_layer_path(densenet: nn.Module) -> None:
    _, activation, gradient = activation_and_gradient(
        densenet, torch.randn(1, 3, 224, 224), 0, "densenet121",
        layer_path="features.denseblock4.denselayer16.conv2")
    assert activation.shape == (1, 32, 7, 7) and gradient.shape == activation.shape


def test_resnet_activation_shape() -> None:
    model = build_model("resnet50", pretrained=False).eval()
    _, activation, gradient = activation_and_gradient(
        model, torch.randn(1, 3, 224, 224), 2, "resnet50")
    assert activation.shape == (1, 2048, 7, 7) and gradient.shape == activation.shape


def test_checkpoint_roundtrip_keeps_outputs(densenet: nn.Module, tmp_path: Path) -> None:
    path = tmp_path / "ckpt" / "model.pth"
    config = TrainConfig(seed=43)
    save_checkpoint(path, densenet, config, extra={"best_epoch": 3})
    loaded, metadata = load_checkpoint(path)
    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(densenet(image), loaded(image), atol=1e-6)
    assert metadata["classes"] == list(PATHOLOGIES)
    assert metadata["config"]["seed"] == 43 and metadata["extra"]["best_epoch"] == 3
    assert not loaded.training


@pytest.mark.skipif(not os.environ.get("CHESTRADIX_TEST_PRETRAINED"),
                    reason="needs internet to download ImageNet weights")
def test_pretrained_weights_load() -> None:
    torch.manual_seed(1)
    random_init = build_model("densenet121", pretrained=False)
    pretrained = build_model("densenet121", pretrained=True)
    difference = (pretrained.features.conv0.weight - random_init.features.conv0.weight).abs().mean()
    assert difference > 1e-3