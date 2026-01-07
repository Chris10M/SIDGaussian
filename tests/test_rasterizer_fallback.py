from pathlib import Path
import sys

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if torch is not None:
    sys.path.append(str(REPO_ROOT / "submodules" / "diff-gaussian-rasterization-confidence"))
    import diff_gaussian_rasterization as dgr
else:
    dgr = None


def _build_inputs(device):
    torch.manual_seed(0)
    means3d = torch.tensor(
        [[0.0, 0.0, 2.0], [0.2, -0.1, 2.5], [-0.3, 0.1, 3.0]],
        device=device,
        dtype=torch.float32,
    )
    means2d = torch.zeros_like(means3d, requires_grad=True)
    colors = torch.tensor(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
        device=device,
        dtype=torch.float32,
    )
    opacities = torch.tensor([[0.6], [0.7], [0.5]], device=device, dtype=torch.float32)
    scales = torch.tensor(
        [[0.1, 0.08, 0.12], [0.12, 0.09, 0.1], [0.08, 0.1, 0.09]],
        device=device,
        dtype=torch.float32,
    )
    rotations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    bg = torch.zeros(3, device=device, dtype=torch.float32)
    return means3d, means2d, colors, opacities, scales, rotations, bg


def _build_settings(device):
    return dgr.GaussianRasterizationSettings(
        image_height=32,
        image_width=32,
        tanfovx=0.5,
        tanfovy=0.5,
        bg=torch.zeros(3, device=device, dtype=torch.float32),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device=device, dtype=torch.float32),
        projmatrix=torch.eye(4, device=device, dtype=torch.float32),
        sh_degree=0,
        campos=torch.zeros(3, device=device, dtype=torch.float32),
        prefiltered=False,
        debug=False,
        confidence=torch.ones(3, 1, device=device, dtype=torch.float32),
    )


def test_torch_rasterizer_outputs_shapes():
    if torch is None or dgr is None:
        pytest.skip("torch is not installed.")
    device = torch.device("cpu")
    means3d, means2d, colors, opacities, scales, rotations, bg = _build_inputs(device)
    settings = _build_settings(device)._replace(bg=bg)

    dgr._FORCE_TORCH_RASTERIZER = True
    rasterizer = dgr.GaussianRasterizer(settings)
    color, radii, depth, alpha = rasterizer(
        means3D=means3d,
        means2D=means2d,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )

    assert color.shape == (3, settings.image_height, settings.image_width)
    assert radii.shape == (means3d.shape[0],)
    assert depth.shape == (1, settings.image_height, settings.image_width)
    assert alpha.shape == (1, settings.image_height, settings.image_width)
    color.sum().backward()
    assert means2d.grad is not None


def test_confidence_scales_grads_in_torch_path():
    if torch is None or dgr is None:
        pytest.skip("torch is not installed.")

    def _run_with_confidence(confidence_value):
        device = torch.device("cpu")
        means3d, means2d, colors, opacities, scales, rotations, bg = _build_inputs(device)
        means3d = means3d.clone().requires_grad_(True)
        colors = colors.clone().requires_grad_(True)
        opacities = opacities.clone().requires_grad_(True)
        scales = scales.clone().requires_grad_(True)
        rotations = rotations.clone().requires_grad_(True)
        settings = _build_settings(device)._replace(
            bg=bg,
            confidence=torch.full((means3d.shape[0], 1), confidence_value, device=device),
        )
        rasterizer = dgr.GaussianRasterizer(settings)
        dgr._FORCE_TORCH_RASTERIZER = True
        color, _, _, _ = rasterizer(
            means3D=means3d,
            means2D=means2d,
            colors_precomp=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        )
        color.sum().backward()
        return means3d.grad, colors.grad, opacities.grad, scales.grad, rotations.grad

    grads_full = _run_with_confidence(1.0)
    grads_half = _run_with_confidence(0.5)

    for grad_half, grad_full in zip(grads_half, grads_full):
        assert torch.allclose(grad_half, grad_full * 0.5, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    dgr is None or torch is None or dgr._C is None or not torch.cuda.is_available(),
    reason="CUDA rasterizer not available for comparison.",
)
def test_torch_matches_cuda_outputs_close():
    if torch is None or dgr is None:
        pytest.skip("torch is not installed.")
    device = torch.device("cuda")
    means3d, means2d, colors, opacities, scales, rotations, bg = _build_inputs(device)
    settings = _build_settings(device)._replace(bg=bg, confidence=torch.ones(3, 1, device=device))

    rasterizer = dgr.GaussianRasterizer(settings)
    dgr._FORCE_TORCH_RASTERIZER = True
    torch_out = rasterizer(
        means3D=means3d,
        means2D=means2d,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )
    dgr._FORCE_TORCH_RASTERIZER = False
    cuda_out = rasterizer(
        means3D=means3d,
        means2D=means2d,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )

    for torch_tensor, cuda_tensor in zip(torch_out, cuda_out):
        assert torch_tensor.shape == cuda_tensor.shape
        assert torch.allclose(torch_tensor, cuda_tensor, atol=1e-2, rtol=1e-2)
