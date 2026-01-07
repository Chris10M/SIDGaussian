#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import os
import torch.nn as nn
import torch
from gaussian_utils.sh_utils import eval_sh

try:
    from . import _C
except Exception:  # pragma: no cover - fallback when extension is unavailable
    _C = None

_FORCE_TORCH_RASTERIZER = os.getenv("USE_TORCH_RASTERIZER", "0") == "1"

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def _project_points(points, viewmatrix, projmatrix, image_width, image_height, means2D=None):
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    points_hom = torch.cat([points, ones], dim=1)
    cam_hom = points_hom @ viewmatrix
    clip_hom = points_hom @ projmatrix
    clip_w = clip_hom[:, 3:4].clamp(min=1e-8)
    ndc = clip_hom[:, :3] / clip_w
    x = (ndc[:, 0] + 1.0) * 0.5 * image_width
    y = (1.0 - ndc[:, 1]) * 0.5 * image_height
    screen_xy = torch.stack([x, y], dim=1)
    if means2D is not None and means2D.numel() != 0:
        screen_xy = screen_xy + means2D[:, :2]
    return cam_hom[:, :3], screen_xy


def _colors_from_sh(sh, sh_degree, means3D, campos):
    sh_view = sh.transpose(1, 2).contiguous()
    dirs = means3D - campos[None, :]
    dirs = dirs / (dirs.norm(dim=1, keepdim=True).clamp(min=1e-8))
    sh2rgb = eval_sh(sh_degree, sh_view, dirs)
    return torch.clamp_min(sh2rgb + 0.5, 0.0)


def _covariance_diag_from_precomp(cov3D_precomp):
    if cov3D_precomp.numel() == 0:
        return None
    diag = torch.stack(
        [cov3D_precomp[:, 0], cov3D_precomp[:, 3], cov3D_precomp[:, 5]], dim=1
    )
    return diag


def _rasterize_gaussians_torch(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    device = means3D.device
    dtype = means3D.dtype
    height = raster_settings.image_height
    width = raster_settings.image_width
    fx = 0.5 * width / raster_settings.tanfovx
    fy = 0.5 * height / raster_settings.tanfovy

    cam_coords, screen_xy = _project_points(
        means3D,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
        width,
        height,
        means2D=means2D,
    )
    cam_z = cam_coords[:, 2]

    if colors_precomp.numel() == 0:
        colors = _colors_from_sh(
            sh, raster_settings.sh_degree, means3D, raster_settings.campos
        )
    else:
        colors = colors_precomp

    opacities = opacities.view(-1).clamp(min=0.0)

    if scales.numel() != 0:
        sigma_world = raster_settings.scale_modifier * scales
        sigma_world = sigma_world.abs().max(dim=1).values
    else:
        diag = _covariance_diag_from_precomp(cov3Ds_precomp)
        if diag is None:
            raise ValueError("Missing covariance data for torch rasterizer.")
        sigma_world = diag.clamp(min=1e-8).max(dim=1).values.sqrt()

    sigma_x = (fx * sigma_world / cam_z.clamp(min=1e-6)).abs()
    sigma_y = (fy * sigma_world / cam_z.clamp(min=1e-6)).abs()
    radii = 3.0 * torch.maximum(sigma_x, sigma_y)

    color_acc = torch.zeros((3, height, width), device=device, dtype=dtype)
    alpha = torch.zeros((1, height, width), device=device, dtype=dtype)
    depth = torch.zeros((1, height, width), device=device, dtype=dtype)

    order = torch.argsort(cam_z)
    for idx in order.tolist():
        if cam_z[idx] <= 0:
            radii[idx] = 0.0
            continue
        radius = radii[idx]
        if radius <= 0:
            radii[idx] = 0.0
            continue
        cx, cy = screen_xy[idx]
        x0 = int(torch.floor(cx - radius).clamp(min=0).item())
        x1 = int(torch.ceil(cx + radius).clamp(max=width - 1).item())
        y0 = int(torch.floor(cy - radius).clamp(min=0).item())
        y1 = int(torch.ceil(cy + radius).clamp(max=height - 1).item())
        if x1 < x0 or y1 < y0:
            radii[idx] = 0.0
            continue

        xs = torch.arange(x0, x1 + 1, device=device, dtype=dtype)
        ys = torch.arange(y0, y1 + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        dx = (xx - cx) / (sigma_x[idx] + 1e-8)
        dy = (yy - cy) / (sigma_y[idx] + 1e-8)
        weight = torch.exp(-0.5 * (dx * dx + dy * dy))
        alpha_i = opacities[idx] * weight

        alpha_patch = alpha[:, y0 : y1 + 1, x0 : x1 + 1]
        transmittance = 1.0 - alpha_patch
        contrib = transmittance * alpha_i

        color_acc[:, y0 : y1 + 1, x0 : x1 + 1] += (
            colors[idx].view(3, 1, 1) * contrib
        )
        depth[:, y0 : y1 + 1, x0 : x1 + 1] += cam_z[idx] * contrib
        alpha[:, y0 : y1 + 1, x0 : x1 + 1] = alpha_patch + contrib

    bg = raster_settings.bg.to(device=device, dtype=dtype).view(3, 1, 1)
    color = color_acc + bg * (1.0 - alpha)
    return color, radii, depth, alpha


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    if _C is None or _FORCE_TORCH_RASTERIZER:
        confidence = raster_settings.confidence
        if confidence is not None:
            def _scale_grad(grad, scale):
                if grad is None:
                    return None
                while scale.dim() < grad.dim():
                    scale = scale.unsqueeze(-1)
                return grad * scale

            def _register_confidence_hook(tensor, scale):
                if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                    tensor.register_hook(lambda grad, s=scale: _scale_grad(grad, s))

            _register_confidence_hook(means3D, confidence)
            _register_confidence_hook(sh, confidence)
            _register_confidence_hook(colors_precomp, confidence)
            _register_confidence_hook(opacities, confidence)
            _register_confidence_hook(scales, confidence)
            _register_confidence_hook(rotations, confidence)
            _register_confidence_hook(cov3Ds_precomp, confidence)
        return _rasterize_gaussians_torch(
            means3D,
            means2D,
            sh,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings,
        )
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, alpha, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, alpha, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha)
        return color, radii, depth, alpha

    @staticmethod
    def backward(ctx, grad_color, grad_radii, grad_depth, grad_alpha):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_color,
                grad_depth,
                grad_alpha,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                alpha,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        # print(grad_means3D.shape, grad_sh.shape, grad_colors_precomp.shape, grad_opacities.shape, grad_scales.shape, grad_rotations.shape, grad_cov3Ds_precomp.shape, grad_cov3Ds_precomp.shape, raster_settings.confidence.shape)
        grads = (
            grad_means3D * raster_settings.confidence,
            grad_means2D,
            grad_sh * raster_settings.confidence[..., None],
            grad_colors_precomp * raster_settings.confidence,
            grad_opacities * raster_settings.confidence,
            grad_scales * raster_settings.confidence,
            grad_rotations * raster_settings.confidence,
            grad_cov3Ds_precomp * raster_settings.confidence,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    confidence : torch.Tensor

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            if _C is not None and not _FORCE_TORCH_RASTERIZER:
                return _C.mark_visible(
                    positions,
                    raster_settings.viewmatrix,
                    raster_settings.projmatrix,
                )
            _, screen_xy = _project_points(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.image_width,
                raster_settings.image_height,
            )
            in_x = (screen_xy[:, 0] >= 0) & (screen_xy[:, 0] < raster_settings.image_width)
            in_y = (screen_xy[:, 1] >= 0) & (screen_xy[:, 1] < raster_settings.image_height)
            return in_x & in_y

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )
