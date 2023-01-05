from typing import Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional_tensor as TFT
from kornia.geometry.transform import (get_perspective_transform,
                                       warp_perspective)
from numpy import ndarray
from PIL import Image
from torch import Tensor
from torch.nn.modules.utils import _reverse_repeat_tuple

from ffhq_align.utils import (FLOAT, INT, normalize_float_tuple,
                              normalize_int_tuple)


def to_rgb_array(image: Union[Image.Image, ndarray]) -> ndarray:
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    return image


def padding_for_functional_pad(rank: int, padding: INT) -> Tuple[int, ...]:
    """
    Convert padding to a tuple of length 2 * rank for `torch.nn.functional.pad`.

    Examples:
        >>> padding_for_functional_pad(2, 1)
        (1, 1, 1, 1)
        >>> padding_for_functional_pad(2, (1, 2))
        (2, 2, 1, 1)
        >>> padding_for_functional_pad(2, (1, 2, 3, 4))
        (3, 4, 1, 2)
    """
    if isinstance(padding, int):
        return (padding,) * rank * 2
    if len(padding) == rank:
        return cast(Tuple[int, ...], _reverse_repeat_tuple(padding, 2))
    if len(padding) == rank * 2:
        return tuple(
            p for i in range(rank - 1, -1, -1) for p in padding[i * 2 : (i + 1) * 2]
        )
    raise ValueError("Invalid padding: {}".format(padding))


def gaussian_blur(
    image: Tensor, kernel_size: INT, sigma: FLOAT, use_separable: bool = True
) -> Tensor:
    kernel_size = normalize_int_tuple(kernel_size, 2)
    sigma = normalize_float_tuple(sigma, 2)
    pad = []
    for k in kernel_size:
        div, mod = divmod(k - 1, 2)
        pad.extend([div + mod, div])
    image = F.pad(image, tuple(reversed(pad)), mode="reflect")

    if use_separable:
        for k in ((kernel_size[0], 1), (1, kernel_size[1])):
            kernel = TFT._get_gaussian_kernel2d(
                k, sigma, dtype=torch.float32, device=image.device
            )
            kernel = kernel.expand(image.shape[-3], 1, -1, -1)
            image = F.conv2d(image, kernel, groups=image.shape[-3])
    else:
        kernel = TFT._get_gaussian_kernel2d(
            kernel_size, sigma, dtype=torch.float32, device=image.device
        )
        kernel = kernel.expand(image.shape[-3], 1, -1, -1)
        image = F.conv2d(image, kernel, groups=image.shape[-3])
    return image


def blur_pad(
    image: Tensor,
    padding: INT,
    sigma: Optional[float] = None,
    truncate: float = 4.0,
    quad_size: Optional[float] = None,
) -> Tensor:
    """
    Reflect padding and add Gaussian blur to padded area.
    Support backpropagation and GPU acceleration.

    Reference to `scipy.gaussian_filter` and `recreate_aligned_images` of
    https://github.com/NVlabs/ffhq-dataset.

    Args:
        image: (N, C, H, W) image tensor
        padding: Padding size.
            (top, bottom, left, right) or (height, width) or int
        sigma: Sigma of Gaussian kernel.
        truncate: Truncate the Gaussian kernel at this many standard deviations.
            kernel_size = 2 * ceil(sigma * truncate) + 1
        quad_size: length of quadratic points of transform.
            max(diagonal length of quadrilateral) / sqrt(2)

    Returns:
        A tensor of blurred image. (N, C, H + 2 * padding, W + 2 * padding)
    """
    padding = padding_for_functional_pad(2, padding)
    padded_image = F.pad(image.float(), padding, mode="reflect")
    if quad_size is None:
        quad_size = cast(float, np.hypot(*padded_image.shape[-2:]) / np.sqrt(2))
    if sigma is None:
        sigma = quad_size * 0.02
    kernel_size = 2 * int(np.ceil(sigma * truncate)) + 1
    blurred_image = gaussian_blur(
        padded_image, kernel_size=(kernel_size,) * 2, sigma=(sigma,) * 2
    )

    l, r, t, b = padding
    L, R, T, B = np.maximum(padding, max(quad_size * 0.3, 1)).astype(int)
    H = image.size(-2) + T + B
    W = image.size(-1) + L + R
    h_mask = torch.arange(H, dtype=torch.float32, device=image.device)
    w_mask = torch.arange(W, dtype=torch.float32, device=image.device)
    h_mask = torch.minimum(h_mask / T, h_mask.flip(0) / B)
    w_mask = torch.minimum(w_mask / L, w_mask.flip(0) / R)
    mask = 1 - torch.minimum(h_mask.unsqueeze(-1), w_mask.unsqueeze(0))
    # crop
    mask = F.pad(mask, (-L + l, -R + r, -T + t, -B + b))

    padded_image = torch.lerp(
        input=padded_image,
        end=blurred_image,
        weight=(mask * 3.0 + 1.0).clip(0.0, 1.0),
    )
    # ffhq-dataset uses median. But torch.median not operate along axis.
    padded_image = torch.lerp(
        input=padded_image,
        end=padded_image.mean(dim=(-2, -1), keepdim=True),
        weight=mask.clip(0.0, 1.0),
    )
    return padded_image.to(dtype=image.dtype)


def quad_transform(
    image: Tensor,
    quad: Tensor,
    resolution: INT,
    mode: str = "bilinear",
    padding_mode: str = "reflection",
    align_corners: bool = False,
) -> Tensor:
    """
    Transform image to a quad.

    Args:
        image: (N, C, H, W) image tensor
        quad: (N, 4, 2) quad tensor
            1-dim order: (left-top, left-bottom, right-bottom, right-top)
        resolution: Resolution of output image. (height, width) or int
        mode: Interpolation mode. One of {"nearest", "bilinear"}.
            Default: "bilinear"
        padding_mode: One of {"zeros", "border", "reflection", "blur"}.
            Default: "reflection"
        align_corners: If True, the corner pixels of the input and output
            tensors are aligned, and thus preserving the values at the corner
            pixels. Default: False

    Returns:
        A tensor of transformed image. (N, C, resolution, resolution)
    """
    is_batch = image.ndim == 4
    if not is_batch:
        image = image.unsqueeze(0)
    if quad.ndim == 2:
        quad = quad.unsqueeze(0)
    NI, NQ = image.size(0), quad.size(0)
    if NI != NQ:
        if NQ == 1:
            quad = quad.expand(NI, -1, -1)
        else:
            raise ValueError(
                "The batch size of image and quad must be same. "
                f"image: {NI}, quad: {NQ}."
            )
    if padding_mode == "blur":
        padding_mode = "zeros"
        with torch.no_grad():
            LT_RB = torch.norm(quad[:, 0] - quad[:, 2], dim=-1)
            LB_RT = torch.norm(quad[:, 1] - quad[:, 3], dim=-1)
            quad_sizes = torch.max(LT_RB, LB_RT).mul(1 / np.sqrt(2)).tolist()
            points = quad.flatten(0, 1).sort(0)[0]
            L = max(0, int(torch.ceil(-points[0][0]).item()))
            T = max(0, int(torch.ceil(-points[0][1]).item()))
            R = max(0, torch.ceil(points[-1][0] - image.size(-1)).int().item())
            B = max(0, torch.ceil(points[-1][1] - image.size(-2)).int().item())
        padding = cast(Tuple[int, ...], (T, B, L, R))
        image = torch.stack(
            tuple(
                blur_pad(
                    image[i],
                    padding=padding,
                    sigma=None,
                    quad_size=quad_sizes[i],
                )
                for i in range(NI)
            ),
            dim=0,
        )
        quad = quad + torch.tensor([[[L, T]]], dtype=quad.dtype, device=quad.device)
    res_H, res_W = normalize_int_tuple(resolution, 2)
    destination = torch.tensor(
        [[0, 0], [0, res_H - 1], [res_W - 1, res_H - 1], [res_W - 1, 0]],
        dtype=torch.float32,
        device=image.device,
    ).expand(NI, -1, -1)
    transform_matrix = get_perspective_transform(quad, destination)
    output = warp_perspective(
        src=image,
        M=transform_matrix,
        dsize=(res_H, res_W),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    if is_batch:
        return output
    return output.squeeze(0)


def image_align(
    image: Tensor,
    landmarks: Tensor,
    resolution: INT = 512,
    padding_mode: str = "blur",
):
    """
    image: (N, C, H, W) image tensor
    landmarks: (N, 68, 2) landmarks tensor
    """
    landmarks = landmarks.float()
    if landmarks.ndim == 2:
        landmarks = landmarks.unsqueeze(0)

    # lm_chin          = landmarks[:, 0  : 17, :2]  # left-right
    # lm_eyebrow_left  = landmarks[:, 17 : 22, :2]  # left-right
    # lm_eyebrow_right = landmarks[:, 22 : 27, :2]  # left-right
    # lm_nose          = landmarks[:, 27 : 31, :2]  # top-down
    # lm_nostrils      = landmarks[:, 31 : 36, :2]  # top-down
    lm_eye_left = landmarks[:, 36:42, :2]  # left-clockwise
    lm_eye_right = landmarks[:, 42:48, :2]  # left-clockwise
    lm_mouth_outer = landmarks[:, 48:60, :2]  # left-clockwise
    # lm_mouth_inner   = landmarks[:, 60 : 68, :2]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = lm_eye_left.mean(1)
    eye_right = lm_eye_right.mean(1)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[:, 0]
    mouth_right = lm_mouth_outer[:, 6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    f = torch.tensor([-1, 1], dtype=landmarks.dtype, device=landmarks.device)
    x = eye_to_eye - eye_to_mouth.flip(-1) * f
    x: Tensor = x / x.norm(dim=-1)
    x = x * torch.max(eye_to_eye.norm(dim=-1) * 2.0, eye_to_mouth.norm(dim=-1) * 1.8)
    y = x.flip(-1) * f
    # center
    c = eye_avg + eye_to_mouth * 0.1

    # point of left up, left down, right down, right up
    quad = torch.stack([c - x - y, c - x + y, c + x + y, c + x - y], 1)
    return quad_transform(image, quad, resolution, padding_mode=padding_mode)
