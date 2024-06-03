from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from PIL import Image
from torch import Tensor

from ffhq_align.ops import image_align, to_rgb_array

try:
    import dlib
except ImportError:
    dlib = None
try:
    import face_alignment
except ImportError:
    face_alignment = None


class LandmarkDLIB:
    def __init__(
        self, model_path: str = "shape_predictor_68_face_landmarks.dat"
    ) -> None:
        super().__init__()
        self.detect_net = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(model_path)

    def single_predict(
        self, image: Image.Image, max_resolution: int = 1024
    ) -> Optional[ndarray]:
        original_image = image
        W, H = image.size
        ratio = min(max_resolution / max(W, H), 1.0)
        if ratio < 1.0:
            image = image.resize(
                size=(int(W * ratio), int(H * ratio)), resample=Image.ANTIALIAS
            )
        bbox, score, idx = self.detect_net.run(to_rgb_array(image), 1, 0)
        number_of_faces = len(bbox)
        if number_of_faces == 0:
            return None

        # predict the first face only
        if ratio < 1.0:
            bbox = dlib.rectangle(
                round(bbox[0].left() / ratio),
                round(bbox[0].top() / ratio),
                round(bbox[0].right() / ratio),
                round(bbox[0].bottom() / ratio),
            )
        else:
            bbox = bbox[0]
        shape = self.shape_predictor(to_rgb_array(original_image), bbox)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        return landmarks

    @torch.no_grad()
    def __call__(self, input: Tensor) -> List[Optional[ndarray]]:
        """
        input: (N, 3, H, W), range [-1, 1]
        """
        if input.ndim == 3:
            input = input.unsqueeze(0)
        C = input.size(1)
        if C == 1:
            input = input.expand(-1, 3, -1, -1)
        elif C == 4:
            input = input[:, :3, :, :]
        else:
            raise ValueError(f"Invalid number of channels: {C}")
        input = input.clamp(-1, 1).mul(127.5).add(127.5).permute(0, 2, 3, 1)
        landmarks = []
        for image in input.byte().cpu().numpy():
            pil_image = Image.fromarray(image)
            landmarks.append(self.single_predict(pil_image, 1024))
        return landmarks


class LandmarkFA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        try:
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                device=self.device.type,
                flip_input=False,
            )
        except Exception:
            raise RuntimeError(
                "face_alignment is not installed. "
                "Please install it by `pip install face-alignment`."
            )

    def _apply(self, fn: Callable[..., Any]) -> "LandmarkFA":
        if "t" in fn.__code__.co_varnames:
            with torch.no_grad():
                device = getattr(fn(torch.empty(0)), "device", "cpu")
            self.device = torch.device(device)
            if self.device.type == "cuda":
                torch.backends.cudnn.benchmark = True
            self.fa.device = self.device.type
            self.fa.face_detector.device = self.device.type

            self.fa.face_alignment_net.to(device=self.device)
            self.fa.face_detector.face_detector.to(device=self.device)
        return super()._apply(fn)

    def train(self, mode: bool = False) -> "LandmarkFA":
        return super().train(False)

    def forward(self, images: Tensor) -> List[Optional[ndarray]]:
        """
        images: (N, C, H, W), range [-1, 1]
        """
        if images.ndim == 3:
            images = images.unsqueeze(0)
        C = images.size(1)
        if C == 1:
            images = images.expand(-1, 3, -1, -1)
        elif C == 4:
            images = images[:, :3, :, :]
        elif C != 3:
            raise ValueError(f"Invalid number of channels: {C}")
        images = images.clamp(-1, 1).mul(127.5).add(127.5)
        landmarks = self.fa.get_landmarks_from_batch(images)
        # If no face detected, face_alignment returns [], not None.
        # If multiple faces detected, face_alignment returns
        # array of shape (multiple * 68, 2), not (multiple, 68, 2).
        # Use the first face only.
        landmarks = [None if isinstance(l, list) else l[:68] for l in landmarks]
        return landmarks


class Aligner(nn.Module):
    def __init__(self, padding_mode: str = "blur"):
        super().__init__()
        self.landmark_model = LandmarkFA()
        self.padding_mode = padding_mode

    def train(self, mode: bool = False) -> "Aligner":
        return super().train(False)

    @torch.no_grad()
    def forward(
        self, images: Tensor, resolution: int = 512, quad_scale: float = 1.0
    ) -> Union[Tensor, Tuple[Optional[Tensor], ...]]:
        """
        images: (N, C, H, W), range [-1, 1]
        resolution: resolution of output images
        quad_scale: scale of the bounding box
            If larger, the face will be smaller in the output image.

        Return:
            if all faces are detected:
                (N, C, resolution, resolution), range [-1, 1]
            else:
                tuple((C, resolution, resolution), ..., None), range [-1, 1]
        """
        if isinstance(images, str):
            images = Image.open(images)
        if isinstance(images, Image.Image):
            images = torch.from_numpy(np.array(images)).permute(2, 0, 1)
            images = images.float().div(127.5).sub(1.0).unsqueeze(0)
        images = images.to(self.landmark_model.device)
        landmarks = self.landmark_model(images)
        # If all faces are detected, run batch processing
        if all(l is not None for l in landmarks):
            landmarks = torch.from_numpy(np.stack(landmarks, 0)).to(images)
            return image_align(
                image=images,
                landmarks=landmarks,
                resolution=resolution,
                padding_mode=self.padding_mode,
                quad_scale=quad_scale,
            )

        aligned_images: List[Optional[Tensor]] = []
        for image, landmark in zip(images, landmarks):
            if landmark is None:
                aligned_images.append(None)
                continue
            aligned_images.append(
                image_align(
                    image=image,
                    landmarks=torch.tensor(landmark).to(image),
                    resolution=resolution,
                    padding_mode=self.padding_mode,
                )
            )
        return tuple(aligned_images)
