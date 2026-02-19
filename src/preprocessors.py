"""
Image preprocessing utilities for ControlNet conditioning.

Provides preprocessors for extracting control signals from input images:
- Canny edge detection
- Monocular depth estimation
- OpenPose skeleton detection
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def resize_and_pad(
    image: Image.Image,
    target_width: int,
    target_height: int,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
    mode: str = "resize",
) -> Image.Image:
    """Resize an image to target dimensions while handling aspect ratio.

    Args:
        image: Input PIL Image.
        target_width: Desired output width.
        target_height: Desired output height.
        pad_color: RGB color for padding areas.
        mode: Resize strategy:
            - 'resize': Direct resize (may distort)
            - 'fit': Resize to fit within bounds, pad remaining area
            - 'crop': Resize to cover bounds, center crop

    Returns:
        Resized PIL Image with the specified dimensions.
    """
    if mode == "resize":
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    src_w, src_h = image.size
    src_aspect = src_w / src_h
    tgt_aspect = target_width / target_height

    if mode == "fit":
        if src_aspect > tgt_aspect:
            new_w = target_width
            new_h = int(target_width / src_aspect)
        else:
            new_h = target_height
            new_w = int(target_height * src_aspect)

        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_width, target_height), pad_color)
        offset_x = (target_width - new_w) // 2
        offset_y = (target_height - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas

    elif mode == "crop":
        if src_aspect > tgt_aspect:
            new_h = target_height
            new_w = int(target_height * src_aspect)
        else:
            new_w = target_width
            new_h = int(target_width / src_aspect)

        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - target_width) // 2
        top = (new_h - target_height) // 2
        return resized.crop((left, top, left + target_width, top + target_height))

    raise ValueError(f"Unknown resize mode: {mode}")


def ensure_multiple_of(value: int, multiple: int = 8) -> int:
    """Round a value down to the nearest multiple."""
    return (value // multiple) * multiple


def normalize_image_array(array: np.ndarray) -> np.ndarray:
    """Normalize an array to 0-255 uint8 range."""
    if array.dtype == np.uint8:
        return array
    arr_min = array.min()
    arr_max = array.max()
    if arr_max - arr_min < 1e-6:
        return np.zeros_like(array, dtype=np.uint8)
    normalized = ((array - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
    return normalized


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR format."""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL Image."""
    if len(image.shape) == 2:
        return Image.fromarray(image).convert("RGB")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


class BasePreprocessor(ABC):
    """Abstract base class for ControlNet image preprocessors."""

    @abstractmethod
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str],
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> Image.Image:
        """Process an image and return a control image.

        Args:
            image: Input image as PIL Image, numpy array, or file path.
            target_width: Optional output width override.
            target_height: Optional output height override.

        Returns:
            Control image as PIL Image in RGB format.
        """
        pass

    def _load_image(self, image: Union[Image.Image, np.ndarray, str]) -> Image.Image:
        """Load an image from various input types."""
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image.convert("RGB")


class CannyPreprocessor(BasePreprocessor):
    """Canny edge detection preprocessor.

    Extracts edge maps using the Canny algorithm with adjustable
    low and high thresholds for controlling edge sensitivity.

    Args:
        low_threshold: Lower bound for hysteresis thresholding (0-255).
        high_threshold: Upper bound for hysteresis thresholding (0-255).
        aperture_size: Aperture size for the Sobel operator (3, 5, or 7).
        l2_gradient: Whether to use L2 norm for gradient magnitude.
    """

    def __init__(
        self,
        low_threshold: int = 100,
        high_threshold: int = 200,
        aperture_size: int = 3,
        l2_gradient: bool = False,
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.aperture_size = aperture_size
        self.l2_gradient = l2_gradient

        if not (0 <= low_threshold < high_threshold <= 255):
            raise ValueError(
                f"Thresholds must satisfy 0 <= low ({low_threshold}) "
                f"< high ({high_threshold}) <= 255"
            )

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str],
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> Image.Image:
        """Extract Canny edges from the input image.

        Args:
            image: Input image.
            target_width: If provided, resize output to this width.
            target_height: If provided, resize output to this height.

        Returns:
            Edge map as a 3-channel PIL Image (white edges on black background).
        """
        pil_image = self._load_image(image)

        if target_width and target_height:
            pil_image = resize_and_pad(pil_image, target_width, target_height, mode="fit")

        cv_image = pil_to_cv2(pil_image)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(
            blurred,
            self.low_threshold,
            self.high_threshold,
            apertureSize=self.aperture_size,
            L2gradient=self.l2_gradient,
        )

        edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edge_rgb)

    def __repr__(self) -> str:
        return (
            f"CannyPreprocessor("
            f"low={self.low_threshold}, "
            f"high={self.high_threshold})"
        )


class DepthPreprocessor(BasePreprocessor):
    """Monocular depth estimation preprocessor.

    Uses MiDaS or Depth Anything models to estimate depth from a single
    RGB image. The output is a depth map suitable for ControlNet conditioning.

    Args:
        model_type: MiDaS model variant ('DPT_Large', 'DPT_Hybrid', 'MiDaS_small')
            or 'depth_anything' for the Depth Anything model.
        device: Target compute device.
    """

    def __init__(
        self,
        model_type: str = "DPT_Large",
        device: Optional[str] = None,
    ):
        self.model_type = model_type
        self.model = None
        self.transform = None
        self._loaded = False

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    def _load_model(self) -> None:
        """Lazy-load the depth estimation model."""
        if self._loaded:
            return

        logger.info("Loading depth model: %s", self.model_type)

        if self.model_type == "depth_anything":
            from transformers import pipeline as hf_pipeline
            self.model = hf_pipeline(
                "depth-estimation",
                model="LiheYoung/depth-anything-base-hf",
                device=self.device,
            )
        else:
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self.model = self.model.to(self.device).eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if self.model_type in ("DPT_Large", "DPT_Hybrid"):
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

        self._loaded = True
        logger.info("Depth model loaded successfully")

    @torch.no_grad()
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str],
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> Image.Image:
        """Estimate depth from the input image.

        Args:
            image: Input RGB image.
            target_width: If provided, resize output to this width.
            target_height: If provided, resize output to this height.

        Returns:
            Depth map as a 3-channel PIL Image (closer = brighter).
        """
        self._load_model()
        pil_image = self._load_image(image)

        if self.model_type == "depth_anything":
            result = self.model(pil_image)
            depth_map = np.array(result["depth"])
        else:
            cv_image = pil_to_cv2(pil_image)
            rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            input_batch = self.transform(rgb).to(self.device)

            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_map = prediction.cpu().numpy()

        depth_normalized = normalize_image_array(depth_map)
        depth_rgb = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2RGB)
        depth_pil = Image.fromarray(depth_rgb)

        if target_width and target_height:
            depth_pil = resize_and_pad(depth_pil, target_width, target_height, mode="fit")

        return depth_pil

    def __repr__(self) -> str:
        return f"DepthPreprocessor(model={self.model_type}, device={self.device})"


class OpenposePreprocessor(BasePreprocessor):
    """OpenPose skeleton detection preprocessor.

    Detects human body pose keypoints and renders a skeleton visualization
    suitable for ControlNet conditioning.

    Args:
        include_hand: Whether to detect hand keypoints.
        include_face: Whether to detect face keypoints.
        device: Target compute device.
    """

    def __init__(
        self,
        include_hand: bool = False,
        include_face: bool = False,
        device: Optional[str] = None,
    ):
        self.include_hand = include_hand
        self.include_face = include_face
        self.detector = None
        self._loaded = False

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    def _load_model(self) -> None:
        """Lazy-load the OpenPose detector."""
        if self._loaded:
            return

        logger.info("Loading OpenPose detector")
        from controlnet_aux import OpenposeDetector

        self.detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        self._loaded = True
        logger.info("OpenPose detector loaded successfully")

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str],
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> Image.Image:
        """Detect pose keypoints and render skeleton visualization.

        Args:
            image: Input image containing one or more people.
            target_width: If provided, resize output to this width.
            target_height: If provided, resize output to this height.

        Returns:
            Skeleton visualization as a 3-channel PIL Image.
        """
        self._load_model()
        pil_image = self._load_image(image)

        if target_width and target_height:
            pil_image = resize_and_pad(pil_image, target_width, target_height, mode="fit")

        pose_image = self.detector(
            pil_image,
            hand_and_face=self.include_hand or self.include_face,
        )

        if pose_image.mode != "RGB":
            pose_image = pose_image.convert("RGB")

        return pose_image

    def __repr__(self) -> str:
        return (
            f"OpenposePreprocessor("
            f"hand={self.include_hand}, "
            f"face={self.include_face})"
        )


class PreprocessorFactory:
    """Factory for creating preprocessor instances by control type name.

    Usage:
        preprocessor = PreprocessorFactory.create("canny", low_threshold=80)
        control_image = preprocessor(input_image)
    """

    _registry = {
        "canny": CannyPreprocessor,
        "depth": DepthPreprocessor,
        "pose": OpenposePreprocessor,
    }

    @classmethod
    def create(cls, control_type: str, **kwargs) -> BasePreprocessor:
        """Create a preprocessor instance for the given control type.

        Args:
            control_type: One of 'canny', 'depth', 'pose'.
            **kwargs: Additional arguments passed to the preprocessor constructor.

        Returns:
            Preprocessor instance.

        Raises:
            ValueError: If control_type is not recognized.
        """
        if control_type not in cls._registry:
            raise ValueError(
                f"Unknown control type '{control_type}'. "
                f"Available types: {list(cls._registry.keys())}"
            )
        return cls._registry[control_type](**kwargs)

    @classmethod
    def available_types(cls) -> list:
        """Return list of available preprocessor types."""
        return list(cls._registry.keys())


def preprocess_image(
    image: Union[Image.Image, np.ndarray, str],
    control_type: str,
    target_width: int = 1024,
    target_height: int = 1024,
    **kwargs,
) -> Image.Image:
    """Convenience function to preprocess an image for ControlNet.

    Args:
        image: Input image.
        control_type: Type of control preprocessing.
        target_width: Output width.
        target_height: Output height.
        **kwargs: Additional preprocessor-specific arguments.

    Returns:
        Preprocessed control image.
    """
    preprocessor = PreprocessorFactory.create(control_type, **kwargs)
    return preprocessor(image, target_width=target_width, target_height=target_height)
