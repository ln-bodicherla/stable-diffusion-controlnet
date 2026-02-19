"""
ControlNet Pipeline for Stable Diffusion XL.

Provides a unified interface for generating images with spatial control
using Canny edges, depth maps, and OpenPose skeleton conditioning.
"""

import logging
from pathlib import Path
from typing import Optional, Union, List

import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image

logger = logging.getLogger(__name__)

CONTROLNET_MODEL_MAP = {
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
    "pose": "thibaud/controlnet-openpose-sdxl-1.0",
}

SCHEDULER_MAP = {
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpm++": DPMSolverMultistepScheduler,
    "unipc": UniPCMultistepScheduler,
}

DEFAULT_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_VAE = "madebyollin/sdxl-vae-fp16-fix"


class ControlNetPipeline:
    """Stable Diffusion XL pipeline with ControlNet conditioning.

    Supports text-to-image and image-to-image generation with spatial
    control via Canny edge maps, depth maps, or OpenPose skeletons.

    Args:
        control_type: Type of ControlNet conditioning ('canny', 'depth', 'pose').
        base_model: HuggingFace model ID for the SDXL base model.
        controlnet_model: HuggingFace model ID for the ControlNet model.
            If None, uses the default model for the given control_type.
        device: Target device ('cuda', 'mps', 'cpu', or None for auto-detect).
        dtype: Torch dtype for inference (default: float16 on GPU, float32 on CPU).
        enable_xformers: Whether to enable xformers memory-efficient attention.
        enable_cpu_offload: Whether to enable sequential CPU offloading for
            reduced VRAM usage at the cost of inference speed.
        scheduler: Scheduler type ('euler_a', 'dpm++', 'unipc').
        vae_model: Optional VAE model ID. Uses fp16-fix VAE by default.
    """

    def __init__(
        self,
        control_type: str = "canny",
        base_model: str = DEFAULT_BASE_MODEL,
        controlnet_model: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        enable_xformers: bool = True,
        enable_cpu_offload: bool = False,
        scheduler: str = "euler_a",
        vae_model: Optional[str] = None,
    ):
        self.control_type = control_type
        self.base_model_id = base_model
        self.enable_cpu_offload = enable_cpu_offload

        if controlnet_model is None:
            if control_type not in CONTROLNET_MODEL_MAP:
                raise ValueError(
                    f"Unknown control type '{control_type}'. "
                    f"Supported types: {list(CONTROLNET_MODEL_MAP.keys())}"
                )
            controlnet_model = CONTROLNET_MODEL_MAP[control_type]

        self.device = self._resolve_device(device)
        self.dtype = dtype or (torch.float16 if self.device.type != "cpu" else torch.float32)

        logger.info("Loading ControlNet model: %s", controlnet_model)
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )

        vae_id = vae_model or DEFAULT_VAE
        logger.info("Loading VAE: %s", vae_id)
        vae = AutoencoderKL.from_pretrained(
            vae_id,
            torch_dtype=self.dtype,
        )

        logger.info("Loading SDXL base model: %s", base_model)
        self.txt2img_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model,
            controlnet=self.controlnet,
            vae=vae,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if self.dtype == torch.float16 else None,
        )

        if scheduler in SCHEDULER_MAP:
            self.txt2img_pipe.scheduler = SCHEDULER_MAP[scheduler].from_config(
                self.txt2img_pipe.scheduler.config
            )

        self.img2img_pipe = None
        self._setup_device_and_optimizations(enable_xformers)

        logger.info(
            "Pipeline ready | control=%s | device=%s | dtype=%s",
            control_type, self.device, self.dtype,
        )

    def _resolve_device(self, device: Optional[str]) -> torch.device:
        """Determine the best available device."""
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _setup_device_and_optimizations(self, enable_xformers: bool) -> None:
        """Move pipeline to device and apply memory optimizations."""
        if self.enable_cpu_offload:
            self.txt2img_pipe.enable_sequential_cpu_offload()
            logger.info("Enabled sequential CPU offloading")
        else:
            self.txt2img_pipe = self.txt2img_pipe.to(self.device)

        if enable_xformers and self.device.type == "cuda":
            try:
                self.txt2img_pipe.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory-efficient attention")
            except ImportError:
                logger.warning(
                    "xformers not available, falling back to default attention"
                )

        self.txt2img_pipe.enable_attention_slicing()

    def _ensure_img2img_pipeline(self) -> None:
        """Lazily initialize the img2img pipeline when needed."""
        if self.img2img_pipe is not None:
            return

        logger.info("Initializing img2img pipeline")
        self.img2img_pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            self.base_model_id,
            controlnet=self.controlnet,
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            torch_dtype=self.dtype,
        )

        if self.enable_cpu_offload:
            self.img2img_pipe.enable_sequential_cpu_offload()
        else:
            self.img2img_pipe = self.img2img_pipe.to(self.device)

    def _prepare_generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        """Create a torch Generator with the given seed for reproducibility."""
        if seed is None:
            return None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return generator

    @staticmethod
    def _validate_dimensions(width: int, height: int) -> tuple[int, int]:
        """Ensure dimensions are multiples of 8 as required by the VAE."""
        width = (width // 8) * 8
        height = (height // 8) * 8
        if width < 64 or height < 64:
            raise ValueError("Width and height must be at least 64 pixels")
        return width, height

    @staticmethod
    def _prepare_control_image(
        control_image: Union[Image.Image, str, np.ndarray],
        width: int,
        height: int,
    ) -> Image.Image:
        """Load and resize a control image to match the target dimensions."""
        if isinstance(control_image, str):
            control_image = load_image(control_image)
        elif isinstance(control_image, np.ndarray):
            control_image = Image.fromarray(control_image)

        if control_image.mode != "RGB":
            control_image = control_image.convert("RGB")

        if control_image.size != (width, height):
            control_image = control_image.resize(
                (width, height), Image.Resampling.LANCZOS
            )

        return control_image

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        control_image: Union[Image.Image, str, np.ndarray],
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.8,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        num_images: int = 1,
        clip_skip: Optional[int] = None,
        callback: Optional[callable] = None,
        callback_steps: int = 1,
    ) -> Union[Image.Image, List[Image.Image]]:
        """Generate images with ControlNet conditioning.

        Args:
            prompt: Text prompt describing the desired image.
            control_image: Control image (Canny edges, depth map, or pose).
                Can be a PIL Image, file path, URL, or numpy array.
            negative_prompt: Text prompt for undesired features.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            controlnet_conditioning_scale: ControlNet influence strength (0.0-2.0).
            width: Output image width (will be rounded to multiple of 8).
            height: Output image height (will be rounded to multiple of 8).
            seed: Random seed for reproducibility.
            num_images: Number of images to generate.
            clip_skip: Number of CLIP layers to skip (None = use all).
            callback: Optional progress callback function.
            callback_steps: Frequency of callback invocations.

        Returns:
            A single PIL Image if num_images=1, otherwise a list of PIL Images.
        """
        width, height = self._validate_dimensions(width, height)
        control_image = self._prepare_control_image(control_image, width, height)
        generator = self._prepare_generator(seed)

        if negative_prompt is None:
            negative_prompt = (
                "low quality, blurry, distorted, deformed, disfigured, "
                "bad anatomy, watermark, text, signature, cropped"
            )

        autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
        autocast_enabled = self.device.type == "cuda"

        with torch.autocast(autocast_device, enabled=autocast_enabled):
            pipe_kwargs = dict(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                width=width,
                height=height,
                generator=generator,
                num_images_per_prompt=num_images,
                callback_on_step_end=callback,
            )

            if clip_skip is not None:
                pipe_kwargs["clip_skip"] = clip_skip

            output = self.txt2img_pipe(**pipe_kwargs)

        images = output.images
        if num_images == 1:
            return images[0]
        return images

    @torch.no_grad()
    def generate_img2img(
        self,
        prompt: str,
        control_image: Union[Image.Image, str, np.ndarray],
        init_image: Union[Image.Image, str],
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.8,
        strength: float = 0.7,
        seed: Optional[int] = None,
        num_images: int = 1,
    ) -> Union[Image.Image, List[Image.Image]]:
        """Generate images with ControlNet conditioning and an initial image.

        Uses the img2img pipeline where an existing image is partially denoised
        and then re-generated with both prompt and control guidance.

        Args:
            prompt: Text prompt describing the desired image.
            control_image: Control conditioning image.
            init_image: Initial image to start denoising from.
            negative_prompt: Text prompt for undesired features.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            controlnet_conditioning_scale: ControlNet influence strength.
            strength: Denoising strength (0.0 = no change, 1.0 = full denoise).
            seed: Random seed for reproducibility.
            num_images: Number of images to generate.

        Returns:
            A single PIL Image if num_images=1, otherwise a list of PIL Images.
        """
        self._ensure_img2img_pipeline()

        if isinstance(init_image, str):
            init_image = load_image(init_image)
        if init_image.mode != "RGB":
            init_image = init_image.convert("RGB")

        width, height = init_image.size
        width, height = self._validate_dimensions(width, height)
        init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)
        control_image = self._prepare_control_image(control_image, width, height)
        generator = self._prepare_generator(seed)

        if negative_prompt is None:
            negative_prompt = (
                "low quality, blurry, distorted, deformed, disfigured, "
                "bad anatomy, watermark, text, signature, cropped"
            )

        autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
        autocast_enabled = self.device.type == "cuda"

        with torch.autocast(autocast_device, enabled=autocast_enabled):
            output = self.img2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                control_image=control_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                strength=strength,
                generator=generator,
                num_images_per_prompt=num_images,
            )

        images = output.images
        if num_images == 1:
            return images[0]
        return images

    def change_control_type(self, control_type: str, controlnet_model: Optional[str] = None) -> None:
        """Switch the ControlNet model to a different control type.

        Args:
            control_type: New control type ('canny', 'depth', 'pose').
            controlnet_model: Optional custom ControlNet model ID.
        """
        if controlnet_model is None:
            if control_type not in CONTROLNET_MODEL_MAP:
                raise ValueError(f"Unknown control type: {control_type}")
            controlnet_model = CONTROLNET_MODEL_MAP[control_type]

        logger.info("Switching ControlNet to: %s (%s)", control_type, controlnet_model)

        new_controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )

        self.controlnet = new_controlnet
        self.control_type = control_type
        self.txt2img_pipe.controlnet = new_controlnet

        if self.img2img_pipe is not None:
            self.img2img_pipe.controlnet = new_controlnet

        if not self.enable_cpu_offload:
            self.controlnet = self.controlnet.to(self.device)

    def get_memory_usage(self) -> dict:
        """Return current GPU memory usage statistics."""
        if self.device.type != "cuda":
            return {"device": str(self.device), "note": "Memory tracking only available for CUDA"}

        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

        return {
            "device": str(self.device),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "max_allocated_gb": round(max_allocated, 2),
        }

    def clear_cache(self) -> None:
        """Free unused GPU memory."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU cache cleared")

    def __repr__(self) -> str:
        return (
            f"ControlNetPipeline("
            f"control_type='{self.control_type}', "
            f"device={self.device}, "
            f"dtype={self.dtype})"
        )
