"""
CLI entry point for generating images with ControlNet conditioning.

Usage:
    python -m src.generate \
        --prompt "a beautiful landscape" \
        --control-type canny \
        --input-image ./input.jpg \
        --output-dir ./outputs
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image

from src.pipeline import ControlNetPipeline
from src.preprocessors import PreprocessorFactory, resize_and_pad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file and return as dictionary."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file not found: %s, using defaults", config_path)
        return {}
    with open(path) as f:
        config = yaml.safe_load(f)
    return config or {}


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser with all generation options."""
    parser = argparse.ArgumentParser(
        description="Generate images with Stable Diffusion XL + ControlNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Canny edge control
  python -m src.generate --prompt "a house in the forest" --control-type canny --input-image house.jpg

  # Depth map control with custom parameters
  python -m src.generate --prompt "cyberpunk city" --control-type depth --input-image city.jpg \\
      --steps 40 --guidance-scale 8.0 --controlnet-scale 0.9 --seed 42

  # OpenPose control
  python -m src.generate --prompt "warrior in battle" --control-type pose --input-image person.jpg
        """,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the desired image",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Text prompt for undesired features (default: built-in negative prompt)",
    )
    parser.add_argument(
        "--control-type",
        type=str,
        required=True,
        choices=["canny", "depth", "pose"],
        help="Type of ControlNet conditioning to use",
    )
    parser.add_argument(
        "--input-image",
        type=str,
        required=True,
        help="Path to the input image for control extraction",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save generated images (default: ./outputs)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of inference steps (default: from config or 30)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale (default: from config or 7.5)",
    )
    parser.add_argument(
        "--controlnet-scale",
        type=float,
        default=None,
        help="ControlNet conditioning scale 0.0-2.0 (default: from config or 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output image width in pixels (default: from config or 1024)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output image height in pixels (default: from config or 1024)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate (default: 1)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML config file (default: configs/default_config.yaml)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="HuggingFace model ID for SDXL base model",
    )
    parser.add_argument(
        "--controlnet-model",
        type=str,
        default=None,
        help="HuggingFace model ID for ControlNet model (overrides control-type default)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Compute device (default: auto-detect)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="euler_a",
        choices=["euler_a", "dpm++", "unipc"],
        help="Diffusion scheduler type (default: euler_a)",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable sequential CPU offloading to reduce VRAM usage",
    )
    parser.add_argument(
        "--no-xformers",
        action="store_true",
        help="Disable xformers memory-efficient attention",
    )
    parser.add_argument(
        "--canny-low",
        type=int,
        default=None,
        help="Canny detector low threshold (default: 100)",
    )
    parser.add_argument(
        "--canny-high",
        type=int,
        default=None,
        help="Canny detector high threshold (default: 200)",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default=None,
        help="Depth estimation model type (default: DPT_Large)",
    )
    parser.add_argument(
        "--save-control-image",
        action="store_true",
        default=True,
        help="Save the preprocessed control image alongside the output (default: True)",
    )

    return parser


def resolve_parameters(args: argparse.Namespace, config: dict) -> dict:
    """Merge CLI arguments with config file defaults.

    CLI arguments take priority over config values.
    """
    gen_config = config.get("generation", {})
    model_config = config.get("model", {})
    preproc_config = config.get("preprocessing", {})

    params = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "control_type": args.control_type,
        "input_image": args.input_image,
        "output_dir": args.output_dir,
        "num_inference_steps": args.steps or gen_config.get("num_inference_steps", 30),
        "guidance_scale": args.guidance_scale or gen_config.get("guidance_scale", 7.5),
        "controlnet_conditioning_scale": (
            args.controlnet_scale or gen_config.get("controlnet_conditioning_scale", 0.8)
        ),
        "seed": args.seed,
        "width": args.width or gen_config.get("width", 1024),
        "height": args.height or gen_config.get("height", 1024),
        "num_images": args.num_images,
        "base_model": args.base_model or model_config.get(
            "base_model", "stabilityai/stable-diffusion-xl-base-1.0"
        ),
        "controlnet_model": args.controlnet_model,
        "device": args.device,
        "scheduler": args.scheduler,
        "cpu_offload": args.cpu_offload,
        "enable_xformers": not args.no_xformers,
        "save_control_image": args.save_control_image,
    }

    canny_config = preproc_config.get("canny", {})
    depth_config = preproc_config.get("depth", {})

    params["canny_low"] = args.canny_low or canny_config.get("low_threshold", 100)
    params["canny_high"] = args.canny_high or canny_config.get("high_threshold", 200)
    params["depth_model"] = args.depth_model or depth_config.get("model_type", "DPT_Large")

    return params


def create_preprocessor(params: dict):
    """Build the appropriate preprocessor based on control type and parameters."""
    control_type = params["control_type"]
    kwargs = {}

    if control_type == "canny":
        kwargs["low_threshold"] = params["canny_low"]
        kwargs["high_threshold"] = params["canny_high"]
    elif control_type == "depth":
        kwargs["model_type"] = params["depth_model"]
        kwargs["device"] = params["device"]
    elif control_type == "pose":
        kwargs["device"] = params["device"]

    return PreprocessorFactory.create(control_type, **kwargs)


def generate_output_filename(params: dict, index: int = 0) -> str:
    """Create a descriptive filename for a generated image."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    control_type = params["control_type"]
    seed_str = f"_s{params['seed']}" if params["seed"] is not None else ""
    idx_str = f"_{index}" if params["num_images"] > 1 else ""
    return f"{timestamp}_{control_type}{seed_str}{idx_str}.png"


def main():
    """Run the ControlNet generation pipeline from CLI arguments."""
    parser = build_argument_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    params = resolve_parameters(args, config)

    input_path = Path(params["input_image"])
    if not input_path.exists():
        logger.error("Input image not found: %s", input_path)
        sys.exit(1)

    output_dir = Path(params["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ControlNet Generation")
    logger.info("=" * 60)
    logger.info("Control type: %s", params["control_type"])
    logger.info("Input image:  %s", params["input_image"])
    logger.info("Prompt:       %s", params["prompt"])
    logger.info("Steps:        %d", params["num_inference_steps"])
    logger.info("Guidance:     %.1f", params["guidance_scale"])
    logger.info("CN Scale:     %.2f", params["controlnet_conditioning_scale"])
    logger.info("Resolution:   %dx%d", params["width"], params["height"])
    logger.info("Seed:         %s", params["seed"] or "random")
    logger.info("Output dir:   %s", params["output_dir"])
    logger.info("-" * 60)

    logger.info("Loading input image: %s", input_path)
    input_image = Image.open(input_path).convert("RGB")
    logger.info("Input size: %dx%d", input_image.width, input_image.height)

    logger.info("Running %s preprocessing...", params["control_type"])
    preprocessor = create_preprocessor(params)
    preprocess_start = time.time()
    control_image = preprocessor(
        input_image,
        target_width=params["width"],
        target_height=params["height"],
    )
    preprocess_time = time.time() - preprocess_start
    logger.info("Preprocessing completed in %.2fs", preprocess_time)

    if params["save_control_image"]:
        control_filename = generate_output_filename(params).replace(".png", "_control.png")
        control_path = output_dir / control_filename
        control_image.save(control_path)
        logger.info("Control image saved: %s", control_path)

    logger.info("Initializing pipeline...")
    pipeline_start = time.time()
    pipeline = ControlNetPipeline(
        control_type=params["control_type"],
        base_model=params["base_model"],
        controlnet_model=params["controlnet_model"],
        device=params["device"],
        enable_xformers=params["enable_xformers"],
        enable_cpu_offload=params["cpu_offload"],
        scheduler=params["scheduler"],
    )
    pipeline_load_time = time.time() - pipeline_start
    logger.info("Pipeline loaded in %.1fs", pipeline_load_time)

    logger.info("Generating %d image(s)...", params["num_images"])
    gen_start = time.time()
    result = pipeline.generate(
        prompt=params["prompt"],
        control_image=control_image,
        negative_prompt=params["negative_prompt"],
        num_inference_steps=params["num_inference_steps"],
        guidance_scale=params["guidance_scale"],
        controlnet_conditioning_scale=params["controlnet_conditioning_scale"],
        width=params["width"],
        height=params["height"],
        seed=params["seed"],
        num_images=params["num_images"],
    )
    gen_time = time.time() - gen_start
    logger.info("Generation completed in %.2fs", gen_time)

    if isinstance(result, list):
        for i, img in enumerate(result):
            filename = generate_output_filename(params, index=i)
            output_path = output_dir / filename
            img.save(output_path, quality=95)
            logger.info("Saved: %s", output_path)
    else:
        filename = generate_output_filename(params)
        output_path = output_dir / filename
        result.save(output_path, quality=95)
        logger.info("Saved: %s", output_path)

    logger.info("-" * 60)
    logger.info("Total time: %.1fs (preprocess: %.1fs, load: %.1fs, generate: %.1fs)",
                preprocess_time + pipeline_load_time + gen_time,
                preprocess_time, pipeline_load_time, gen_time)

    mem = pipeline.get_memory_usage()
    if "allocated_gb" in mem:
        logger.info("Peak GPU memory: %.2f GB", mem["max_allocated_gb"])

    pipeline.clear_cache()
    logger.info("Done.")


if __name__ == "__main__":
    main()
