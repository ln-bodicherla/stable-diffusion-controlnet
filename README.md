# Stable Diffusion XL + ControlNet Pipeline

A production-ready pipeline for generating controlled images using Stable Diffusion XL with ControlNet conditioning. Supports depth maps, Canny edge detection, and OpenPose skeletal maps for precise spatial control over generated images.

## Architecture

```
                         +-------------------+
                         |   Input Image     |
                         +--------+----------+
                                  |
                    +-------------+-------------+
                    |             |              |
              +-----v----+ +-----v----+ +------v-----+
              |  Canny    | |  Depth   | |  OpenPose  |
              |  Detector | |  Est.    | |  Detector  |
              +-----+----+ +-----+----+ +------+-----+
                    |             |              |
                    +-------------+-------------+
                                  |
                         +--------v----------+
                         |  Control Image    |
                         +--------+----------+
                                  |
                         +--------v----------+
                         |   ControlNet      |
                         |   Encoder         |
                         +--------+----------+
                                  |
              +-------------------v-------------------+
              |        SDXL U-Net (with control)      |
              |                                       |
              |  Text Prompt ---> Cross Attention     |
              |  Noise       ---> Self Attention      |
              |  Control     ---> Residual Addition   |
              +-------------------+-------------------+
                                  |
                         +--------v----------+
                         |   VAE Decoder     |
                         +--------+----------+
                                  |
                         +--------v----------+
                         |  Generated Image  |
                         +-------------------+
```

## Features

- **Multiple Control Types**: Canny edge detection, monocular depth estimation, and OpenPose skeleton detection
- **SDXL Base**: Uses Stable Diffusion XL 1.0 for high-resolution 1024x1024 image generation
- **Flexible Conditioning**: Adjustable ControlNet conditioning scale for balancing creativity vs. control
- **Image-to-Image**: Supports img2img generation with ControlNet guidance
- **Batch Processing**: Shell script for processing entire directories of control images
- **Memory Efficient**: xformers attention, FP16 inference, and sequential CPU offloading support
- **Configurable**: YAML-based configuration for all model and generation parameters

## Setup

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (recommended) or MPS (Apple Silicon) or CPU
- 12GB+ VRAM recommended for SDXL inference

### Installation

```bash
git clone https://github.com/yourusername/stable-diffusion-controlnet.git
cd stable-diffusion-controlnet
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Model Downloads

Models are automatically downloaded from Hugging Face on first run. To pre-download:

```python
from src.pipeline import ControlNetPipeline
pipeline = ControlNetPipeline(control_type="canny")
```

## Usage

### Command Line

```bash
# Generate with Canny edge control
python -m src.generate \
    --prompt "a beautiful house in a forest, photorealistic, 4k" \
    --control-type canny \
    --input-image ./examples/house.jpg \
    --output-dir ./outputs \
    --steps 30 \
    --guidance-scale 7.5 \
    --controlnet-scale 0.8

# Generate with depth control
python -m src.generate \
    --prompt "a cyberpunk cityscape at night, neon lights" \
    --control-type depth \
    --input-image ./examples/city.jpg \
    --output-dir ./outputs \
    --seed 42

# Generate with OpenPose control
python -m src.generate \
    --prompt "a warrior standing in a battlefield, oil painting" \
    --control-type pose \
    --input-image ./examples/person.jpg \
    --output-dir ./outputs
```

### Python API

```python
from src.pipeline import ControlNetPipeline
from src.preprocessors import CannyPreprocessor
from PIL import Image

# Initialize
pipeline = ControlNetPipeline(control_type="canny")
preprocessor = CannyPreprocessor(low_threshold=100, high_threshold=200)

# Preprocess and generate
input_image = Image.open("input.jpg")
control_image = preprocessor(input_image)

result = pipeline.generate(
    prompt="a beautiful landscape, photorealistic",
    control_image=control_image,
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.8,
    seed=42
)

result.save("output.png")
```

### Batch Processing

```bash
chmod +x scripts/run_batch.sh
./scripts/run_batch.sh --input-dir ./images --control-type canny --prompt "watercolor painting"
```

## Project Structure

```
stable-diffusion-controlnet/
├── configs/
│   └── default_config.yaml      # Default model and generation settings
├── scripts/
│   └── run_batch.sh             # Batch processing script
├── src/
│   ├── __init__.py
│   ├── generate.py              # CLI entry point
│   ├── pipeline.py              # Core ControlNet pipeline
│   └── preprocessors.py         # Image preprocessing utilities
├── outputs/                     # Generated images (gitignored)
├── .gitignore
├── README.md
└── requirements.txt
```

## Tech Stack

- **Stable Diffusion XL 1.0** - Base generative model
- **ControlNet** - Spatial conditioning architecture
- **Diffusers** - Hugging Face diffusion model library
- **PyTorch** - Deep learning framework
- **controlnet-aux** - Preprocessing detectors (OpenPose, depth, etc.)
- **OpenCV** - Image processing and Canny edge detection
- **Gradio** - Optional web interface
- **xformers** - Memory-efficient attention

## Configuration

Edit `configs/default_config.yaml` to change default parameters. All config values can be overridden via CLI arguments.

## Performance Notes

| Device | Resolution | Steps | Time (approx) |
|--------|-----------|-------|---------------|
| A100 40GB | 1024x1024 | 30 | ~8s |
| RTX 3090 | 1024x1024 | 30 | ~15s |
| RTX 3080 | 1024x1024 | 30 | ~20s |
| M2 Max | 1024x1024 | 30 | ~45s |

## License

MIT License
