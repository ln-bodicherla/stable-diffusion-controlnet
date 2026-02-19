#!/bin/bash
#
# Batch generation script for ControlNet pipeline.
# Processes all images in a directory with the same prompt and control type.
#
# Usage:
#   ./scripts/run_batch.sh --input-dir ./images --control-type canny --prompt "watercolor painting"
#   ./scripts/run_batch.sh --input-dir ./photos --control-type depth --prompt "oil painting" --steps 40

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR=""
CONTROL_TYPE="canny"
PROMPT=""
NEGATIVE_PROMPT=""
OUTPUT_DIR="${PROJECT_DIR}/outputs/batch"
STEPS=30
GUIDANCE_SCALE=7.5
CONTROLNET_SCALE=0.8
WIDTH=1024
HEIGHT=1024
SEED=""
CONFIG="${PROJECT_DIR}/configs/default_config.yaml"
EXTENSIONS="jpg jpeg png bmp webp tiff"
MAX_PARALLEL=1
DRY_RUN=false

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Batch generate images using ControlNet conditioning.

Required:
    --input-dir DIR        Directory containing input images
    --prompt TEXT           Text prompt for generation

Options:
    --control-type TYPE    Control type: canny, depth, pose (default: canny)
    --output-dir DIR       Output directory (default: ./outputs/batch)
    --negative-prompt TEXT Negative prompt
    --steps N              Inference steps (default: 30)
    --guidance-scale F     Guidance scale (default: 7.5)
    --controlnet-scale F   ControlNet scale (default: 0.8)
    --width N              Output width (default: 1024)
    --height N             Output height (default: 1024)
    --seed N               Random seed (applied to all images)
    --config PATH          Config file path
    --extensions LIST      Space-separated file extensions (default: jpg jpeg png bmp webp tiff)
    --dry-run              Print commands without executing
    -h, --help             Show this help message
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --control-type)
            CONTROL_TYPE="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --negative-prompt)
            NEGATIVE_PROMPT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --guidance-scale)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --controlnet-scale)
            CONTROLNET_SCALE="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --extensions)
            EXTENSIONS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            ;;
    esac
done

if [[ -z "$INPUT_DIR" ]]; then
    echo "Error: --input-dir is required"
    usage
fi

if [[ -z "$PROMPT" ]]; then
    echo "Error: --prompt is required"
    usage
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

IMAGE_FILES=()
for ext in $EXTENSIONS; do
    while IFS= read -r -d '' file; do
        IMAGE_FILES+=("$file")
    done < <(find "$INPUT_DIR" -maxdepth 1 -iname "*.${ext}" -print0 2>/dev/null)
done

IFS=$'\n' IMAGE_FILES=($(sort <<< "${IMAGE_FILES[*]}")); unset IFS

if [[ ${#IMAGE_FILES[@]} -eq 0 ]]; then
    echo "Error: No image files found in $INPUT_DIR"
    echo "Searched for extensions: $EXTENSIONS"
    exit 1
fi

echo "============================================================"
echo "  ControlNet Batch Generation"
echo "============================================================"
echo "  Input directory:  $INPUT_DIR"
echo "  Images found:     ${#IMAGE_FILES[@]}"
echo "  Control type:     $CONTROL_TYPE"
echo "  Prompt:           $PROMPT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Steps:            $STEPS"
echo "  Guidance scale:   $GUIDANCE_SCALE"
echo "  ControlNet scale: $CONTROLNET_SCALE"
echo "  Resolution:       ${WIDTH}x${HEIGHT}"
echo "  Seed:             ${SEED:-random}"
echo "============================================================"
echo ""

SUCCESSFUL=0
FAILED=0
START_TIME=$(date +%s)

for i in "${!IMAGE_FILES[@]}"; do
    image="${IMAGE_FILES[$i]}"
    basename=$(basename "$image")
    index=$((i + 1))
    total=${#IMAGE_FILES[@]}

    echo "[${index}/${total}] Processing: $basename"

    CMD="python -m src.generate"
    CMD+=" --prompt \"$PROMPT\""
    CMD+=" --control-type $CONTROL_TYPE"
    CMD+=" --input-image \"$image\""
    CMD+=" --output-dir \"$OUTPUT_DIR\""
    CMD+=" --steps $STEPS"
    CMD+=" --guidance-scale $GUIDANCE_SCALE"
    CMD+=" --controlnet-scale $CONTROLNET_SCALE"
    CMD+=" --width $WIDTH"
    CMD+=" --height $HEIGHT"
    CMD+=" --config \"$CONFIG\""
    CMD+=" --save-control-image"

    if [[ -n "$NEGATIVE_PROMPT" ]]; then
        CMD+=" --negative-prompt \"$NEGATIVE_PROMPT\""
    fi

    if [[ -n "$SEED" ]]; then
        CMD+=" --seed $SEED"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] $CMD"
        echo ""
        continue
    fi

    IMAGE_START=$(date +%s)

    if eval "cd \"$PROJECT_DIR\" && $CMD"; then
        IMAGE_END=$(date +%s)
        ELAPSED=$((IMAGE_END - IMAGE_START))
        echo "  Completed in ${ELAPSED}s"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo "  FAILED: $basename"
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "============================================================"
echo "  Batch Complete"
echo "============================================================"
echo "  Total images:  ${#IMAGE_FILES[@]}"
echo "  Successful:    $SUCCESSFUL"
echo "  Failed:        $FAILED"
echo "  Total time:    ${TOTAL_TIME}s"
if [[ $SUCCESSFUL -gt 0 ]]; then
    AVG=$((TOTAL_TIME / SUCCESSFUL))
    echo "  Avg per image: ${AVG}s"
fi
echo "  Output dir:    $OUTPUT_DIR"
echo "============================================================"

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
