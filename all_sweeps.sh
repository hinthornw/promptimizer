#!/bin/bash
set -e  # Exit on error

VENV_PATH=".venv/bin/activate"
DEFAULT_SWEEP="experiments/tool_sweeps.jsonl"
SWEEP_FILE=""
EPOCHS=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --sweep)
            SWEEP_FILE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--sweep path/to/sweep.jsonl] [--epochs N]"
            exit 1
            ;;
    esac
done

if [ -z "$SWEEP_FILE" ]; then
    SWEEP_FILE="$DEFAULT_SWEEP"
fi

# Fewshots:
# math_multi -> 4o
#####
# cs10 -> o1, o1-mini, claude, 4o
# tooluse -> o1, o1-mini, claude
# email_cs_simple -> o1, o1-mini, claude, 4o


directories=(
    "experiments/email_elon"
    "experiments/email_cs"
    "experiments/email_cs10"
    "experiments/math_multi"
    "experiments/email_cs_simple"
)

if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

source "$VENV_PATH" || {
    echo "Error: Failed to activate virtual environment"
    exit 1
}

# Verify sweep file exists
if [ ! -f "$SWEEP_FILE" ]; then
    echo "Error: Sweep file not found at $SWEEP_FILE"
    exit 1
fi

for dir in "${directories[@]}"; do
    if [ ! -f "$dir/config.json" ]; then
        echo "Warning: Config file not found at $dir/config.json"
        continue
    fi
    echo "Starting sweep for $dir using sweep file: $SWEEP_FILE"
    promptim train --task "$dir/config.json" --sweep "$SWEEP_FILE" --epochs $EPOCHS &
done

wait
echo "All sweeps completed"
