#!/bin/bash
# Run DictDyna commands inside the Sinergym Docker container.
#
# Usage:
#   ./docker_run.sh bash                    # Interactive shell
#   ./docker_run.sh collect --env Eplus-5zone-hot-continuous-v1
#   ./docker_run.sh baseline -m sac --env Eplus-5zone-hot-continuous-v1 --steps 8760
#   ./docker_run.sh train --env Eplus-5zone-hot-continuous-v1 --steps 26280

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="sailugr/sinergym:latest"

# GPU support (--gpus all if nvidia runtime available)
GPU_FLAG=""
if docker info 2>/dev/null | grep -q "nvidia"; then
    GPU_FLAG="--gpus all"
fi

# PYTHONPATH: project workspace + EnergyPlus API + Sinergym source
PYPATH="/workspace:/usr/local/EnergyPlus-25-1-0:/workspaces/sinergym"

# Install lightweight project deps
SETUP='pip install -q omegaconf pydantic loguru typer 2>/dev/null'

# Fix ownership of generated files so host user can manage them
FIXOWN="chown -R $(id -u):$(id -g) /workspace/output /workspace/data 2>/dev/null || true"

# Clean up EnergyPlus working directories after each run
CLEANUP="rm -rf /workspace/Eplus-*"

if [ $# -eq 0 ] || [ "$1" = "bash" ]; then
    echo "Starting interactive Sinergym container (GPU: ${GPU_FLAG:-none})..."
    docker run -it --rm ${GPU_FLAG} \
        -v "${SCRIPT_DIR}:/workspace" \
        -w /workspace \
        -e PYTHONPATH="${PYPATH}" \
        "${IMAGE}" \
        bash -c "${SETUP} && bash; ${FIXOWN}; ${CLEANUP}"
else
    echo "Running: cli.py $* (GPU: ${GPU_FLAG:-none})"
    docker run -i --rm ${GPU_FLAG} \
        -v "${SCRIPT_DIR}:/workspace" \
        -w /workspace \
        -e PYTHONPATH="${PYPATH}" \
        "${IMAGE}" \
        bash -c "${SETUP} && python3 cli.py $* ; ${FIXOWN}; ${CLEANUP}"
fi
