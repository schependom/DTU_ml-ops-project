
###########
# GENERAL #
###########

# Starting from a base image (uv-based image)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install some essentials
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

####################
# PROJECT SPECIFIC #
####################

##
# Copy our project files into the container
#   ->  we only want the essential parts to keep
#       our Docker image as small as possible

# COPY requirements.txt requirements.txt
# not needed since we use uv and pyproject.toml
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
COPY data/ data/
COPY configs/ configs/
COPY tests/ tests/
COPY README.md README.md
COPY LICENSE LICENSE

##
# Install our project dependencies
# --locked      enforces strict adherence to uv.lock
# --no-cache    disables writing temporary download/wheel files to keep image size small

WORKDIR /

##
# Install dependencies using uv
##

## INEFFICIENT VERSION
## that always rebuilds everything from scratch
# RUN uv sync --locked --no-cache

## OPTIMIZED VERSION
# Below is an optimized version that uses caching of uv downloads/wheels between builds
# to speed up the build process while still keeping the final image small.
# This makes sure we don't always rebuild everything from scratch.
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

##
# Create directories for saving outputs
# Create 'models' directory as it's expected to verify checkpoints if passed via volumes
RUN mkdir -p models reports/figures

##
# The entry point is the application we want to run
# when the container starts up

ENTRYPOINT ["uv", "run", "src/ml_ops_project/evaluate.py"]

## Building the Docker image:
#
#     docker build -f dockerfiles/evaluate.dockerfile . -t evaluate:latest
#
# Running the image:
#
#     # You usually need to mount the directory containing your trained models
#     # so the container can access them.
#     docker run --rm -v $(pwd)/models:/models evaluate:latest ckpt_path=/models/best_model.ckpt
#
