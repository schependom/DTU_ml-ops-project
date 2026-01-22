# End-to-End Sentiment Analysis Pipeline

_Group 7_

This repository contains the code and [MLOps](https://skaftenicki.github.io/dtu_mlops/) pipeline for a sentiment analysis project. The primary focus of this project is not the model architecture itself, but the full machine learning lifecycle, including experimentation, reproducibility, deployment, and monitoring.

## Project Description

### Overview

The goal of this project is to build a **reproducible, testable, and deployable MLOps pipeline** for an NLP task. Specifically, we will train a model that predicts whether a Rotten Tomatoes **critic review** is positive or negative based solely on the review text. The focus is not achieving state-of-the-art performance, but rather building a clean, automated end-to-end workflow: data ingestion → preprocessing → training → evaluation → packaging → deployment, with strong MLOps practices around it.

### Data

We will use the Kaggle dataset [Rotten Tomatoes Movies and Critic Reviews Dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset).

The dataset includes thousands of critic reviews covering many movies, along with metadata. From this raw data, we will construct our modeling dataset using the review text and an associated outcome field.

### Prediction target

The primary task is to, based on the review text, predict whether the review is positive or negative. This is a binary classification task:

- `0 = negative`
- `1 = positive`

### Models

We will do transfer learning using the **DistilBERT (`distilbert-base-uncased`)** model, fine-tuned using Hugging Face `transformers`.

DistilBERT is ~40% smaller and ~60% faster than BERT while retaining ~97% of its performance. This enables:

- Faster experimentation
- Practical hyperparameter sweeps (W&B)
- CI-compatible training smoke tests

### Training and Experimentation

Our training pipeline will include:

- **Config-driven runs** (Hydra)
- **Experiment tracking via W&B** (metrics, artifact storage, training curves)
- Checkpointing + reproducible seeds
- Cloud Computing via Google Cloud Platform (GCP):
    - Buckets for storing data and models (in combination with `dvc`)
    - Artifact Registry for storing container images
    - Vertex AI for training

## Installation

TODO: uv prerequisite
TODO: devcontainer
TODO: native

### `uv run` alias

I recommend creating an alias `uvr` for `uv run` to make running scripts easier.
Add the following line to your `~/.bashrc` (or equivalent on Windows/Linux):

```bash
echo "alias uvr='uv run'" >> ~/.bashrc
source ~/.bashrc
```

### Virtual environment

Initialize the virtual environment and install dependencies with:

```bash
uv sync
```

Activate the virtual environment with:

```bash
source .venv/bin/activate
```

Install an optional dependency group with:

```bash
uv sync --group <group-name>
```

Install all dependency groups with:

```bash
uv sync --all-groups
```

### Dependency management

Add a new dependency with (or add it straight to `pyproject.toml` and run `uv sync`):

```bash
uv add <package-name>
# e.g. uv add numpy
```

To add a development dependency, use:

```bash
uv add <package-name> --group dev
# e.g. uv add pytest --group dev
```

### Enabling pre-commit

```bash
uvr pre-commit install
uvr pre-commit run --all-files
```

### Version control

#### GitHub

Clone the repo:

```bash
git clone git@github.com:schependom/DTU_ml-operations-project.git
cd DTU_ml-operations-project
```

### Weights & Biases

To use Weights & Biases for experiment tracking, you need to set up your environment variables. Create a `.env` file in the project root with the following content:

```bash
WANDB_API_KEY=your_api_key_here
WANDB_ENTITY=your_entity_name
WANDB_PROJECT=your_project_name
WANDB_ORGANIZATION=your_organization_name
```

### GCP

#### Authentication

To get started with Google Cloud Platform (GCP), follow these steps.

Log in to your GCP account.

```bash
gcloud auth login
gcloud auth application-default login
```

#### Project setup

Set the project

```bash
gcloud config set project dtumlops-484016
```

#### DVC remote

I have created the following bucket:

```bash
gs://ml_ops_project_g7
```

You can add it to your local DVC config with:

```bash
dvc remote add -d gcp gs://ml_ops_project_g7
```

Then, follow the usage instructions below to pull the data.

#### Hosting the API on Google cloud.
The FastAPI app defining the API is in `src/ml_ops_project/api.py`. The corresponding dockerfile can be found in `dockerfiles/api.dockerfile`.

To host the API, you first need to push it to the artifact registry:
- open Docker Desktop
- Run the following command from the project root:
`docker build -f ./dockerfiles/api.dockerfile . -t api:latest`
- Tag the image (find project-id and repository-id by entering a repository in the artifact registry on Google Cloud)
`docker tag api europe-west1-docker.pkg.dev/<project-id>/<repository-id>/api:latest`
- Push the image:
`docker push europe-west1-docker.pkg.dev/<project-id>/<repository-id>/api:latest`
- Deploy a service on Cloud Run configured from the image or via the following command.
`gcloud run deploy <service-name> --image <image-name>:<image-tag> --platform managed --region europe-west1 --allow-unauthenticated`

Finally, verify that it's up and running:   
`gcloud run services list`
`gcloud run services describe <service-name> --region europe-west1`



## Usage

You can use `invoke` to run common tasks. To list available tasks, run:

```bash
invoke --list
# Available tasks:
#
#   build-docs        Build documentation.
#   docker-build      Build docker images.
#   preprocess-data   Preprocess data.
#   serve-docs        Serve documentation.
#   test              Run tests.
#   train             Train model.
```

Now, to run a task, use:

```bash
invoke <task-name>
# e.g. invoke preprocess-data
```

To train the model using the default configuration (`configs/config.yaml`), run either of the following commands:

```bash
uvr invoke train
# or
uvr python src/ml_ops_project/train.py
```

When you run the training script:

1.  **Configuration**: Hydra loads and composes configuration from `configs/`.
2.  **Environment**: The script loads environment variables from `.env`.
3.  **WandB**: Initializes tracking (if enabled) using credentials from `.env`.
4.  **Data**: Loads processed rotten tomatoes data.
5.  **Execution**: Runs the training loop, logging loss and accuracy.
6.  **Artifacts**: Saves the trained model to `models/model.pth` and training plots to `reports/figures/`.

### Custom Hyperparameters (Hydra)

You can override any configuration parameter from the command line:

```bash
# Change learning rate and batch size
uvr src/ml_ops_project/train.py optimizer.lr=0.01 batch_size=64

# Change number of epochs
uvr src/ml_ops_project/train.py epochs=20

# Disable WandB for a specific run
uvr src/ml_ops_project/train.py wandb.enabled=false
```

Or simply modify the configs in `configs/` and run `uvr invoke train`.

### Hyperparameter Sweeps (WandB)

1.  **Initialize the sweep**:

    ```bash
    uv run wandb sweep configs/wandb/sweep.yaml
    ```

    This prints a sweep ID (e.g., `entity/project/sweep_ID`).

2.  **Start the agent**:

    ```bash
    uv run wandb agent entity/project/sweep_ID
    ```

    The agent will run multiple training jobs with arguments defined in `parameters` section of `configs/wandb/sweep.yaml`.

3.  **Link the best model to the registry**:

    After the sweep is complete, you can link the best model to a WandB model registry using the provided script:

    ```bash
    uvr src/ml_ops_project/link_best_model.py --sweep-id entity/project/sweep_ID
    ```

### Model Registry Management

To link the best model from a specific hyperparameter sweep to the registry, run:

```bash
uvr src/ml_ops/link_best_model.py --sweep-id <sweep_id>
```

### DVC

Pull data from DVC remote:

```bash
uv run dvc pull
```

For data version control:

```bash
dvc add data
git add data.dvc # or `git add .`
git commit -m "Add new data"
git tag -a v2.0 -m "Version 2.0"
dvc push
git push origin main --tags
```

Or simply use (possible thanks to `tasks.py`):

```bash
uvr invoke dvc --folder 'data' --message 'Add new data'
```

To go back to a specific version later, you can checkout the git tag:

```bash
git switch v1.0 # or `git checkout v1.0`
dvc checkout
```

To go back to the latest version, use:

```bash
git switch main # or `git checkout main`
dvc checkout
```

## Containerization

Docker containers provide isolated, reproducible environments for training and evaluating models. This project includes optimized Dockerfiles for both operations.

### Building Docker Images

Build the training image:

```bash
docker build -f dockerfiles/train.dockerfile . -t train:latest
```

Build the evaluation image:

```bash
docker build -f dockerfiles/evaluate.dockerfile . -t evaluate:latest
```

<details>
<summary>Cross-platform builds (e.g., building for AMD64 on ARM Mac)</summary>

Some systems (like Apple M1/M2 Macs) use ARM architecture, which can lead to compatibility issues when sharing Docker images with others using AMD64 architecture (common in cloud and many desktops). To ensure compatibility, you can build images for a specific platform using the `--platform` flag.

```bash
# ARM Mac (Apple Silicon) -> AMD64
docker build --platform linux/amd64 -f dockerfiles/train.dockerfile . -t train:latest

# Windows on AMD64 -> ARM64 (e.g. Apple Silicon)
docker build --platform linux/arm64 -f dockerfiles/train.dockerfile . -t train:latest
```

</details>

### Running Docker Containers

Run training (using configurations in `configs/`):

```bash
docker run --rm --name train train:latest
```

Run training with **custom** parameters:

```bash
docker run --rm --name train train:latest <parameters>
```

Run training with a custom config file (must be included in the image or mounted as a volume):

```bash
# assumes custom_config.yaml is in `configs/`
docker run --rm --name train train:latest --config-name custom_config

# mounts custom_config.yaml from host
docker run --rm --name train -v $(pwd)/configs/custom_config.yaml:/configs/custom_config.yaml train:latest --config-name custom_config
```

Run evaluation (requires model file in image or mounted as volume):

```bash
docker run --rm --name eval evaluate:latest model_checkpoint=models/model.pth

# Mounted
docker run --rm --name eval -v $(pwd)/models/model.pth:/models/model.pth evaluate:latest model_checkpoint=/models/model.pth
```

### Mounting volumes

Use volumes to share data between host and container.

#### When to mount volumes?

If data changes frequently, or if you want to automatically sync outputs (models, reports) to your host machine, use mounted volumes:

- Models (`models/`)
- Configs (`configs/`)

#### When not to mount volumes?

If data is static and large, or if you want a fully self-contained container, **copy** data into the image during build, don't mount volumes:

- Data (`data/`)

#### Examples

Run evaluation with mounted volumes (keeps models and configs on host):

```bash
# Mount model and data directories
docker run --rm --name eval \
  -v $(pwd)/models:/models \
  -v $(pwd)/configs:/configs \
  evaluate:latest \
  model_checkpoint=/models/model.pth

# Or mount specific files
docker run --rm --name eval \
  -v $(pwd)/models/model.pth:/models/model.pth \
  -v $(pwd)/configs/config.yaml:/configs/config.yaml \
  evaluate:latest \
  model_checkpoint=/models/model.pth
```

### Interactive Mode

Debug or explore the container interactively:

```bash
docker run --rm -it --entrypoint sh train:latest
```

Exit the container with the `exit` command.

### Copying Files from Container

After training, copy outputs from container to host:

```bash
# Trained model
docker cp experiment1:/models/model.pth models/model.pth
# Training statistics figure
docker cp experiment1:/reports/figures/training_statistics.png reports/figures/training_statistics.png
```

If you mounted `models/` and `reports/` as volumes using respectively `-v $(pwd)/models:/models` and `-v $(pwd)/reports:/reports`, the files will already be on your own machine after training.

### Container and Image Management

#### Containers

List all **containers** (running and stopped):

```bash
docker ps -a
# or `docker container ls -a`
```

Remove a specific container:

```bash
docker rm train
```

Clean up stopped containers:

```bash
docker container prune
```

#### Images

List all **images**:

```bash
docker images
```

Remove a specific image (only if you want to rebuild or no longer need it):

```bash
docker rmi train:latest
```

Clean up dangling images (unnamed `<none>` images from rebuilds):

```bash
docker image prune
```

#### System-wide Cleanup

Clean up everything (stopped containers, dangling images, unused networks):

```bash
docker system prune
```

### Api container

Build the api image:

```bash
docker build -f dockerfiles/api.dockerfile -t sentiment_api:latest .
```

Run the api container:

```bash
docker run --env-file .env -p 8080:8080 --rm \
  -v <path-to-credentials>/dtumlops-cred.json:/gcp/creds.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/creds.json \
  sentiment_api:latest
```

Note that `.env` should look something like this:

```
WANDB_API_KEY=...
WANDB_PROJECT=MLOps-Project
WANDB_ENTITY=MLOpsss
WANDB_ORGANIZATION=turtle_team-org
```

### Monitoring container

Build the monitoring image:

```bash
docker build -f dockerfiles/monitoring.dockerfile -t sentiment_monitoring:latest .
```

Run the monitoring container:

```bash
docker run --env-file .env -p 8080:8080 --rm \
  -v <path-to-credentials>/dtumlops-cred.json:/gcp/creds.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/creds.json \
  sentiment_monitoring:latest
```

The `init` flag is important to handle signal forwarding (e.g. `CTRL+C`) correctly.
