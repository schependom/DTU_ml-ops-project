# MLOps Project: End-to-End Sentiment Analysis Pipeline

_Group 7_

This repository contains the code and MLOps pipeline for a sentiment analysis project. The primary focus of this project is not the model architecture itself, but the operationalization of the machine learning lifecycle, including experimentation, reproducibility, deployment, and monitoring.

## Project Description

### Overall Goal of the Project

The goal of this project is to build a **reproducible, testable, and deployable MLOps pipeline** for an NLP task. Specifically, we will train a model that predicts whether a Rotten Tomatoes **critic review** is positive or negative based solely on the review text. The focus is not achieving state-of-the-art performance, but rather building a clean, automated end-to-end workflow: data ingestion → preprocessing → training → evaluation → packaging → deployment, with strong MLOps practices around it.

---

### Data

We will use the Kaggle dataset **"Rotten Tomatoes Movies and Critic Reviews Dataset"**  
Source: https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset

The dataset includes thousands of critic reviews covering many movies, along with metadata. From this raw data, we will construct our modeling dataset using the review text and an associated outcome field.

**Prediction target**

-   **Primary task: Binary sentiment classification**  
    Map the dataset's fresh/rotten-style field (or equivalent) to:
    -   `0 = negative`
    -   `1 = positive`

If the score/rating fields are clean enough, we may also explore a secondary task (regression or ordinal prediction), but the binary classifier is the main deliverable.

**Data handling**

-   Clean and normalize review text (handle duplicates, missing text, odd encodings, very short reviews).
-   Deterministic **train/val/test** split with a fixed random seed.
-   Create a tiny "**smoke test subset**" (<1% of data) for CI/CD so the full pipeline can run fast on GitHub Actions.
-   **DVC** to version either the raw Kaggle archive, the processed dataset, or both.

---

### Models

#### Baseline (for grounding performance)

-   **TF-IDF + Logistic Regression**  
    Simple, fast, and ensures the deep model actually adds value.

#### Primary model (deep learning)

-   **DistilBERT (`distilbert-base-uncased`)**, fine-tuned using Hugging Face `transformers`.

DistilBERT is ~40% smaller and ~60% faster than BERT while retaining ~97% of its performance. This enables:

-   Faster experimentation
-   Practical hyperparameter sweeps (W&B)
-   CI-compatible training smoke tests

---

### Training and Experimentation

Our training pipeline will include:

-   **Config-driven runs** (Hydra or similar)
-   **Experiment tracking via W&B** (metrics, artifact storage, training curves)
-   Evaluation metrics: **accuracy, F1-score**, confusion matrix, and possibly ROC-AUC
-   Checkpointing + reproducible seeds

---

## Project structure

The directory structure of the project looks like this:

```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.dockerfile
│   └── train.dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```

Created using [DTU_ml-ops-template](https://github.com/schependom/DTU_ml-ops-template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) based on [mlops_template](https://github.com/SkafteNicki/mlops_template) by Nicki Skafte.

## Installation and notes on using `uv`

### Prerequisites

#### VSCode extensions

You need to download these extensions in order to make the settings in `.vscode/settings.json` to work properly:

-   [`ty`](https://marketplace.visualstudio.com/items?itemName=astral-sh.ty)
-   [`ruff`](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff-vscode)
-

#### `uv run` alias

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

### Running scripts

Running a script inside the virtual environment can be done with:

```bash
uv run <script-name>.py
# e.g. uv run src/ml_ops/train.py
```

This can get quite tedious, so we can make an alias `uvr` for this:

```bash
echo "alias uvr='uv run'" >> ~/.bashrc
source ~/.bashrc
```

Now you can run scripts like this:

```bash
uvr script.py
# e.g. uvr src/ml_ops/train.py
```

### Other `uv` commands

Change Python version with:

```bash
uv python pin <version>
```

### Using `uvx` for global tools

`uvx` can be used to install _tools_ (which are external command line tools, not libraries used in your code) globally on your machine. Tools include `black`, `ruff`, `pytest` or the simple `cowsay`. You can install such tools with `uvx`. For example:

```bash
uvx add cowsay
```

Then you can run the tool like this:

```bash
uvx cowsay -t "muuh"
```

If you run above command without having installed `cowsay` with `uvx`, it will install it for you automatically.

### Enabling pre-commit

```bash
uvr pre-commit install
uvr pre-commit run --all-files
```

## Usage

### Version control

Clone the repo:

```bash
git clone git@github.com:schependom/DTU_ML-Operations.git
cd DTU_ML-Operations
```

Authenticate DVC using SSH (make sure you have access to the remote):

```bash
dvc remote modify --local myremote auth ssh
```

Pull data from DVC remote:

```bash
dvc pull
```

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

After preprocessing data (v2), you can push (Data Version Control [DVC]) changes to the remote with:

```bash
dvc add data
git add data.dvc # or `git add .`
git commit -m "Add new data"
git tag -a v2.0 -m "Version 2.0"
# Why tag? To mark a specific point in git history as important (e.g., a release)
#   -a to create an annotated tag
#   -m to add a message to the tag
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

### Environment Setup (WandB)

To use Weights & Biases for experiment tracking, you need to set up your environment variables. Create a `.env` file in the project root with the following content:

```bash
WANDB_API_KEY=your_api_key_here
WANDB_ENTITY=your_entity_name
WANDB_PROJECT=your_project_name
WANDB_ORGANIZATION=your_organization_name
```

To train the model using the default configuration (`configs/config.yaml`), run either of the following commands:

```bash
uvr invoke train
uvr python src/ml_ops_project/train.py
uvr train # because we configured a script entry point in pyproject.toml
```

**Training Process Overview:**
When you run the training script:

1.  **Configuration**: Hydra loads and composes configuration from `configs/`.
2.  **Environment**: The script loads environment variables from `.env`.
3.  **WandB**: Initializes tracking (if enabled) using credentials from `.env`.
4.  **Data**: Loads processed MNIST data.
5.  **Execution**: Runs the training loop, logging loss and accuracy.
6.  **Artifacts**: Saves the trained model to `models/model.pth` and training plots to `reports/figures/`.

#### Custom Hyperparameters (Hydra)

You can override any configuration parameter from the command line:

```bash
# Change learning rate and batch size
uvr src/ml_ops/train.py optimizer.lr=0.01 batch_size=64

# Change number of epochs
uvr src/ml_ops/train.py epochs=20

# Switch optimizer config group (e.g. to nesterov.yaml)
uvr src/ml_ops/train.py optimizer=nesterov

# Disable WandB for a specific run
uvr src/ml_ops/train.py wandb.enabled=false
```

#### Hyperparameter Sweeps (WandB)

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

#### Model Registry Management

To manage your models in the WandB Model Registry, we provide two scripts:

1.  **Link Best Model from Sweep** (`link_best_model.py`):
    Links the best model from a specific hyperparameter sweep.

    ```bash
    uvr src/ml_ops/link_best_model.py --sweep-id <sweep_id>
    ```

2.  **Auto-Register Best Model from History** (`auto_register_best_model.py`):
    Scans all versions of a source artifact (e.g., `corrupt_mnist_model`) and links the one with the best metadata metric (e.g., highest `accuracy`) to the registry with "best" and "staging" aliases.

    ```bash
    uvr python src/ml_ops/promote_model.py \
        --project-name "ml_ops_corrupt_mnist" \
        --source-artifact "corrupt_mnist_model" \
        --target-registry "Model-registry" \
        --target-collection "corrupt-mnist" \
        --metric-name "accuracy"
    ```

#### Custom Configuration Files

You can also create a new config file `configs/custom_config.yaml` with:

```yaml
defaults:
    - my_new_model_conf
    - my_new_training_conf
    - optimizer: my_preferred_optimizer
    - _self_
wandb:
    enabled: true # or false to disable
use_my_new_model_conf: true
use_my_new_training_conf: true
```

Then run training with the new config:

```bash
uvr src/ml_ops/train.py --config-name=custom_config
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

-   Models (`models/`)
-   Configs (`configs/`)

#### When not to mount volumes?

If data is static and large, or if you want a fully self-contained container, **copy** data into the image during build, don't mount volumes:

-   Data (`data/`)
-

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

### Docker Best Practices

-   **Use `--rm`**: Automatically remove containers after they exit to avoid clutter
-   **Mount volumes**: For **models** (`models/`), **outputs** (`reports/`) and **configs** (`configs/`) instead of copying files
-   **Copy, and don't mount, static data**: For large, unchanging datasets (`data/`) to keep container self-contained
-   **Use `.dockerignore`**: Exclude unnecessary files from build context for faster builds
-   **Name your containers**: Makes them easier to reference with `--name`
-   **Tag images properly**: Use meaningful tags beyond `latest` for versioning

## GCP

### Authentication

To get started with Google Cloud Platform (GCP), follow these steps.

Log in to your GCP account.

```bash
gcloud auth login
gcloud auth application-default login
```

Set the project

```bash
gcloud config set project dtumlops-484016
```

I have created the following bucket:

```bash
gs://ml_ops_project_g7
```

### Build and push Docker image to Artifact Registry

To build and push Docker image to Artifact Registry using the cloudbuild.yaml file:

```bash
gcloud builds submit . --config=GCP/cloudbuild.yaml
```

TODO: add `WANDB_API_KEY_PROJ` to the GCP secrets

### Using Vertex AI

First, add your secrets (e.g. `WANDB_API_KEY_PROJ`) to Secret Manager in GCP.
Make sure the cloudbuild service account has access to your secrets.

```bash
gcloud secrets add-iam-policy-binding WANDB_API_KEY_PROJ \
  --project=dtumlops-484016 \
  --member="serviceAccount:1041875805298@cloudbuild.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding dtumlops-484016 \
    --member="serviceAccount:1041875805298-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

Now, you can use the `cloudbuild/vertex_ai_train.yaml` to run training on Vertex AI:

```bash
gcloud builds submit . --config=GCP/vertex_ai_train.yaml
```