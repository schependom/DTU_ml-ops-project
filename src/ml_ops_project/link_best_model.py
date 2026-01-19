"""Script to link the best model from a W&B sweep to the Model Registry.

After running a hyperparameter sweep, this script identifies the best-performing
run and promotes its model artifact to the W&B Model Registry for versioning
and deployment.

Usage:
    # With full sweep path:
    python -m ml_ops_project.link_best_model --sweep-id "entity/project/abc123"

    # With separate entity/project:
    python -m ml_ops_project.link_best_model --sweep-id abc123 --entity MLOpsss --project MLOps

    # Custom registry/collection:
    python -m ml_ops_project.link_best_model --sweep-id abc123 --registry MyRegistry --collection production_models
"""

import argparse

import wandb


def link_best_model(
    sweep_id: str,
    entity: str | None = None,
    project: str | None = None,
    target_registry: str = "Model_registry",
    target_collection: str = "sentiment_analysis_models",
) -> None:
    """Find the best run in a W&B sweep and link its model artifact to the registry.

    This function:
    1. Fetches the sweep metadata from W&B API
    2. Identifies the best run based on the sweep's optimization metric
    3. Retrieves the model artifact logged by PyTorch Lightning
    4. Links the artifact to the specified Model Registry collection

    Args:
        sweep_id: W&B sweep identifier. Can be full path ("entity/project/sweep_id")
            or just the sweep ID if entity and project are provided separately.
        entity: W&B entity (username or team). Optional if included in sweep_id.
        project: W&B project name. Optional if included in sweep_id.
        target_registry: Name of the Model Registry to link to.
        target_collection: Collection within the registry for organizing models.

    Returns:
        None. Prints status messages to stdout.

    Raises:
        wandb.errors.CommError: If the sweep cannot be found or API call fails.
    """
    api = wandb.Api()

    # Construct full sweep path: allows users to pass just the sweep ID
    # when entity/project are provided as separate arguments
    if entity and project and "/" not in sweep_id:
        sweep_path = f"{entity}/{project}/{sweep_id}"
    else:
        sweep_path = sweep_id

    print(f"Fetching sweep: {sweep_path}")
    sweep = api.sweep(sweep_path)

    # --- Find Best Run ---
    # W&B determines "best" based on the metric configured in the sweep YAML
    best_run = sweep.best_run()
    if not best_run:
        print("No runs found in sweep or no metric to optimize.")
        return

    # Log which run was selected and its performance
    metric_name = sweep.config.get("metric", {}).get("name")
    metric_value = best_run.summary.get(sweep.config["metric"]["name"])
    print(f"Best run: {best_run.name} (ID: {best_run.id})")
    print(f"Metric ({metric_name}): {metric_value}")

    # --- Find Model Artifact ---
    # PyTorch Lightning's WandbLogger logs checkpoints as artifacts with type="model"
    artifacts = best_run.logged_artifacts(type="model")

    if not artifacts:
        print("No model artifacts found in the best run.")
        return

    # Take the first artifact - typically the best checkpoint when using
    # ModelCheckpoint with save_top_k=1 and WandbLogger(log_model="all")
    model_artifact = artifacts[0]
    print(f"Found artifact: {model_artifact.name}")

    # --- Link to Registry ---
    # Registry path format: entity/registry_name/collection_name
    # Use the run's entity as fallback if not explicitly provided
    target_entity = entity or best_run.entity
    target_path = f"{target_entity}/{target_registry}/{target_collection}"

    print(f"Linking artifact to {target_path}...")
    model_artifact.link(target_path)
    print("Successfully linked!")

    # --- Download to models/ ---
    print("Downloading best model to models/ directory...")
    import os
    import shutil
    
    # Download artifact to a temporary directory
    download_dir = model_artifact.download()
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Find the checkpoint file and move/rename it
    found = False
    for filename in os.listdir(download_dir):
        if filename.endswith(".ckpt"):
            src_path = os.path.join(download_dir, filename)
            dest_path = os.path.join("models", "best_model.ckpt")
            shutil.copy(src_path, dest_path)
            print(f"Model saved to {dest_path}")
            found = True
            break
            
    if not found:
        print("Warning: No .ckpt file found in artifact!")



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the link_best_model script.

    Returns:
        Namespace containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Link the best model from a W&B sweep to the Model Registry.")
    parser.add_argument(
        "--sweep-id",
        type=str,
        required=True,
        help="WandB Sweep ID (entity/project/sweep_id or just sweep_id)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="WandB Entity (username or team)",
    )
    parser.add_argument(
        "--project",
        type=str,
        help="WandB Project name",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="Model_registry",
        help="Model Registry name (default: Model_registry)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="sentiment_analysis_models",
        help="Model Registry collection (default: sentiment_analysis_models)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    link_best_model(
        args.sweep_id,
        args.entity,
        args.project,
        args.registry,
        args.collection,
    )
