"""Script to promote the best model from all runs in a W&B project.

This script scans all finished runs in the specified project, identifies the
best-performing run based on a given metric (e.g., val_accuracy), and promotes
its model artifact to the Model Registry. It also downloads the model locally.

Usage:
    python -m ml_ops_project.promote_best_model --project MLOps-Project
    python -m ml_ops_project.promote_best_model --project MLOps-Project --metric test_accuracy
"""

import argparse
import os
import shutil

import wandb


def promote_best_model(
    entity: str | None,
    project: str,
    metric: str = "val_accuracy",
    mode: str = "max",
    target_registry: str = "Model_registry",
    target_collection: str = "sentiment_analysis_models",
    alias: str = "inference",
    target_entity: str | None = None,
) -> None:
    """Find the best run in the project and promote its model artifact.

    Args:
        entity: W&B entity (username or team) for the SOURCE project.
        project: W&B project name.
        metric: Metric name to optimize (default: val_accuracy).
        mode: Optimization mode ('max' or 'min').
        target_registry: Name of the Model Registry.
        target_collection: Collection within the registry.
        alias: Alias to assign to the promoted model (default: inference).
        target_entity: Entity where the Model Registry lives (default: same as source entity).
    """
    api = wandb.Api()

    # Construct project path
    if entity:
        path = f"{entity}/{project}"
    else:
        path = project

    print(f"Scanning runs in project: {path}...")
    try:
        runs = api.runs(path)
    except Exception as e:
        print(f"Error fetching runs: {e}")
        return

    print(f"Found {len(runs)} total runs.")

    # Filter for successful runs that have the metric
    valid_runs = []
    for run in runs:
        if run.state == "finished" and metric in run.summary:
            valid_runs.append(run)

    if not valid_runs:
        print(f"No finished runs found with metric '{metric}'.")
        return

    print(f"Found {len(valid_runs)} valid runs with metric '{metric}'.")

    # Sort runs based on metric
    # Note: run.summary[metric] gives the value
    reverse = True if mode == "max" else False

    sorted_runs = sorted(
        valid_runs,
        key=lambda r: r.summary.get(metric, float("-inf") if mode == "max" else float("inf")),
        reverse=reverse,
    )

    best_run = sorted_runs[0]
    best_value = best_run.summary.get(metric)

    print(f"Best run: {best_run.name} (ID: {best_run.id})")
    print(f"Metric ({metric}): {best_value}")

    # --- Find Model Artifact ---
    artifacts = best_run.logged_artifacts(type="model")

    if not artifacts:
        print("No model artifacts found in the best run.")
        return

    # Take the first artifact (usually the best checkpoint for that run)
    model_artifact = artifacts[0]
    print(f"Found artifact: {model_artifact.name}")

    # --- Link to Registry ---
    # Registry path format: entity/registry_name/collection_name
    # Use target_entity if provided, else source entity (or run's entity)
    final_target_entity = target_entity or entity or best_run.entity
    target_path = f"{final_target_entity}/{target_registry}/{target_collection}"

    # Enforce exclusive alias
    print(f"Ensuring alias '{alias}' is exclusive in {target_path}...")
    try:
        collection = api.artifact_collection(target_path)
        # Iterate over versions in the collection
        for version in collection.artifacts():
            if alias in version.aliases:
                print(f"Removing alias '{alias}' from version {version.version}...")
                version.remove_alias(alias)
                version.save()  # Ensure clean state
    except Exception as e:
        print(f"Collection access/cleanup note (safe to ignore if new): {e}")

    print(f"Linking artifact to {target_path} with alias '{alias}'...")
    model_artifact.link(target_path, aliases=[alias])
    print("Successfully linked!")

    # --- Download to models/ ---
    print("Downloading best model to models/ directory...")

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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Promote the best model from a W&B project.")
    parser.add_argument("--entity", type=str, help="W&B Entity")
    parser.add_argument("--project", type=str, required=True, help="W&B Project name")
    parser.add_argument("--metric", type=str, default="val_accuracy", help="Metric to optimize")
    parser.add_argument("--mode", type=str, default="max", choices=["max", "min"], help="Optimization mode")
    parser.add_argument("--registry", type=str, default="Model_registry", help="Registry name")
    parser.add_argument("--collection", type=str, default="sentiment_analysis_models", help="Collection name")
    parser.add_argument(
        "--alias", type=str, default="inference", help="Alias for the promoted model (default: inference)"
    )
    parser.add_argument(
        "--target-entity", type=str, help="Entity where the Model Registry is located (if different from source)"
    )
    # Deployment arguments
    parser.add_argument("--deploy-service", type=str, help="Name of the Cloud Run service to redeploy")
    parser.add_argument("--deploy-region", type=str, help="Region of the Cloud Run service")
    parser.add_argument("--deploy-project", type=str, help="GCP Project ID of the Cloud Run service")

    return parser.parse_args()


def trigger_cloud_run_redeployment(service_name: str, region: str, project_id: str) -> None:
    """Trigger a redeployment of a Cloud Run service to pull the latest model.

    This function forces a new revision by updating a dummy environment variable
    or simply creating a new revision with the same image.

    Args:
        service_name: Name of the Cloud Run service.
        region: GCP Region (e.g., europe-west1).
        project_id: GCP Project ID.
    """
    try:
        from google.cloud import run_v2
    except ImportError:
        print("Error: google-cloud-run not installed. Skipping deployment.")
        return

    print(f"Triggering redeployment for Cloud Run service: {service_name}...")

    client = run_v2.ServicesClient()
    service_path = client.service_path(project_id, region, service_name)

    # Get current service configuration
    request = run_v2.GetServiceRequest(name=service_path)
    service = client.get_service(request=request)

    # Create a new revision by updating the update_time env var (or just re-submitting)
    # Just sending an update request with the existing config is often enough to trigger
    # a new revision if we change something small, like an annotation or env var.
    # Here we will add/update a "LAST_DEPLOYED" env var to force a change.

    # Find the container (usually the first one)
    if not service.template.containers:
        print("Error: No containers found in service template.")
        return

    container = service.template.containers[0]

    import datetime

    timestamp = datetime.datetime.now().isoformat()

    # Update or add the env var
    env_var_found = False
    for env in container.env:
        if env.name == "LAST_DEPLOYED":
            env.value = timestamp
            env_var_found = True
            break

    if not env_var_found:
        container.env.append(run_v2.EnvVar(name="LAST_DEPLOYED", value=timestamp))

    # Update the service
    operation = client.update_service(service=service)
    print("Waiting for operation to complete...")
    response = operation.result()
    print(f"Service updated successfully! New revision: {response.latest_ready_revision}")


if __name__ == "__main__":
    args = parse_args()

    promote_best_model(
        args.entity,
        args.project,
        args.metric,
        args.mode,
        args.registry,
        args.collection,
        args.alias,
        args.target_entity,
    )

    if args.deploy_service:
        if not args.deploy_region or not args.deploy_project:
            print("Error: --deploy-region and --deploy-project are required when --deploy-service is set.")
        else:
            trigger_cloud_run_redeployment(
                service_name=args.deploy_service, region=args.deploy_region, project_id=args.deploy_project
            )
