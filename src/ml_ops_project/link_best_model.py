import argparse

import wandb


def link_best_model(
    sweep_id: str,
    entity: str = None,
    project: str = None,
    target_registry: str = "Model_registry",
    target_collection: str = "sentiment_analysis_models",
):
    """
    Finds the best run in a sweep and links its model to the registry.
    """
    api = wandb.Api()

    # Construct full sweep path if entity/project provided
    if entity and project and "/" not in sweep_id:
        sweep_path = f"{entity}/{project}/{sweep_id}"
    else:
        sweep_path = sweep_id

    print(f"Fetching sweep: {sweep_path}")
    sweep = api.sweep(sweep_path)

    # Get the best run
    best_run = sweep.best_run()
    if not best_run:
        print("No runs found in sweep or no metric to optimize.")
        return

    print(f"Best run: {best_run.name} (ID: {best_run.id})")
    print(
        f"Metric ({sweep.config.get('metric', {}).get('name')}): {best_run.summary.get(sweep.config['metric']['name'])}"
    )

    # Find model artifact
    # PyTorch Lightning WandbLogger logs models with type "model"
    artifacts = best_run.logged_artifacts(type="model")

    if not artifacts:
        print("No model artifacts found in the best run.")
        return

    # Assuming the "best" model is the one we want.
    # Usually PL logs "model-epoch=XX-val_loss=YY" or similar.
    # We take the latest or iterate. For simplicity, we take the last logged one which is often the best if PL checkpointing logic is correct.
    # Or ideally we check the aliases.

    model_artifact = artifacts[0]  # Pick the first/latest one

    print(f"Found artifact: {model_artifact.name}")

    # Link to registry
    # Target path: entity/registry/collection
    # If target_registry doesn't contain entity, use run's entity

    target_entity = entity or best_run.entity
    target_path = f"{target_entity}/{target_registry}/{target_collection}"

    print(f"Linking artifact to {target_path}...")
    model_artifact.link(target_path)
    print("Successfully linked!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-id", type=str, required=True, help="WandB Sweep ID (entity/project/sweep_id or just sweep_id)"
    )
    parser.add_argument("--entity", type=str, help="WandB Entity")
    parser.add_argument("--project", type=str, help="WandB Project")
    parser.add_argument("--registry", type=str, default="Model_registry", help="Model Registry name")
    parser.add_argument("--collection", type=str, default="sentiment_analysis_models", help="Model Registry collection")

    args = parser.parse_args()

    link_best_model(args.sweep_id, args.entity, args.project, args.registry, args.collection)
