import os

import pytest
from dotenv import load_dotenv

import wandb

load_dotenv()


def test_wandb_service_account_access():
    """
    Verifies that the GitHub Actions Service Account:
    1. Has a valid API key.
    2. Can connect to the WandB API.
    3. Has explicit access to the target Project and Entity.
    """

    # Load the required secrets from environment variables (.env locally or GitHub Actions secrets).
    api_key = os.environ.get("WANDB_API_KEY")
    project = os.environ.get("WANDB_PROJECT")
    entity = os.environ.get("WANDB_ENTITY")

    # Fail fast with clear messages if any required secret is missing.
    if not api_key:
        pytest.fail("CRITICAL: WANDB_API_KEY is missing.")
    if not project:
        pytest.fail("CRITICAL: WANDB_PROJECT is missing.")
    if not entity:
        pytest.fail("CRITICAL: WANDB_ENTITY is missing.")
    assert project is not None and entity is not None

    # Initialize the API client using the explicit key (no local WandB config required).
    try:
        api = wandb.Api(api_key=api_key)
    except Exception as e:
        pytest.fail(f"Failed to initialize WandB API client: {e}")

    # Verify the service account can read the target project (permissions + existence).
    target_path = f"{entity}/{project}"

    try:
        # fetch the project details with a timeout to prevent hanging
        project_obj = api.project(entity, project)
        # Note: wandb.Api() doesn't support a global timeout easily,
        # but rapid failures are better than hangs.
        # Accessing properties triggers the network call.
        _ = project_obj.name

        # If we reach here, we have successful Read Access
        print(f"\nSUCCESS: Connected to project '{project_obj.name}' at '{project_obj.url}'")

    except wandb.errors.CommError:
        pytest.fail("Network Error: Could not reach WandB servers.")
    except wandb.errors.AuthenticationError:
        pytest.fail("Auth Error: The WANDB_API_KEY is invalid or expired.")
    except Exception as e:
        # This catches 404s (Project not found) or 403s (Permission denied)
        pytest.fail(
            f"Access Error: Could not access '{target_path}'. Check if the Entity/Project exists and the Service Account has permissions. Details: {e}"
        )
