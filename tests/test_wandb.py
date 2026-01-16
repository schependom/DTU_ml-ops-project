import os

import pytest
import wandb
from dotenv import load_dotenv

load_dotenv()


def test_wandb_service_account_access():
    """
    Verifies that the GitHub Actions Service Account:
    1. Has a valid API key.
    2. Can connect to the WandB API.
    3. Has explicit access to the target Project and Entity.
    """

    # 1. Load Environment Variables injected by GitHub Actions
    api_key = os.environ.get("WANDB_API_KEY")
    project = os.environ.get("WANDB_PROJECT")
    entity = os.environ.get("WANDB_ENTITY")

    # Fail immediately if secrets were not passed correctly
    if not api_key:
        pytest.fail("CRITICAL: WANDB_API_KEY is missing.")
    if not project:
        pytest.fail("CRITICAL: WANDB_PROJECT is missing.")
    if not entity:
        pytest.fail("CRITICAL: WANDB_ENTITY is missing.")

    # 2. Initialize the API Client (This does NOT start a run)
    # We pass the key explicitly to avoid relying on local config files
    try:
        api = wandb.Api(api_key=api_key)
    except Exception as e:
        pytest.fail(f"Failed to initialize WandB API client: {e}")

    # 3. Verify Project Access
    # We attempt to retrieve the specific project object.
    # If the user lacks permissions or the project doesn't exist, this raises an error.
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
