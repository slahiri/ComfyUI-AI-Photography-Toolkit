"""Model download utilities with corruption recovery."""

import os
import shutil
from pathlib import Path
from typing import Optional

import folder_paths

# Global verbose flag (set by caption node)
_verbose = False

# Global HF token (set by caption node)
_hf_token: Optional[str] = None


def set_verbose(verbose: bool) -> None:
    """Set global verbose logging flag."""
    global _verbose
    _verbose = verbose


def set_hf_token(token: Optional[str]) -> None:
    """Set HuggingFace token for gated model access."""
    global _hf_token
    _hf_token = token if token and token.strip() else None


def _get_hf_token() -> Optional[str]:
    """Get HF token from global setting or environment."""
    if _hf_token:
        return _hf_token
    # Fall back to environment variable
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _log(message: str) -> None:
    """Print message if verbose is enabled."""
    if _verbose:
        print(f"[SID-Download] {message}")


def download_model(
    model_id: str,
    config: dict,
    force_download: bool = False,
) -> Path:
    """
    Download model from HuggingFace Hub with corruption recovery.

    Args:
        model_id: HuggingFace model ID (e.g., "Salesforce/blip2-opt-2.7b")
        config: Model config dict with 'download' settings
        force_download: Force re-download even if exists

    Returns:
        Path to local model directory
    """
    download_config = config.get("download", {})
    subfolder = download_config.get("subfolder", "LLM")
    use_symlinks = download_config.get("use_symlinks", False)

    model_name = model_id.split("/")[-1]
    local_path = Path(folder_paths.models_dir) / subfolder / model_name

    _log(f"Model ID: {model_id}")
    _log(f"Local path: {local_path}")
    _log(f"Subfolder: {subfolder}, Symlinks: {use_symlinks}")

    # Check for existing valid download
    if local_path.exists() and not force_download:
        config_file = local_path / "config.json"
        if config_file.exists():
            _log(f"Found existing valid download at {local_path}")
            return local_path
        else:
            print(f"[SID-Toolkit] Incomplete download detected, re-downloading...")
            _log(f"Missing config.json, will re-download")
            force_download = True

    print(f"[SID-Toolkit] Downloading {model_id} to {local_path}...")
    _log(f"Starting HuggingFace snapshot_download...")
    _log(f"Force download: {force_download}")

    from huggingface_hub import snapshot_download

    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Get HF token for gated models
    hf_token = _get_hf_token()
    if hf_token:
        _log("Using HuggingFace token for authentication")
    else:
        _log("No HuggingFace token set (gated models may fail)")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_path,
            local_dir_use_symlinks=use_symlinks,
            force_download=force_download,
            token=hf_token,
        )
        _log(f"snapshot_download completed successfully")
    except OSError as e:
        if "Consistency check failed" in str(e):
            print(f"[SID-Toolkit] Corrupted download detected, cleaning up and retrying...")
            _log(f"Corruption error: {e}")
            _log(f"Deleting {local_path} and retrying...")
            if local_path.exists():
                shutil.rmtree(local_path)
            snapshot_download(
                repo_id=model_id,
                local_dir=local_path,
                local_dir_use_symlinks=use_symlinks,
                force_download=True,
                token=hf_token,
            )
            _log(f"Retry completed successfully")
        else:
            _log(f"Download error: {e}")
            raise
    except Exception as e:
        error_str = str(e)
        # Handle gated repository errors
        if "GatedRepoError" in type(e).__name__ or "403" in error_str or "gated repo" in error_str.lower():
            raise RuntimeError(
                f"[SID-Toolkit] Model '{model_id}' requires HuggingFace authentication.\n"
                f"This is a gated model. To use it:\n"
                f"1. Visit https://huggingface.co/{model_id} and request access\n"
                f"2. Add your HF token to the 'hf_token' input on the node\n"
                f"3. Or run 'huggingface-cli login' in terminal\n"
                f"4. Or manually download: huggingface-cli download {model_id} --local-dir {local_path}"
            ) from e
        _log(f"Download error: {e}")
        raise

    print(f"[SID-Toolkit] Download complete: {model_name}")
    _log(f"Download complete, files saved to {local_path}")
    return local_path


def delete_model(model_id: str, config: dict) -> bool:
    """
    Delete local model files.

    Args:
        model_id: HuggingFace model ID
        config: Model config dict

    Returns:
        True if deleted, False if not found
    """
    download_config = config.get("download", {})
    subfolder = download_config.get("subfolder", "LLM")

    model_name = model_id.split("/")[-1]
    local_path = Path(folder_paths.models_dir) / subfolder / model_name

    if local_path.exists():
        shutil.rmtree(local_path)
        print(f"[SID-Toolkit] Deleted {model_name}")
        return True

    return False
