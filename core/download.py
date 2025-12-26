"""Model download utilities with corruption recovery."""

import os
import shutil
from pathlib import Path
from typing import Optional

import folder_paths

from .log import log, log_start, log_end, log_warn

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
        log("Download", message)


def _validate_model_files(local_path: Path) -> bool:
    """
    Validate that all required model files are present.

    Checks for:
    - Single model file (model.safetensors or pytorch_model.bin)
    - OR all shards if sharded (model-00001-of-00002.safetensors, etc.)

    Returns:
        True if model files are complete, False otherwise
    """
    import json

    # Check for model index file (indicates sharded model)
    index_file = local_path / "model.safetensors.index.json"
    if index_file.exists():
        try:
            with open(index_file, "r") as f:
                index = json.load(f)

            # Get list of required shard files
            weight_map = index.get("weight_map", {})
            required_shards = set(weight_map.values())

            # Check if all shards exist
            for shard in required_shards:
                shard_path = local_path / shard
                if not shard_path.exists():
                    _log(f"Missing shard: {shard}")
                    return False

            _log(f"All {len(required_shards)} model shards present")
            return True

        except Exception as e:
            _log(f"Error reading index file: {e}")
            return False

    # Check for pytorch index file
    pytorch_index = local_path / "pytorch_model.bin.index.json"
    if pytorch_index.exists():
        try:
            with open(pytorch_index, "r") as f:
                index = json.load(f)

            weight_map = index.get("weight_map", {})
            required_shards = set(weight_map.values())

            for shard in required_shards:
                shard_path = local_path / shard
                if not shard_path.exists():
                    _log(f"Missing shard: {shard}")
                    return False

            return True

        except Exception as e:
            _log(f"Error reading pytorch index file: {e}")
            return False

    # Check for single model file
    single_model = local_path / "model.safetensors"
    if single_model.exists():
        return True

    single_pytorch = local_path / "pytorch_model.bin"
    if single_pytorch.exists():
        return True

    # Check for any .safetensors files (some models have different naming)
    safetensors_files = list(local_path.glob("*.safetensors"))
    if safetensors_files:
        _log(f"Found {len(safetensors_files)} safetensors files")
        return True

    _log("No model files found")
    return False


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
            # Also check for complete model files
            is_complete = _validate_model_files(local_path)
            if is_complete:
                _log(f"Found existing valid download at {local_path}")
                return local_path
            else:
                log_warn("Download", "Incomplete model files detected, re-downloading...")
                _log(f"Missing model shards, will re-download")
                force_download = True
        else:
            log_warn("Download", "Incomplete download detected, re-downloading...")
            _log(f"Missing config.json, will re-download")
            force_download = True

    download_start = log_start("Download", f"Downloading {model_id}")
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
            log_warn("Download", "Corrupted download detected, cleaning up and retrying...")
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

    log_end("Download", f"{model_name} downloaded", download_start)
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
        log("Download", f"Deleted {model_name}", force=True)
        return True

    return False
