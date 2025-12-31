"""
SID_LLM_Local_Text - Local Text-Only Model Client

Text-only Qwen models for prompt modification tasks.
No vision capabilities - optimized for text generation.

Supported Models:
- Qwen2.5 (1.5B, 3B, 7B, 14B)
- Qwen3 (1.7B, 4B, 8B)

Features:
- VRAM-efficient text-only models
- Multiple quantization options (4-bit, 8-bit, FP16)
- Model caching for faster inference
- OpenAI-compatible interface
"""

import os
import gc
import sys
import threading
import time as time_module
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import folder_paths


# =============================================================================
# Utilities
# =============================================================================

def clear_memory():
    """Clear GPU and system memory."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def get_optimal_dtype(device: str):
    """Get optimal dtype for GPU (BF16 for Ampere+, FP16 for older)."""
    import torch
    if device != "cuda":
        return torch.float32
    try:
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    except Exception:
        return torch.float16


class TextInferenceSpinner:
    """Console progress spinner for inference."""
    SPINNER_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Generating"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None

    def _spinner_loop(self):
        idx = 0
        while not self._stop_event.is_set():
            elapsed = time_module.time() - self._start_time
            spinner = self.SPINNER_CHARS[idx % len(self.SPINNER_CHARS)]
            sys.stdout.write(f"\r[LocalTextModelClient] {spinner} {self.message}... {elapsed:.1f}s")
            sys.stdout.flush()
            idx += 1
            time_module.sleep(0.1)

    def start(self):
        self._start_time = time_module.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spinner_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        elapsed = time_module.time() - self._start_time if self._start_time else 0
        sys.stdout.write(f"\r[LocalTextModelClient] ✓ {self.message} complete ({elapsed:.1f}s)        \n")
        sys.stdout.flush()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# Model download directory
LLM_LOCAL_DIR = os.path.join(folder_paths.models_dir, "LLM", "text")
os.makedirs(LLM_LOCAL_DIR, exist_ok=True)


@dataclass
class TextModelInfo:
    """Information about a text-only model."""
    name: str
    repo_id: str
    vram_fp16: float
    vram_8bit: float
    vram_4bit: float
    max_output_tokens: int = 4096
    description: str = ""


# =============================================================================
# Model Registry - Text-Only Qwen Models
# =============================================================================

TEXT_MODELS: Dict[str, TextModelInfo] = {
    # Qwen2.5 Text Models
    "Qwen2.5-1.5B-Instruct": TextModelInfo(
        name="Qwen2.5 1.5B Instruct",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        vram_fp16=3.0, vram_8bit=2.0, vram_4bit=1.2,
        description="Smallest, fastest for simple tasks",
    ),
    "Qwen2.5-3B-Instruct": TextModelInfo(
        name="Qwen2.5 3B Instruct",
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        vram_fp16=6.0, vram_8bit=3.5, vram_4bit=2.0,
        description="Good balance of speed and quality",
    ),
    "Qwen2.5-7B-Instruct": TextModelInfo(
        name="Qwen2.5 7B Instruct",
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        vram_fp16=14.0, vram_8bit=8.0, vram_4bit=4.5,
        description="High quality, needs 8GB+ VRAM",
    ),
    "Qwen2.5-14B-Instruct": TextModelInfo(
        name="Qwen2.5 14B Instruct",
        repo_id="Qwen/Qwen2.5-14B-Instruct",
        vram_fp16=28.0, vram_8bit=15.0, vram_4bit=8.0,
        description="Best quality, needs 12GB+ VRAM",
    ),
    # Qwen3 Text Models
    "Qwen3-1.7B": TextModelInfo(
        name="Qwen3 1.7B",
        repo_id="Qwen/Qwen3-1.7B",
        vram_fp16=3.5, vram_8bit=2.2, vram_4bit=1.3,
        description="Fast, latest generation",
    ),
    "Qwen3-4B": TextModelInfo(
        name="Qwen3 4B",
        repo_id="Qwen/Qwen3-4B",
        vram_fp16=8.0, vram_8bit=4.5, vram_4bit=2.5,
        description="Balanced, latest generation",
    ),
    "Qwen3-8B": TextModelInfo(
        name="Qwen3 8B",
        repo_id="Qwen/Qwen3-8B",
        vram_fp16=16.0, vram_8bit=9.0, vram_4bit=5.0,
        description="High quality, latest generation",
    ),
    # Abliterated (uncensored)
    "Qwen2.5-7B-Abliterated": TextModelInfo(
        name="Qwen2.5 7B Abliterated (Uncensored)",
        repo_id="huihui-ai/Qwen2.5-7B-Instruct-abliterated",
        vram_fp16=14.0, vram_8bit=8.0, vram_4bit=4.5,
        description="Uncensored, needs 8GB+ VRAM",
    ),
    "Qwen2.5-14B-Abliterated": TextModelInfo(
        name="Qwen2.5 14B Abliterated (Uncensored)",
        repo_id="huihui-ai/Qwen2.5-14B-Instruct-abliterated",
        vram_fp16=28.0, vram_8bit=15.0, vram_4bit=8.0,
        description="Best uncensored, needs 12GB+ VRAM",
    ),
}


# =============================================================================
# Local Text Model Client
# =============================================================================

class LocalTextModelClient:
    """
    Client for local text-only models.
    Provides OpenAI-compatible interface.
    """

    # Class-level cache
    _cached_model = None
    _cached_tokenizer = None
    _cached_signature = None

    def __init__(
        self,
        model_name: str,
        quantization: str = "4-bit",
        device: str = "auto",
        keep_model_loaded: bool = True,
        repo_id: str = "",
        hf_token: Optional[str] = None,
        repetition_penalty: float = 1.3,
        top_p: float = 0.9,
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.device = device
        self.keep_model_loaded = keep_model_loaded
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p

        self.model = None
        self.tokenizer = None
        self.model_info = TEXT_MODELS.get(model_name)

    def _get_device_info(self) -> Dict[str, Any]:
        """Detect GPU and available memory."""
        import torch

        info = {"type": "cpu", "available_gb": 0, "total_gb": 0}

        if torch.cuda.is_available():
            info["type"] = "cuda"
            props = torch.cuda.get_device_properties(0)
            info["total_gb"] = props.total_memory / (1024**3)
            info["available_gb"] = (props.total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
            print(f"[LocalTextModelClient] CUDA: {props.name}, {info['available_gb']:.1f}GB free")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["type"] = "mps"
            info["total_gb"] = 16
            info["available_gb"] = 16
            print("[LocalTextModelClient] MPS: Apple Silicon")
        else:
            print("[LocalTextModelClient] WARNING: No GPU, using CPU (slow)")

        return info

    def _load_model(self):
        """Load model and tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import snapshot_download

        # Check cache
        signature = (self.model_name, self.quantization, self.device)
        if (LocalTextModelClient._cached_model is not None and
            LocalTextModelClient._cached_signature == signature):
            print(f"[LocalTextModelClient] Reusing cached model: {self.model_name}")
            self.model = LocalTextModelClient._cached_model
            self.tokenizer = LocalTextModelClient._cached_tokenizer
            return

        # Clear previous model
        if LocalTextModelClient._cached_model is not None:
            print("[LocalTextModelClient] Unloading previous model...")
            LocalTextModelClient._cached_model = None
            LocalTextModelClient._cached_tokenizer = None
            LocalTextModelClient._cached_signature = None
            clear_memory()

        if not self.model_info:
            raise ValueError(f"Unknown model: {self.model_name}")

        print(f"[LocalTextModelClient] Loading: {self.model_info.name}")

        # Download model
        model_path = os.path.join(LLM_LOCAL_DIR, self.model_name)

        if not os.path.exists(model_path):
            print(f"  Downloading to {model_path}...")
            repo = self.repo_id or self.model_info.repo_id
            download_kwargs = {
                "repo_id": repo,
                "local_dir": model_path,
                "ignore_patterns": ["*.md", ".git*"],
            }
            if self.hf_token:
                download_kwargs["token"] = self.hf_token
            snapshot_download(**download_kwargs)

        # Determine device
        device_info = self._get_device_info()
        device = device_info["type"] if self.device == "auto" else self.device

        # Auto-downgrade quantization if needed
        if device == "cuda":
            available = device_info["available_gb"]
            safety = 1.2
            if self.quantization == "None (FP16)" and self.model_info.vram_fp16 * safety > available:
                print(f"  Auto-downgrading to 8-bit (need {self.model_info.vram_fp16 * safety:.1f}GB)")
                self.quantization = "8-bit"
            if self.quantization == "8-bit" and self.model_info.vram_8bit * safety > available:
                print(f"  Auto-downgrading to 4-bit (need {self.model_info.vram_8bit * safety:.1f}GB)")
                self.quantization = "4-bit"

        print(f"  Device: {device}, Quantization: {self.quantization}")

        # Get quantization config
        quant_config = None
        dtype = get_optimal_dtype(device)

        if device == "cuda" and self.quantization != "None (FP16)":
            try:
                from transformers import BitsAndBytesConfig
                if self.quantization == "4-bit":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                elif self.quantization == "8-bit":
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                print("  bitsandbytes not available, using FP16")

        # Load model
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }

        if device == "cuda":
            load_kwargs["device_map"] = {"": 0}
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()
        self.model.config.use_cache = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Cache
        if self.keep_model_loaded:
            LocalTextModelClient._cached_model = self.model
            LocalTextModelClient._cached_tokenizer = self.tokenizer
            LocalTextModelClient._cached_signature = signature

        print(f"[LocalTextModelClient] Model loaded successfully")

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs
    ) -> "TextModelResponse":
        """Create a chat completion (OpenAI-compatible)."""
        import torch

        if self.model is None:
            self._load_model()

        # Build conversation
        conversation = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                conversation.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Extract text from content list
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                conversation.append({"role": role, "content": " ".join(text_parts)})

        # Apply chat template
        chat_text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(chat_text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
            "repetition_penalty": self.repetition_penalty,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = self.top_p

        with TextInferenceSpinner("Generating"):
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode
        input_len = inputs["input_ids"].shape[1]
        response_ids = outputs[0][input_len:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Truncate runaway generation
        response_text = self._truncate_runaway(response_text)

        # Cleanup
        self._cleanup()

        return TextModelResponse(response_text.strip())

    def _truncate_runaway(self, text: str) -> str:
        """Detect and truncate runaway repetition in LLM output."""
        import re

        if not text:
            return text

        # 0. Remove thinking blocks (<think>...</think>) - Qwen3 internal reasoning
        # First try to remove complete blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Then remove incomplete/open think blocks (model didn't close it)
        if '<think>' in text:
            think_pos = text.find('<think>')
            if think_pos > 0:
                text = text[:think_pos].rstrip()
                print(f"[LocalTextModelClient] Removed incomplete <think> block")
            else:
                # Think block at start - remove everything
                text = ""
                print(f"[LocalTextModelClient] Removed <think> block at start")

        if not text:
            return text

        # 1. Detect character repetition (e.g., "dddddddddd", "aaaaaaa")
        char_repeat_match = re.search(r'(.)\1{9,}', text)
        if char_repeat_match:
            truncate_pos = char_repeat_match.start()
            if truncate_pos > 50:
                text = text[:truncate_pos].rstrip()
                print(f"[LocalTextModelClient] Truncated character repetition at position {truncate_pos}")

        # 2. Detect word/phrase repetition
        words = text.split()
        if len(words) > 50:
            for ngram_size in [3, 4, 5]:
                ngrams = [' '.join(words[i:i+ngram_size]) for i in range(len(words) - ngram_size + 1)]
                if len(ngrams) > 10:
                    for i in range(len(ngrams) - 5):
                        if ngrams[i] == ngrams[i+1] == ngrams[i+2] == ngrams[i+3] == ngrams[i+4]:
                            word_pos = i
                            if word_pos > 20:
                                text = ' '.join(words[:word_pos]).rstrip()
                                print(f"[LocalTextModelClient] Truncated phrase repetition at word {word_pos}")
                                break

        # 3. Hard limit on word count (safety net)
        max_words = 400
        words = text.split()
        if len(words) > max_words:
            text = ' '.join(words[:max_words])
            print(f"[LocalTextModelClient] Truncated to {max_words} words (safety limit)")

        return text

    def _cleanup(self):
        """Clean up GPU state after generation."""
        import torch
        try:
            torch.set_grad_enabled(True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass


class TextModelResponse:
    """OpenAI-compatible response wrapper."""
    def __init__(self, content: str):
        self.choices = [TextModelChoice(content)]


class TextModelChoice:
    """OpenAI-compatible choice wrapper."""
    def __init__(self, content: str):
        self.message = TextModelMessage(content)


class TextModelMessage:
    """OpenAI-compatible message wrapper."""
    def __init__(self, content: str):
        self.content = content
