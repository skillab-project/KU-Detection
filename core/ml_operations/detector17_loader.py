"""
Loader for the KU-Detector-17 UnixCoder checkpoint.

v2 — supports both:
  (a) local checkpoint directory (set DETECTOR17_MODEL_PATH env var)
  (b) Hugging Face model repo (set DETECTOR17_HF_REPO env var, e.g.
      "ElisavetKanidou/ku-detector-17")

Order of resolution:
  1. If DETECTOR17_HF_REPO is set → download from HF (cached by huggingface_hub)
  2. Else if DETECTOR17_MODEL_PATH points to a dir with best_model.pt → use it
  3. Else → raise FileNotFoundError

The HF cache lives in ~/.cache/huggingface/hub/ (or $HF_HOME if set), so after
the first boot subsequent restarts are instant.
"""

import os
import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from config.detector17_kus import DETECTOR17_KU_IDS, NUM_DETECTOR17_KUS

BASE_MODEL_NAME = "microsoft/unixcoder-base"
MAX_LENGTH      = 512

log = logging.getLogger(__name__)


class Detector17Classifier(nn.Module):
    """UnixCoder [CLS] → Linear(768,256) → GELU → Linear(256,17)."""
    def __init__(self, base_model_name: str = BASE_MODEL_NAME,
                 num_labels: int = NUM_DETECTOR17_KUS):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


class Detector17Model:
    """Runtime wrapper. Call .predict(code_text) for a single file."""
    def __init__(self, model, tokenizer, device, threshold: float = 0.5):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device
        self.threshold = threshold
        self.name      = "Detector17"

    def __str__(self):
        return self.name

    @torch.no_grad()
    def predict(self, code: str) -> dict:
        enc = self.tokenizer(
            code[:4096],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(enc["input_ids"], enc["attention_mask"])
        probs  = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        return {DETECTOR17_KU_IDS[i]: float(probs[i])
                for i in range(NUM_DETECTOR17_KUS)}

    def predict_binary(self, code: str) -> dict:
        probs = self.predict(code)
        return {k: 1 if v >= self.threshold else 0 for k, v in probs.items()}


# ─────────────────────────────────────────────────────────────────────────
# Resolver: local dir OR HF download
# ─────────────────────────────────────────────────────────────────────────

def _resolve_checkpoint_dir() -> str:
    """
    Returns a local directory containing best_model.pt + tokenizer files.
    Downloads from HuggingFace if DETECTOR17_HF_REPO is set.
    """
    hf_repo = os.getenv("DETECTOR17_HF_REPO")
    local_path = os.getenv("DETECTOR17_MODEL_PATH")

    # Priority 1: HuggingFace Hub
    if hf_repo:
        from huggingface_hub import snapshot_download
        hf_token = os.getenv("HF_TOKEN")  # optional, for private repos
        hf_revision = os.getenv("DETECTOR17_HF_REVISION", "main")

        log.info(f"[Detector17] Downloading from HF: {hf_repo}@{hf_revision}")
        cache_dir = os.getenv("HF_HOME", None)  # defaults to ~/.cache/huggingface
        local_dir = snapshot_download(
            repo_id=hf_repo,
            revision=hf_revision,
            cache_dir=cache_dir,
            token=hf_token,
            # Download only what we actually need — skip README/model cards.
            # vocab.json/merges.txt/special_tokens_map.json are optional
            # (tokenizer.json is self-contained for fast tokenizers).
            allow_patterns=[
                "best_model.pt",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ],
        )
        log.info(f"[Detector17] Downloaded to {local_dir}")
        return local_dir

    # Priority 2: Local path
    if local_path and os.path.isfile(os.path.join(local_path, "best_model.pt")):
        log.info(f"[Detector17] Using local model at {local_path}")
        return local_path

    raise FileNotFoundError(
        "Detector17 model not found.\n"
        "Set either:\n"
        "  DETECTOR17_HF_REPO=ElisavetKanidou/ku-detector-17   (recommended)\n"
        "or:\n"
        "  DETECTOR17_MODEL_PATH=/app/models/ku-detector-17    (local mount)\n"
        f"Current values: HF_REPO={os.getenv('DETECTOR17_HF_REPO')!r}, "
        f"MODEL_PATH={os.getenv('DETECTOR17_MODEL_PATH')!r}"
    )


def load_detector17_model(checkpoint_dir: str = None,
                           base_model_name: str = BASE_MODEL_NAME) -> Detector17Model:
    """
    Load Detector17 model. If checkpoint_dir is None, resolves from env vars.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_dir is None:
        checkpoint_dir = _resolve_checkpoint_dir()

    # Tokenizer: ALWAYS use the base UnixCoder tokenizer.
    # We only fine-tuned the classification head (768→256→17), not the tokenizer,
    # so the base tokenizer is identical to the fine-tuned one. Loading from the
    # local checkpoint can fail with version incompatibilities between the
    # transformers version that saved it (newer) and the one running here (4.41.2).
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    log.info(f"[Detector17] Using base tokenizer: {base_model_name}")

    model = Detector17Classifier(base_model_name).to(device)
    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"best_model.pt not found in {checkpoint_dir}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    log.info(f"[Detector17] Loaded weights from {ckpt_path} on {device}")
    return Detector17Model(model, tokenizer, device)
