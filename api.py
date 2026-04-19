"""
api.py
──────
Flask REST API that wraps the BioBERT DDI classifier.
Powers the React UI at http://localhost:3000.

Run with:
    python api.py

The model is loaded once at startup (~30 s on CPU).
All /predict calls are synchronous; typical latency ~200 ms.
"""

import os
import re
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from ddi_dataset import get_tokenizer, ID2LABEL
from model import DDIClassifier

app  = Flask(__name__)
CORS(app)  # allow the Vite dev server on :3000

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "biobert_ddi_best.pt")
MODEL_NAME = "dmis-lab/biobert-v1.1"
MAX_LENGTH = 128
DEVICE     = "cpu"   # always CPU for API stability

# ── Load once at startup ──────────────────────────────────────────────────────
print("[API] Loading tokenizer …")
_tokenizer = get_tokenizer(MODEL_NAME)

print("[API] Building model …")
_model = DDIClassifier(
    model_name    = MODEL_NAME,
    vocab_size    = len(_tokenizer),
    freeze_layers = 0,
)

if os.path.exists(MODEL_PATH):
    _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print(f"[API] Loaded weights → {MODEL_PATH}")
else:
    print(f"[API] WARNING: {MODEL_PATH} not found — predictions will be random")

_model.eval()
_model.to(DEVICE)
print("[API] Ready.  POST http://localhost:5001/predict")

# ── Label metadata (sent to the UI for rendering) ─────────────────────────────
LABEL_META = {
    "negative": {
        "color": "#6b7280",
        "bg":    "#1f2937",
        "icon":  "✓",
        "title": "No Interaction",
        "description": (
            "No significant drug-drug interaction detected between "
            "these compounds at standard therapeutic doses."
        ),
    },
    "mechanism": {
        "color": "#f59e0b",
        "bg":    "#2a2010",
        "icon":  "⚗",
        "title": "Pharmacokinetic",
        "description": (
            "One drug alters the absorption, distribution, metabolism, "
            "or excretion (ADME) of the other, changing its plasma levels."
        ),
    },
    "effect": {
        "color": "#ef4444",
        "bg":    "#2a1010",
        "icon":  "⚠",
        "title": "Pharmacodynamic",
        "description": (
            "The combined pharmacological effect of these drugs is altered — "
            "potentially causing toxicity, loss of efficacy, or unexpected side effects."
        ),
    },
    "advise": {
        "color": "#eab308",
        "bg":    "#252010",
        "icon":  "📋",
        "title": "Clinical Advisory",
        "description": (
            "A clinical warning or dosing recommendation exists for "
            "co-administration. Monitor the patient and consider dose adjustment."
        ),
    },
    "int": {
        "color": "#a855f7",
        "bg":    "#1e1530",
        "icon":  "↔",
        "title": "General Interaction",
        "description": (
            "An interaction between these drugs is documented, but the "
            "precise mechanism or effect type is not specified in this context."
        ),
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mark_entities(sentence: str, drug1: str, drug2: str) -> str:
    """
    Insert [E1] / [E2] entity markers around the first occurrence of each
    drug name (case-insensitive).  Falls back to prepending/appending if the
    name is not found in the sentence text.
    """
    marked = sentence

    p1 = re.compile(re.escape(drug1), re.IGNORECASE)
    if p1.search(marked):
        marked = p1.sub(lambda m: f"[E1] {m.group()} [/E1]", marked, count=1)
    else:
        marked = f"[E1] {drug1} [/E1] {marked}"

    p2 = re.compile(re.escape(drug2), re.IGNORECASE)
    if p2.search(marked):
        marked = p2.sub(lambda m: f"[E2] {m.group()} [/E2]", marked, count=1)
    else:
        marked = f"{marked} [E2] {drug2} [/E2]"

    return marked


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}

    sentence = body.get("sentence", "").strip()
    drug1    = body.get("drug1",    "").strip() or "Drug A"
    drug2    = body.get("drug2",    "").strip() or "Drug B"

    if not sentence:
        return jsonify({"error": "sentence is required"}), 400

    marked = _mark_entities(sentence, drug1, drug2)

    enc  = _tokenizer(
        marked,
        max_length     = MAX_LENGTH,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt",
    )
    ids    = enc["input_ids"].to(DEVICE)
    mask   = enc["attention_mask"].to(DEVICE)
    tokens = ids[0].tolist()

    e1_id  = _tokenizer.convert_tokens_to_ids("[E1]")
    e2_id  = _tokenizer.convert_tokens_to_ids("[E2]")
    e1_pos = torch.tensor([tokens.index(e1_id) if e1_id in tokens else 1])
    e2_pos = torch.tensor([tokens.index(e2_id) if e2_id in tokens else 2])

    with torch.no_grad():
        logits = _model(ids, mask, e1_pos, e2_pos)
        probs  = torch.softmax(logits, dim=-1)[0].tolist()

    pred_id    = int(torch.argmax(logits).item())
    pred_label = ID2LABEL[pred_id]

    return jsonify({
        "prediction":      pred_label,
        "confidence":      round(max(probs) * 100, 1),
        "probabilities":   {ID2LABEL[i]: round(p * 100, 1) for i, p in enumerate(probs)},
        "meta":            LABEL_META,
        "marked_sentence": marked,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "device": DEVICE})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
