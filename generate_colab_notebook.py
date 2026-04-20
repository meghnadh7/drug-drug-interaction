"""
generate_colab_notebook.py
──────────────────────────
Run this once to produce DDI_FineTune_Colab.ipynb.
Then upload that notebook to Google Colab.

Usage:
    python generate_colab_notebook.py
"""

import json

# ─────────────────────────────────────────────────────────────────────────────
# Cell sources — written as plain Python strings, json.dump handles escaping
# ─────────────────────────────────────────────────────────────────────────────

CELL_GPU_CHECK = """\
import torch

assert torch.cuda.is_available(), (
    "No GPU detected!  Go to: Runtime > Change runtime type > T4 GPU, then reconnect."
)
print(f"GPU  : {torch.cuda.get_device_name(0)}")
print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("Ready.")
"""

CELL_INSTALL = """\
%%capture
!pip install transformers==4.40.0 scikit-learn tqdm
print("Packages ready.")
"""

CELL_UPLOAD_CORPUS = """\
import os, zipfile
from google.colab import files

print("Step 1: zip your DDICorpus folder on your Mac first:")
print("  Right-click DDICorpus on Desktop > Compress")
print("  This creates DDICorpus.zip (~8-9 MB)")
print()
print("Step 2: Click 'Choose Files' below and upload DDICorpus.zip")
print()

uploaded = files.upload()

fname = list(uploaded.keys())[0]
print(f"Uploaded: {fname}")

with zipfile.ZipFile(fname, "r") as z:
    z.extractall("/content/")

# Find corpus root (handles zip containing subfolder or not)
if os.path.isdir("/content/DDICorpus"):
    CORPUS_ROOT = "/content/DDICorpus"
elif os.path.isdir("/content/DDICorpus "):          # trailing-space edge case
    CORPUS_ROOT = "/content/DDICorpus "
else:
    # Find it dynamically
    for entry in os.listdir("/content"):
        if "DDI" in entry.upper() and os.path.isdir(f"/content/{entry}"):
            CORPUS_ROOT = f"/content/{entry}"
            break

print(f"Corpus root : {CORPUS_ROOT}")
print("Contents    :", os.listdir(CORPUS_ROOT))
"""

CELL_DATASET_CODE = """\
# ── ddi_dataset.py (inlined) ──────────────────────────────────────────────────
import os, glob
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

LABEL2ID = {"negative": 0, "mechanism": 1, "effect": 2, "advise": 3, "int": 4}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def _parse_char_offset(offset_str):
    parts = offset_str.split(";")[0]
    start, end = parts.split("-")
    return int(start), int(end)


def load_xml_files(folder_path):
    examples = []
    xml_files = glob.glob(os.path.join(folder_path, "**", "*.xml"), recursive=True)
    xml_files.sort()

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for sentence in root.iter("sentence"):
            sent_text = sentence.attrib.get("text", "")

            entities = {}
            for ent in sentence.findall("entity"):
                eid = ent.attrib["id"]
                start, end = _parse_char_offset(ent.attrib["charOffset"])
                etext = ent.attrib.get("text", sent_text[start:end + 1])
                entities[eid] = (start, end, etext)

            for pair in sentence.findall("pair"):
                e1_id = pair.attrib["e1"]
                e2_id = pair.attrib["e2"]
                ddi   = pair.attrib.get("ddi", "false").lower() == "true"
                itype = pair.attrib.get("type", "").lower() if ddi else "negative"

                if e1_id not in entities or e2_id not in entities:
                    continue

                label = LABEL2ID.get(itype, 0)
                e1_start, e1_end, e1_text = entities[e1_id]
                e2_start, e2_end, e2_text = entities[e2_id]
                marked = _mark_entities(sent_text, e1_start, e1_end, e2_start, e2_end)

                examples.append({
                    "text":    marked,
                    "label":   label,
                    "e1_text": e1_text,
                    "e2_text": e2_text,
                })
    return examples


def _mark_entities(text, e1_start, e1_end, e2_start, e2_end):
    if e1_start < e2_start:
        insertions = [
            (e2_end + 1,   " [/E2]"),
            (e2_start,     "[E2] "),
            (e1_end + 1,   " [/E1]"),
            (e1_start,     "[E1] "),
        ]
    else:
        insertions = [
            (e1_end + 1,   " [/E1]"),
            (e1_start,     "[E1] "),
            (e2_end + 1,   " [/E2]"),
            (e2_start,     "[E2] "),
        ]
    for pos, marker in insertions:
        text = text[:pos] + marker + text[pos:]
    return text


class DDIDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        e1_token_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        e2_token_id = self.tokenizer.convert_tokens_to_ids("[E2]")

        e1_pos = (input_ids == e1_token_id).nonzero(as_tuple=True)[0]
        e2_pos = (input_ids == e2_token_id).nonzero(as_tuple=True)[0]
        e1_pos = e1_pos[0].item() if len(e1_pos) > 0 else 0
        e2_pos = e2_pos[0].item() if len(e2_pos) > 0 else 0

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "e1_pos":         torch.tensor(e1_pos,      dtype=torch.long),
            "e2_pos":         torch.tensor(e2_pos,      dtype=torch.long),
            "label":          torch.tensor(ex["label"], dtype=torch.long),
        }


def get_tokenizer(model_name="dmis-lab/biobert-v1.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    )
    return tokenizer

print("Dataset code loaded.")
"""

CELL_MODEL_CODE = """\
# ── model.py (inlined) ───────────────────────────────────────────────────────
import torch
import torch.nn as nn
from transformers import AutoModel


class DDIClassifier(nn.Module):
    def __init__(self, model_name="dmis-lab/biobert-v1.1",
                 num_labels=NUM_LABELS, dropout=0.1,
                 vocab_size=None, freeze_layers=0):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        if vocab_size is not None:
            self.bert.resize_token_embeddings(vocab_size)

        # freeze_layers=0 means train ALL layers (full fine-tuning)
        if freeze_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for layer_idx in range(freeze_layers):
                for param in self.bert.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False

        hidden_size = self.bert.config.hidden_size   # 768
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos):
        outputs   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden    = outputs.last_hidden_state        # (batch, seq_len, 768)
        batch_size = hidden.size(0)

        e1_hidden = hidden[torch.arange(batch_size), e1_pos, :]  # (batch, 768)
        e2_hidden = hidden[torch.arange(batch_size), e2_pos, :]  # (batch, 768)
        combined  = torch.cat([e1_hidden, e2_hidden], dim=-1)    # (batch, 1536)
        combined  = self.dropout(combined)
        return self.classifier(combined)                          # (batch, 5)

print("Model code loaded.")
"""

CELL_TRAIN_CODE = """\
# ── train.py (inlined) ───────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from collections import Counter
from tqdm import tqdm


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, reduction="mean"):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        probs     = torch.exp(log_probs)
        log_pt    = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt        = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t   = self.alpha[targets]
        loss      = -alpha_t * (1 - pt) ** self.gamma * log_pt
        return loss.mean() if self.reduction == "mean" else loss.sum()


def compute_class_weights(examples):
    counts  = Counter(ex["label"] for ex in examples)
    total   = sum(counts.values())
    weights = [total / (NUM_LABELS * counts.get(i, 1)) for i in range(NUM_LABELS)]
    return torch.tensor(weights, dtype=torch.float)


def _val_f1(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["e1_pos"].to(device),
                batch["e2_pos"].to(device),
            )
            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(batch["label"].tolist())
    return f1_score(labels, preds, average="macro", zero_division=0)


def train_model(model, train_ds, val_ds, config):
    device = config["device"]
    model  = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    bert_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())

    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": config["bert_lr"]},
        {"params": head_params, "lr": config["head_lr"]},
    ], weight_decay=0.01)

    total_steps  = len(train_loader) * config["epochs"]
    warmup_steps = int(0.1 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    class_weights = compute_class_weights(train_ds.examples).to(device)
    criterion     = FocalLoss(alpha=class_weights, gamma=2.0)

    best_f1, best_state = 0.0, None

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            e1_pos         = batch["e1_pos"].to(device)
            e2_pos         = batch["e2_pos"].to(device)
            labels_b       = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, e1_pos, e2_pos)
            loss   = criterion(logits, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        val_f1 = _val_f1(model, val_loader, device)
        print(f"Epoch {epoch:2d} | loss={total_loss/len(train_loader):.4f} | val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"         New best! val_f1={best_f1:.4f}")

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    print(f"\\nTraining done. Best val macro-F1: {best_f1:.4f}")
    return model

print("Training code loaded.")
"""

CELL_EVALUATE_CODE = """\
# ── evaluate.py (inlined) ────────────────────────────────────────────────────
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score
)


def evaluate(model, test_ds, config):
    device = config["device"]
    model  = model.to(device)
    model.eval()

    loader = DataLoader(test_ds, batch_size=config["batch_size"],
                        shuffle=False, num_workers=2)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["e1_pos"].to(device),
                batch["e2_pos"].to(device),
            )
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    label_names  = [ID2LABEL[i] for i in range(NUM_LABELS)]
    positive_ids = [1, 2, 3, 4]

    print("\\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds,
                                 target_names=label_names, zero_division=0))

    macro_f1 = f1_score(all_labels, all_preds, labels=positive_ids,
                         average="macro", zero_division=0)
    macro_p  = precision_score(all_labels, all_preds, labels=positive_ids,
                                average="macro", zero_division=0)
    macro_r  = recall_score(all_labels, all_preds, labels=positive_ids,
                             average="macro", zero_division=0)

    print("="*60)
    print("OFFICIAL DDI METRIC (positive classes only)")
    print("="*60)
    print(f"  Precision : {macro_p:.4f}")
    print(f"  Recall    : {macro_r:.4f}")
    print(f"  Macro-F1  : {macro_f1:.4f}  ← headline number")

    return {"macro_f1": macro_f1, "precision": macro_p, "recall": macro_r}

print("Evaluate code loaded.")
"""

CELL_FINETUNE_RUN = """\
import os, random
import torch
from torch.utils.data import random_split
from collections import Counter

# ── Configuration ─────────────────────────────────────────────────────────────
torch.manual_seed(42)

TRAIN_DIR = os.path.join(CORPUS_ROOT, "Train")
TEST_DIR  = os.path.join(CORPUS_ROOT, "Test", "Test for DDI Extraction task")

CONFIG = {
    "model_name":    "dmis-lab/biobert-v1.1",
    "epochs":        5,        # 5 epochs for full fine-tuning on GPU
    "batch_size":    16,       # safe for T4 16 GB VRAM
    "bert_lr":       2e-5,     # standard BERT fine-tuning LR
    "head_lr":       1e-4,     # slightly higher for the classification head
    "freeze_layers": 0,        # 0 = ALL 12 layers trainable (full fine-tuning)
    "max_length":    128,
    "val_frac":      0.1,
    "device":        "cuda",
    "seed":          42,
}

print(f"Device      : {CONFIG['device']}")
print(f"GPU         : {torch.cuda.get_device_name(0)}")
print(f"freeze_layers = {CONFIG['freeze_layers']}  (0 = full fine-tuning, all 108M params)")
print(f"Epochs      : {CONFIG['epochs']}")
print(f"Batch size  : {CONFIG['batch_size']}")

# ── Load corpus ───────────────────────────────────────────────────────────────
print("\\nLoading corpus from XML files...")
train_examples = load_xml_files(TRAIN_DIR)
test_examples  = load_xml_files(TEST_DIR)
print(f"Train pairs : {len(train_examples)}")
print(f"Test  pairs : {len(test_examples)}")

dist = Counter(ex["label"] for ex in train_examples)
print("Label distribution:")
for lid, count in sorted(dist.items()):
    pct = 100 * count / len(train_examples)
    print(f"  {ID2LABEL[lid]:12s}: {count:6d}  ({pct:.1f}%)")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("\\nLoading BioBERT tokenizer...")
tokenizer = get_tokenizer(CONFIG["model_name"])

# ── Datasets ──────────────────────────────────────────────────────────────────
print("Tokenizing examples...")
full_train = DDIDataset(train_examples, tokenizer, CONFIG["max_length"])
test_ds    = DDIDataset(test_examples,  tokenizer, CONFIG["max_length"])

val_size   = int(len(full_train) * CONFIG["val_frac"])
train_size = len(full_train) - val_size
train_ds, val_ds = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(CONFIG["seed"])
)
train_ds.examples = [full_train.examples[i] for i in train_ds.indices]
val_ds.examples   = [full_train.examples[i] for i in val_ds.indices]
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# ── Model ─────────────────────────────────────────────────────────────────────
print("\\nBuilding DDIClassifier (freeze_layers=0)...")
model = DDIClassifier(
    model_name    = CONFIG["model_name"],
    vocab_size    = len(tokenizer),
    freeze_layers = CONFIG["freeze_layers"],
)
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params     : {total:,}")
print(f"Trainable params : {trainable:,}  (was 14M with freeze_layers=10)")

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"\\nStarting full fine-tuning for {CONFIG['epochs']} epochs...")
print("Expected time: ~20-25 min on T4 GPU\\n")
model = train_model(model, train_ds, val_ds, CONFIG)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\\nEvaluating on test set...")
results = evaluate(model, test_ds, CONFIG)
print(f"\\nFine-tuned macro-F1 : {results['macro_f1']:.4f}  (was 0.6303 before)")
print(f"Precision           : {results['precision']:.4f}")
print(f"Recall              : {results['recall']:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "biobert_ddi_finetuned.pt")
print("\\nWeights saved → biobert_ddi_finetuned.pt")
print("Run the next cell to download them.")
"""

CELL_DOWNLOAD = """\
from google.colab import files

files.download("biobert_ddi_finetuned.pt")

print("Download started.")
print()
print("Once downloaded, do this on your Mac:")
print("  1. Move biobert_ddi_finetuned.pt into your project folder")
print("  2. Rename (or copy) it to biobert_ddi_best.pt")
print("  3. Restart api.py  — it will load the new weights automatically")
print()
print("The UI and all predictions will now use the fine-tuned model.")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Build notebook structure
# ─────────────────────────────────────────────────────────────────────────────

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code_cell(source):
    return {
        "cell_type":        "code",
        "execution_count":  None,
        "metadata":         {},
        "outputs":          [],
        "source":           source,
    }


notebook = {
    "nbformat":       4,
    "nbformat_minor": 0,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name":        "DDI_FineTune_Colab.ipynb",
            "provenance":  [],
            "gpuType":     "T4"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name":         "python3"
        },
        "language_info": {"name": "python"}
    },
    "cells": [
        md_cell(
            "# DDI Fine-Tuning — Google Colab\n\n"
            "Full fine-tuning of BioBERT with **all 12 layers trainable** "
            "(`freeze_layers=0`).\n\n"
            "**Steps:**\n"
            "1. `Runtime > Change runtime type > T4 GPU` (free tier is fine)\n"
            "2. `Runtime > Run all`\n"
            "3. When prompted, upload your `DDICorpus.zip`\n"
            "4. Wait ~20-25 min — weights download automatically at the end\n\n"
            "**Expected improvement:** macro-F1 0.63 → ~0.75–0.80"
        ),
        md_cell("## Step 1 — Check GPU"),
        code_cell(CELL_GPU_CHECK),
        md_cell("## Step 2 — Install packages"),
        code_cell(CELL_INSTALL),
        md_cell(
            "## Step 3 — Upload corpus\n\n"
            "First, zip your corpus on your Mac:\n"
            "- Find `DDICorpus` folder on your Desktop\n"
            "- Right-click > Compress → produces `DDICorpus.zip` (~8-9 MB)\n\n"
            "Then upload it below."
        ),
        code_cell(CELL_UPLOAD_CORPUS),
        md_cell("## Step 4 — Load model code"),
        code_cell(CELL_DATASET_CODE),
        code_cell(CELL_MODEL_CODE),
        code_cell(CELL_TRAIN_CODE),
        code_cell(CELL_EVALUATE_CODE),
        md_cell("## Step 5 — Run full fine-tuning (~20-25 min)"),
        code_cell(CELL_FINETUNE_RUN),
        md_cell("## Step 6 — Download fine-tuned weights"),
        code_cell(CELL_DOWNLOAD),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Write to file
# ─────────────────────────────────────────────────────────────────────────────

out_path = "DDI_FineTune_Colab.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written → {out_path}")
print(f"Upload this file to https://colab.research.google.com")
