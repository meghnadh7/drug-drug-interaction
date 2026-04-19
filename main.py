"""
main.py
───────
Entry point. Runs the full BioBERT DDI pipeline:
  1. Load + preprocess the local XML corpus
  2. Build the tokenizer and model
  3. Train with Focal Loss
  4. Evaluate on the test set and print F1 / Precision / Recall

Run with:
    python main.py
"""

import os
import torch
from torch.utils.data import random_split

from ddi_dataset import load_xml_files, DDIDataset, get_tokenizer, ID2LABEL
from model import DDIClassifier
from train import train_model
from evaluate import evaluate

# ── Paths to local corpus ─────────────────────────────────────────────────────
CORPUS_ROOT = os.path.expanduser("~/Desktop/DDICorpus")
TRAIN_DIR   = os.path.join(CORPUS_ROOT, "Train")
TEST_DIR    = os.path.join(CORPUS_ROOT, "Test", "Test for DDI Extraction task")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — change values here, not scattered through the code
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Model — swap the two lines to switch between BioBERT and PubMedBERT
    "model_name": "dmis-lab/biobert-v1.1",
    # "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",

    # Training
    "epochs":        3,      # 3 epochs is standard for BERT fine-tuning
    "batch_size":    32,     # reduce to 16 if you get out-of-memory errors
    "bert_lr":       2e-5,   # LR for the trainable BERT layers
    "head_lr":       1e-4,   # LR for the classification head
    "freeze_layers": 10,     # freeze bottom 10/12 BERT layers; only train top 2
                             # reduces trainable params from 108M → ~14M,
                             # cuts per-batch time by ~80% on CPU

    # Data
    "max_length": 128,    # DDI sentences are short; 128 covers >99% of them
    "val_frac":   0.1,    # fraction of train data held out for validation

    # Demo mode — set True to run on a small subset (~5 min on CPU)
    # to verify the full pipeline works before committing to full training.
    # Set False for the real training run.
    "demo_mode":       False,
    "demo_train_size": 1500,   # training examples in demo mode
    "demo_test_size":  300,    # test examples in demo mode

    # Hardware
    "device": (
        "cuda" if torch.cuda.is_available() else "cpu"
    ),

    "seed": 42,
}


def main():
    torch.manual_seed(CONFIG["seed"])
    print(f"\nDevice : {CONFIG['device']}")
    print(f"Model  : {CONFIG['model_name']}\n")

    if CONFIG["demo_mode"]:
        print("*** DEMO MODE — running on small subset to verify pipeline ***")
        print("*** Set demo_mode=False in CONFIG for the full training run  ***\n")

    # ── Step 1: Parse XML corpus ──────────────────────────────────────────────
    print("[1/5] Loading corpus from XML files ...")
    train_examples = load_xml_files(TRAIN_DIR)
    test_examples  = load_xml_files(TEST_DIR)

    if CONFIG["demo_mode"]:
        import random
        rng = random.Random(CONFIG["seed"])
        rng.shuffle(train_examples)
        rng.shuffle(test_examples)
        train_examples = train_examples[: CONFIG["demo_train_size"]]
        test_examples  = test_examples[: CONFIG["demo_test_size"]]
    print(f"       Train pairs : {len(train_examples)}")
    print(f"       Test  pairs : {len(test_examples)}")

    # Show label distribution so you can see the class imbalance
    from collections import Counter
    dist = Counter(ex["label"] for ex in train_examples)
    print("       Train label distribution:")
    for lid, count in sorted(dist.items()):
        pct = 100 * count / len(train_examples)
        print(f"         {ID2LABEL[lid]:12s}: {count:6d}  ({pct:.1f}%)")

    # ── Step 2: Load tokenizer ────────────────────────────────────────────────
    print(f"\n[2/5] Loading tokenizer ...")
    tokenizer = get_tokenizer(CONFIG["model_name"])

    # ── Step 3: Build PyTorch datasets ────────────────────────────────────────
    print("[3/5] Tokenizing examples ...")
    full_train = DDIDataset(train_examples, tokenizer, CONFIG["max_length"])
    test_ds    = DDIDataset(test_examples,  tokenizer, CONFIG["max_length"])

    val_size   = int(len(full_train) * CONFIG["val_frac"])
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG["seed"]),
    )
    # Attach .examples so train.py can compute class weights
    train_ds.examples = [full_train.examples[i] for i in train_ds.indices]
    val_ds.examples   = [full_train.examples[i] for i in val_ds.indices]

    print(f"       Train : {len(train_ds)}")
    print(f"       Val   : {len(val_ds)}")
    print(f"       Test  : {len(test_ds)}")

    # ── Step 4: Build model ───────────────────────────────────────────────────
    print(f"\n[4/5] Building DDIClassifier ...")
    model = DDIClassifier(
        model_name=CONFIG["model_name"],
        vocab_size=len(tokenizer),
        freeze_layers=CONFIG["freeze_layers"],
    )
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       Total params    : {total:,}")
    print(f"       Trainable params: {trainable:,}")

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    print(f"\n[5/5] Training for {CONFIG['epochs']} epochs ...")
    model = train_model(model, train_ds, val_ds, CONFIG)

    # ── Step 6: Final evaluation ──────────────────────────────────────────────
    print("\n[6/6] Evaluating on test set ...")
    results = evaluate(model, test_ds, CONFIG)

    print(f"\nFinal macro-F1 (positive classes only): {results['macro_f1']:.4f}")

    torch.save(model.state_dict(), "biobert_ddi_best.pt")
    print("Model weights saved → biobert_ddi_best.pt")


if __name__ == "__main__":
    main()
