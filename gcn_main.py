"""
gcn_main.py
───────────
Entry point for Track 2: GCN over Dependency Parse Trees.

Run with:
    python gcn_main.py

First run parses all sentences with spaCy and caches them (~5-10 min).
Subsequent runs load from cache and start training immediately.

Full training (demo_mode=False) takes ~15-30 min on CPU
because GCN is much lighter than BioBERT.
"""

import os
import torch
from torch.utils.data import random_split

from gcn_dataset import load_xml_files_gcn, GCNDataset, load_spacy, ID2LABEL
from gcn_model    import DDIGCN
from gcn_train    import train_gcn
from gcn_evaluate import evaluate_gcn

# ── Corpus paths (same dataset as BioBERT track) ──────────────────────────────
CORPUS_ROOT = os.path.expanduser("~/Desktop/DDICorpus")
TRAIN_DIR   = os.path.join(CORPUS_ROOT, "Train")
TEST_DIR    = os.path.join(CORPUS_ROOT, "Test", "Test for DDI Extraction task")

# ── Configuration ─────────────────────────────────────────────────────────────
GCN_CONFIG = {
    # Model architecture
    "input_dim":  300,    # spaCy word vector dimension (fixed)
    "hidden_dim": 256,    # GCN hidden layer size
    "num_layers": 3,      # number of GCN layers
    "dropout":    0.5,    # dropout (GCNs typically use 0.5)

    # Training
    "epochs":       10,   # GCN trains from scratch → needs more epochs than BERT
    "batch_size":   64,
    "lr":           1e-3, # higher LR than BioBERT (no pre-trained weights)
    "weight_decay": 5e-4,

    # Data
    "val_frac": 0.1,

    # Demo mode — set True for fast pipeline verification (~2 min)
    # Set False for full training (~20 min on CPU)
    "demo_mode":       False,
    "demo_train_size": 1500,
    "demo_test_size":  300,

    # Cache file for spaCy parse results
    "cache_file": "gcn_cache.pkl",

    # Hardware
    "device": (
        "cuda" if torch.cuda.is_available() else "cpu"
    ),

    "seed": 42,
}


def main():
    torch.manual_seed(GCN_CONFIG["seed"])
    print(f"\nDevice : {GCN_CONFIG['device']}")
    print(f"Model  : GCN ({GCN_CONFIG['num_layers']} layers, "
          f"hidden={GCN_CONFIG['hidden_dim']})\n")

    if GCN_CONFIG["demo_mode"]:
        print("*** DEMO MODE — small subset for pipeline verification ***")
        print("*** Set demo_mode=False in GCN_CONFIG for full training ***\n")

    # ── Step 1: Load spaCy ────────────────────────────────────────────────────
    print("[1/5] Loading spaCy model (en_core_web_md) ...")
    nlp = load_spacy()
    print("      spaCy loaded.")

    # ── Step 2: Parse corpus + build graphs ───────────────────────────────────
    # This is slow the FIRST time (~5-10 min for full corpus).
    # Results are cached to gcn_cache.pkl for instant reload on next run.
    print("\n[2/5] Building dependency graphs from XML corpus ...")
    print("      (First run parses with spaCy and caches — be patient)")

    train_examples = load_xml_files_gcn(
        TRAIN_DIR, nlp, cache_path=GCN_CONFIG["cache_file"]
    )
    test_examples = load_xml_files_gcn(
        TEST_DIR, nlp, cache_path=GCN_CONFIG["cache_file"]
    )

    # Optionally trim to demo size
    if GCN_CONFIG["demo_mode"]:
        import random
        rng = random.Random(GCN_CONFIG["seed"])
        rng.shuffle(train_examples)
        rng.shuffle(test_examples)
        train_examples = train_examples[: GCN_CONFIG["demo_train_size"]]
        test_examples  = test_examples[:  GCN_CONFIG["demo_test_size"]]

    print(f"       Train pairs : {len(train_examples)}")
    print(f"       Test  pairs : {len(test_examples)}")

    # Show label distribution
    from collections import Counter
    dist = Counter(ex["label"] for ex in train_examples)
    print("       Label distribution (train):")
    for lid, count in sorted(dist.items()):
        pct = 100 * count / len(train_examples)
        print(f"         {ID2LABEL[lid]:12s}: {count:5d}  ({pct:.1f}%)")

    # ── Step 3: Build PyTorch datasets ────────────────────────────────────────
    print("\n[3/5] Building PyTorch datasets ...")
    full_train = GCNDataset(train_examples)
    test_ds    = GCNDataset(test_examples)

    val_size   = int(len(full_train) * GCN_CONFIG["val_frac"])
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(GCN_CONFIG["seed"]),
    )
    train_ds.examples = [full_train.examples[i] for i in train_ds.indices]
    val_ds.examples   = [full_train.examples[i] for i in val_ds.indices]

    print(f"       Train : {len(train_ds)}")
    print(f"       Val   : {len(val_ds)}")
    print(f"       Test  : {len(test_ds)}")

    # ── Step 4: Build model ───────────────────────────────────────────────────
    print(f"\n[4/5] Building DDIGCN ...")
    model = DDIGCN(
        input_dim  = GCN_CONFIG["input_dim"],
        hidden_dim = GCN_CONFIG["hidden_dim"],
        num_layers = GCN_CONFIG["num_layers"],
        dropout    = GCN_CONFIG["dropout"],
    )
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       Total params    : {total:,}")
    print(f"       Trainable params: {trainable:,}")
    print(f"       (Compare: BioBERT has 108M params — GCN is ~{108_321_029 // total}x smaller)")

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    print(f"\n[5/5] Training for {GCN_CONFIG['epochs']} epochs ...")
    model = train_gcn(model, train_ds, val_ds, GCN_CONFIG)

    # ── Step 6: Evaluate ──────────────────────────────────────────────────────
    print("\n[6/6] Evaluating on test set ...")
    results = evaluate_gcn(model, test_ds, GCN_CONFIG)

    print(f"\nGCN Final macro-F1 (positive classes): {results['macro_f1']:.4f}")

    torch.save(model.state_dict(), "gcn_ddi_best.pt")
    print("GCN model weights saved → gcn_ddi_best.pt")


if __name__ == "__main__":
    main()
