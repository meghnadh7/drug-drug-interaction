"""
gcn_evaluate.py
───────────────
Evaluation report for the GCN model on the test set.
Mirrors evaluate.py from the BioBERT track so both tracks
produce identical, comparable output formats.
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from gcn_dataset import ID2LABEL, NUM_LABELS


def evaluate_gcn(model, test_dataset, config):
    """
    Run the GCN on the full test set and print a detailed report.

    Args:
        model        : trained DDIGCN
        test_dataset : GCNDataset wrapping the test split
        config       : dict with device and batch_size

    Returns:
        dict with macro_f1, precision, recall (positive classes only)
    """
    device = config["device"]
    model  = model.to(device)
    model.eval()

    loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            adj           = batch["adj"].to(device)
            node_features = batch["node_features"].to(device)
            e1_pos        = batch["e1_pos"].to(device)
            e2_pos        = batch["e2_pos"].to(device)
            labels        = batch["label"]

            logits = model(adj, node_features, e1_pos, e2_pos)
            preds  = logits.argmax(dim=-1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    label_names  = [ID2LABEL[i] for i in range(NUM_LABELS)]
    positive_ids = [1, 2, 3, 4]   # exclude negative class from official metric

    # ── Full report ────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("GCN — FULL CLASSIFICATION REPORT (all 5 classes)")
    print("="*65)
    print(classification_report(
        all_labels, all_preds,
        target_names=label_names,
        zero_division=0,
    ))

    # ── Official DDI metric ────────────────────────────────────────────────────
    macro_f1 = f1_score(all_labels, all_preds, labels=positive_ids, average="macro",  zero_division=0)
    macro_p  = precision_score(all_labels, all_preds, labels=positive_ids, average="macro", zero_division=0)
    macro_r  = recall_score(all_labels,   all_preds, labels=positive_ids, average="macro", zero_division=0)

    print("="*65)
    print("GCN — OFFICIAL DDI METRIC (macro over positive classes)")
    print("="*65)
    print(f"  Macro Precision : {macro_p:.4f}")
    print(f"  Macro Recall    : {macro_r:.4f}")
    print(f"  Macro F1        : {macro_f1:.4f}  ← headline number")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("GCN — CONFUSION MATRIX")
    print("Rows = true label, Columns = predicted label")
    print("Order:", label_names)
    print("="*65)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_LABELS)))
    col_header = "        " + "  ".join(f"{n[:6]:>6}" for n in label_names)
    print(col_header)
    for i, row in enumerate(cm):
        row_str = f"{label_names[i][:8]:8s}" + "  ".join(f"{v:6d}" for v in row)
        print(row_str)

    # ── Per-class breakdown ────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("GCN — PER-CLASS BREAKDOWN (positive classes)")
    print("="*65)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_p  = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_r  = recall_score(all_labels,   all_preds, average=None, zero_division=0)

    print(f"  {'Class':<18} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*50}")
    for i in positive_ids:
        print(f"  {ID2LABEL[i]:<18} {per_class_p[i]:>10.4f} {per_class_r[i]:>10.4f} {per_class_f1[i]:>10.4f}")

    return {"macro_f1": macro_f1, "precision": macro_p, "recall": macro_r}
