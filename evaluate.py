"""
evaluate.py
───────────
Generates the final evaluation report on the test set.

METRICS EXPLAINED:

  Precision (per class):
    Of all the times the model predicted class X, how often was it right?
    High precision = low false positive rate.

  Recall (per class):
    Of all the actual instances of class X, how many did the model find?
    High recall = low false negative rate.

  F1 Score (per class):
    Harmonic mean of precision and recall.
    F1 = 2 * (P * R) / (P + R)
    This is the primary metric for DDI — both missing an interaction AND
    falsely reporting one are costly in a medical setting.

  Macro-F1:
    Average F1 across all classes, treating each class equally.
    This is the official SemEval-2013 DDI metric.

  We EXCLUDE the "negative" class from the official macro-F1, because
  that is the convention used in the SemEval-2013 evaluation. The negative
  class is trivially easy and inflates the score if included.
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
import numpy as np

from ddi_dataset import ID2LABEL, NUM_LABELS


def evaluate(model, test_dataset, config):
    """
    Run the model on the full test set and print a detailed report.

    Args:
        model        : trained DDIClassifier
        test_dataset : DDIDataset wrapping the test split
        config       : dict with device and batch_size keys

    Returns:
        dict with macro_f1, precision, recall (excluding negative class)
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
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            e1_pos         = batch["e1_pos"].to(device)
            e2_pos         = batch["e2_pos"].to(device)
            labels         = batch["label"]

            logits = model(input_ids, attention_mask, e1_pos, e2_pos)
            preds  = logits.argmax(dim=-1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # ── Full classification report ─────────────────────────────────────────────
    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]

    print("\n" + "="*65)
    print("FULL CLASSIFICATION REPORT (all 5 classes)")
    print("="*65)
    print(classification_report(
        all_labels, all_preds,
        target_names=label_names,
        zero_division=0,
    ))

    # ── Official SemEval-2013 metric: macro-F1 over positive classes ONLY ─────
    # Positive class IDs: 1 (mechanism), 2 (effect), 3 (advise), 4 (int)
    positive_ids = [1, 2, 3, 4]

    macro_f1  = f1_score(all_labels,  all_preds, labels=positive_ids, average="macro",  zero_division=0)
    macro_p   = precision_score(all_labels, all_preds, labels=positive_ids, average="macro",  zero_division=0)
    macro_r   = recall_score(all_labels,   all_preds, labels=positive_ids, average="macro",  zero_division=0)

    print("="*65)
    print("OFFICIAL DDI METRIC (macro over positive classes only)")
    print("="*65)
    print(f"  Macro Precision : {macro_p:.4f}")
    print(f"  Macro Recall    : {macro_r:.4f}")
    print(f"  Macro F1        : {macro_f1:.4f}  ← this is the headline number")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("CONFUSION MATRIX")
    print("Rows = true label, Columns = predicted label")
    print("Order:", label_names)
    print("="*65)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_LABELS)))
    # Print with row labels
    col_header = "        " + "  ".join(f"{n[:6]:>6}" for n in label_names)
    print(col_header)
    for i, row in enumerate(cm):
        row_str = f"{label_names[i][:8]:8s}" + "  ".join(f"{v:6d}" for v in row)
        print(row_str)

    # ── Per-class breakdown ────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("PER-CLASS BREAKDOWN (positive classes)")
    print("="*65)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_p  = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_r  = recall_score(all_labels, all_preds, average=None, zero_division=0)

    print(f"  {'Class':<18} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*50}")
    for i in positive_ids:
        name = ID2LABEL[i]
        print(f"  {name:<18} {per_class_p[i]:>10.4f} {per_class_r[i]:>10.4f} {per_class_f1[i]:>10.4f}")

    return {
        "macro_f1":  macro_f1,
        "precision": macro_p,
        "recall":    macro_r,
    }
