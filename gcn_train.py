"""
gcn_train.py
────────────
Training loop for the GCN DDI classifier.

Reuses FocalLoss and compute_class_weights from train.py
since they are model-agnostic — the same class imbalance problem
(85% negative) exists for both the BioBERT and GCN tracks.

KEY DIFFERENCES from BioBERT training:
  - No pre-trained weights → higher learning rate (1e-3 vs 2e-5)
  - No warmup scheduler needed (GCNs don't have BERT's sensitivity to LR)
  - More epochs needed (10 vs 3) because GCN learns from scratch
  - Faster per-batch (no 108M param backward pass)
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

# Reuse FocalLoss and compute_class_weights from the BioBERT track
from train import FocalLoss, compute_class_weights


def train_gcn(model, train_dataset, val_dataset, config):
    """
    Train the GCN model.

    Args:
        model         : DDIGCN instance
        train_dataset : GCNDataset for training
        val_dataset   : GCNDataset for validation
        config        : dict with hyperparameters (from gcn_main.py)

    Returns:
        model : best trained model (by validation macro-F1)
    """
    device = config["device"]
    model  = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # GCN trains from scratch so we use a higher LR than BioBERT fine-tuning.
    # Adam with weight decay (AdamW) prevents overfitting on small GCN.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # ── Learning rate scheduler ────────────────────────────────────────────────
    # ReduceLROnPlateau: halves LR when val F1 stops improving for 3 epochs.
    # More robust than linear decay for smaller models.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # ── Loss function ──────────────────────────────────────────────────────────
    class_weights = compute_class_weights(train_dataset.examples).to(device)
    criterion     = FocalLoss(alpha=class_weights, gamma=2.0)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    best_state  = None

    for epoch in range(1, config["epochs"] + 1):

        # ── Train phase ────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']} [train]"):
            adj            = batch["adj"].to(device)
            node_features  = batch["node_features"].to(device)
            e1_pos         = batch["e1_pos"].to(device)
            e2_pos         = batch["e2_pos"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(adj, node_features, e1_pos, e2_pos)
            loss   = criterion(logits, labels)
            loss.backward()

            # Gradient clipping (good practice even for small models)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Validation phase ───────────────────────────────────────────────────
        val_f1 = _evaluate_f1(model, val_loader, device)
        scheduler.step(val_f1)

        print(f"Epoch {epoch:2d} | loss={avg_loss:.4f} | val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"         ✓ New best! Saving checkpoint (val_f1={best_val_f1:.4f})")

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    print(f"\nTraining complete. Best val macro-F1: {best_val_f1:.4f}")
    return model


def _evaluate_f1(model, loader, device):
    """Quick macro-F1 over positive classes for use inside training loop."""
    model.eval()
    all_preds, all_labels = [], []

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

    return f1_score(all_labels, all_preds, average="macro", zero_division=0)
