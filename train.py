"""
train.py
────────
Training loop for the DDI BioBERT classifier.

KEY DESIGN DECISIONS EXPLAINED:

1. FOCAL LOSS (not standard CrossEntropy):
   The DDI dataset is severely imbalanced — roughly 75-80% of all entity
   pairs are "negative" (no interaction). If we use plain CrossEntropy,
   the model learns to predict "negative" for everything and still gets
   ~80% accuracy — which is useless.

   Focal Loss adds a factor (1 - p_t)^gamma that DOWN-WEIGHTS easy examples
   (samples the model already classifies correctly with high confidence) and
   UP-WEIGHTS hard examples (the rare positive interaction classes).

   Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
   - gamma=2 is the standard setting from the original Focal Loss paper.
   - alpha is a per-class weight we compute from the training data.

2. LEARNING RATE SCHEDULE:
   We use a linear warmup for the first 10% of steps, then linear decay.
   This is the standard recipe for fine-tuning BERT-family models.
   A large LR at the start causes the pre-trained weights to "forget"
   everything; warmup prevents this.

3. GRADIENT CLIPPING:
   Clips gradients to max norm 1.0 to prevent exploding gradients,
   which are common when fine-tuning large Transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Args:
        alpha  : (num_classes,) tensor — per-class weights.
                  Usually set to inverse class frequency so rare classes
                  get more attention.
        gamma  : focusing parameter. gamma=0 → standard cross-entropy.
                  gamma=2 is the standard value from the paper.
    """

    def __init__(self, alpha, gamma=2.0, reduction="mean"):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits  : (batch, num_classes)  raw model outputs
            targets : (batch,)              integer ground-truth labels
        """
        # Standard cross-entropy gives us log(p_t) per sample
        log_probs = F.log_softmax(logits, dim=-1)           # (batch, C)
        probs     = torch.exp(log_probs)                    # (batch, C)

        # Gather the probability and log-probability for the TRUE class
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (batch,)
        pt     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)      # (batch,)

        # Per-class weight for each sample
        alpha_t = self.alpha[targets]  # (batch,)

        # Focal loss formula
        loss = -alpha_t * (1 - pt) ** self.gamma * log_pt  # (batch,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(examples):
    """
    Compute inverse-frequency weights for each class.
    Rarer classes get higher weight so the model pays attention to them.
    """
    from collections import Counter
    from ddi_dataset import NUM_LABELS

    counts = Counter(ex["label"] for ex in examples)
    total  = sum(counts.values())

    weights = []
    for cls_id in range(NUM_LABELS):
        count = counts.get(cls_id, 1)  # avoid div-by-zero
        weights.append(total / (NUM_LABELS * count))

    return torch.tensor(weights, dtype=torch.float)


# ── Training function ─────────────────────────────────────────────────────────

def train_model(model, train_dataset, val_dataset, config):
    """
    Full training loop.

    Args:
        model         : DDIClassifier instance
        train_dataset : DDIDataset for training
        val_dataset   : DDIDataset for validation
        config        : dict with hyperparameters (see main.py)

    Returns:
        model : trained model (best checkpoint by val F1)
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
    # Small LR for the pre-trained BERT layers, slightly larger for the new
    # classification head (it starts random so it can afford a bigger step).
    bert_params       = list(model.bert.parameters())
    head_params       = list(model.classifier.parameters()) + list(model.dropout.parameters())

    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": config["bert_lr"]},
        {"params": head_params, "lr": config["head_lr"]},
    ], weight_decay=0.01)

    # ── LR scheduler ──────────────────────────────────────────────────────────
    total_steps  = len(train_loader) * config["epochs"]
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Loss function ──────────────────────────────────────────────────────────
    class_weights = compute_class_weights(train_dataset.examples).to(device)
    criterion     = FocalLoss(alpha=class_weights, gamma=2.0)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_f1   = 0.0
    best_state    = None

    for epoch in range(1, config["epochs"] + 1):
        # ── Train phase ────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']} [train]"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            e1_pos         = batch["e1_pos"].to(device)
            e2_pos         = batch["e2_pos"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask, e1_pos, e2_pos)
            loss   = criterion(logits, labels)

            loss.backward()
            # Gradient clipping — prevents huge gradient updates
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ── Validation phase ───────────────────────────────────────────────────
        val_f1 = evaluate_during_training(model, val_loader, device)

        print(f"Epoch {epoch:2d} | train_loss={avg_train_loss:.4f} | val_macro_f1={val_f1:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"         ✓ New best! Saving checkpoint (val_f1={best_val_f1:.4f})")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    print(f"\nTraining complete. Best val macro-F1: {best_val_f1:.4f}")
    return model


def evaluate_during_training(model, loader, device):
    """Quick F1 computation used inside the training loop (no detailed report)."""
    from sklearn.metrics import f1_score

    model.eval()
    all_preds, all_labels = [], []

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

    # macro-F1 across all 5 classes
    return f1_score(all_labels, all_preds, average="macro", zero_division=0)
