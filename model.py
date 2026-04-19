"""
model.py
────────
Defines the BioBERT-based DDI classifier.

ARCHITECTURE EXPLAINED:

  [Marked Sentence]
        ↓
  BioBERT (12-layer Transformer)
        ↓
  Hidden states for every token  (shape: batch × seq_len × 768)
        ↓
  Extract hidden state at [E1] position  →  768-dim vector
  Extract hidden state at [E2] position  →  768-dim vector
        ↓
  Concatenate them                       →  1536-dim vector
        ↓
  Dropout (regularisation)
        ↓
  Linear layer: 1536 → 5 classes
        ↓
  Output logits (raw scores, one per class)

WHY THIS APPROACH:
  - Using [E1] and [E2] hidden states (rather than [CLS]) focuses BioBERT
    specifically on the two drug entities and their context.
  - This is the standard "entity start" approach from the RE literature
    and consistently outperforms [CLS]-only classification.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from ddi_dataset import NUM_LABELS


class DDIClassifier(nn.Module):
    """
    BioBERT encoder + entity-span classification head.

    Args:
        model_name : HuggingFace model ID. Default is BioBERT v1.1.
                     Can swap for "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
                     if you want PubMedBERT instead.
        num_labels : number of output classes (5 for DDI).
        dropout    : dropout probability applied before the linear head.
        vocab_size : new vocab size after adding entity marker tokens.
                     Pass tokenizer.vocab_size + 4 from main.py.
    """

    def __init__(
        self,
        model_name="dmis-lab/biobert-v1.1",
        num_labels=NUM_LABELS,
        dropout=0.1,
        vocab_size=None,
        freeze_layers=10,
    ):
        """
        Args:
            freeze_layers : number of BERT encoder layers to freeze (0-12).
                            Frozen layers skip the backward pass, making
                            training significantly faster on CPU.
                            freeze_layers=10 means only the top 2 layers
                            (layers 10 and 11) + the head are trained.
                            This reduces trainable params from 108M → ~14M
                            and cuts backward-pass time by ~80%.
        """
        super().__init__()

        # ── 1. Load pre-trained BioBERT ───────────────────────────────────────
        self.bert = AutoModel.from_pretrained(model_name)

        # ── 2. Resize token embeddings ────────────────────────────────────────
        if vocab_size is not None:
            self.bert.resize_token_embeddings(vocab_size)

        # ── 3. Freeze bottom layers ───────────────────────────────────────────
        # Freezing means: no gradient computed → no weight update → faster.
        # The frozen layers still run the forward pass (we need their output),
        # but we don't backpropagate through them.
        # Embeddings are always frozen when any layers are frozen.
        if freeze_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for layer_idx in range(freeze_layers):
                for param in self.bert.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # 768 for bert-base models

        # ── 4. Classification head ────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos):
        """
        Args:
            input_ids      : (batch, seq_len)  token IDs
            attention_mask : (batch, seq_len)  1 for real tokens, 0 for padding
            e1_pos         : (batch,)           token index of [E1] marker
            e2_pos         : (batch,)           token index of [E2] marker

        Returns:
            logits         : (batch, num_labels)  raw class scores
        """
        # Run BioBERT — outputs hidden states for every token
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden  = outputs.last_hidden_state  # (batch, seq_len, 768)

        batch_size = hidden.size(0)

        # Extract the hidden state at the [E1] token position for each example
        # hidden[i, e1_pos[i], :] gives the 768-dim vector for sample i
        e1_hidden = hidden[torch.arange(batch_size), e1_pos, :]  # (batch, 768)
        e2_hidden = hidden[torch.arange(batch_size), e2_pos, :]  # (batch, 768)

        # Concatenate both entity representations
        combined = torch.cat([e1_hidden, e2_hidden], dim=-1)  # (batch, 1536)

        combined = self.dropout(combined)
        logits   = self.classifier(combined)                  # (batch, 5)

        return logits
