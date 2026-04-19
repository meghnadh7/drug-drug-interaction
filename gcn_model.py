"""
gcn_model.py
────────────
Defines the Graph Convolutional Network (GCN) for DDI classification.

ARCHITECTURE EXPLAINED:

  [Dependency Parse Graph of the sentence]
  Nodes = words,  Edges = grammatical relationships
        ↓
  GCN Layer 1: each node gathers information from its neighbours
        ↓
  GCN Layer 2: each node now knows about 2-hop neighbours
        ↓
  GCN Layer 3: each node has seen 3 hops away — reaches most of the tree
        ↓
  Extract node embedding at drug1 position   →  256-dim vector
  Extract node embedding at drug2 position   →  256-dim vector
        ↓
  Concatenate                                →  512-dim vector
        ↓
  Fully-connected layers: 512 → 256 → 5 classes
        ↓
  Output logits (one per interaction type)

HOW A GCN LAYER WORKS:
  Standard formula: H' = ReLU( D^{-1} A H W )
  - A = adjacency matrix (which nodes are connected)
  - D = degree matrix (normalisation)
  - H = current node features (matrix of shape nodes × features)
  - W = learnable weight matrix
  In plain English: each node's new representation is the
  AVERAGE of its neighbours' representations, passed through
  a learned linear transformation.

WHY 3 LAYERS?
  With 1 layer: each drug node only sees its direct grammatical neighbours.
  With 2 layers: it sees 2 hops away (e.g. "inhibit" can see "metabolism").
  With 3 layers: most sentence trees have diameter ≤ 6, so 3 layers covers
  most of the relevant path between the two drugs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcn_dataset import NUM_LABELS, NODE_DIM


# ── Single GCN Layer ──────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """
    One graph convolutional layer.

    Performs: H' = ReLU( BN( A_norm @ H @ W ) )
    where A_norm is the pre-normalised adjacency matrix passed in.

    Args:
        in_features  : dimension of input node features
        out_features : dimension of output node features
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn     = nn.BatchNorm1d(out_features)

    def forward(self, x, adj):
        """
        Args:
            x   : (batch, max_nodes, in_features)  — current node features
            adj : (batch, max_nodes, max_nodes)     — normalised adjacency

        Returns:
            h   : (batch, max_nodes, out_features)  — updated node features
        """
        # Step 1: Message aggregation
        # Each node collects a weighted average of its neighbours' features
        aggregated = torch.bmm(adj, x)                # (batch, nodes, in_feat)

        # Step 2: Linear transformation (learned weights)
        h = self.linear(aggregated)                   # (batch, nodes, out_feat)

        # Step 3: Batch normalisation
        # BN expects (N, C) — reshape, normalise, reshape back
        batch, nodes, feat = h.shape
        h = self.bn(h.view(batch * nodes, feat))
        h = h.view(batch, nodes, feat)

        # Step 4: Non-linearity
        return F.relu(h)


# ── Full GCN Model ────────────────────────────────────────────────────────────

class DDIGCN(nn.Module):
    """
    Multi-layer GCN for DDI relation classification.

    Args:
        input_dim  : node feature dimension (300 for spaCy en_core_web_md)
        hidden_dim : hidden dimension of GCN layers
        num_layers : number of GCN layers (3 is a good default)
        dropout    : dropout probability (0.5 is standard for GCNs)
        num_labels : number of output classes (5 for DDI)
    """

    def __init__(
        self,
        input_dim  = NODE_DIM,   # 300
        hidden_dim = 256,
        num_layers = 3,
        dropout    = 0.5,
        num_labels = NUM_LABELS,
    ):
        super().__init__()

        # ── GCN encoder stack ─────────────────────────────────────────────────
        self.gcn_layers = nn.ModuleList()

        # First layer: input_dim → hidden_dim
        self.gcn_layers.append(GCNLayer(input_dim, hidden_dim))

        # Remaining layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # ── Classification head ───────────────────────────────────────────────
        # Concatenation of two drug node embeddings → hidden_dim * 2
        # Two-layer MLP for final classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, adj, node_features, e1_pos, e2_pos):
        """
        Args:
            adj           : (batch, MAX_NODES, MAX_NODES)  adjacency matrix
            node_features : (batch, MAX_NODES, input_dim)  word vectors
            e1_pos        : (batch,)  token index of drug 1
            e2_pos        : (batch,)  token index of drug 2

        Returns:
            logits        : (batch, num_labels)
        """
        x = node_features   # (batch, nodes, 300)

        # Pass through stacked GCN layers
        for layer in self.gcn_layers:
            x = layer(x, adj)       # (batch, nodes, hidden_dim)
            x = self.dropout(x)

        # Extract the final embedding at each drug's node position
        batch_size = x.size(0)
        e1_emb = x[torch.arange(batch_size), e1_pos, :]   # (batch, hidden_dim)
        e2_emb = x[torch.arange(batch_size), e2_pos, :]   # (batch, hidden_dim)

        # Concatenate drug representations and classify
        combined = torch.cat([e1_emb, e2_emb], dim=-1)    # (batch, hidden_dim*2)
        logits   = self.classifier(combined)               # (batch, num_labels)

        return logits
