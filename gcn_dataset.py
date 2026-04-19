"""
gcn_dataset.py
──────────────
Loads the DDI 2013 XML corpus and converts each sentence into a
mathematical graph that the GCN can process.

HOW THE GCN APPROACH WORKS (read this carefully):

Instead of reading words in sequence like BioBERT does, the GCN approach
treats each sentence as a GRAPH — specifically, a dependency parse tree.

What is a dependency parse tree?
  Every sentence has grammatical structure. In:
    "Aspirin may inhibit the metabolism of Warfarin."
  spaCy finds:
    inhibit → subject: Aspirin
    inhibit → object: metabolism
    metabolism → prep-of: Warfarin
    may → aux of: inhibit

  This creates a TREE where each word is a NODE and each grammatical
  relationship is an EDGE connecting two nodes.

Why is this useful for DDI?
  The shortest path between the two drug nodes in this tree
  ("Aspirin → inhibit → metabolism → Warfarin") captures exactly the
  interaction phrase, cutting out irrelevant words.

The Pipeline:
  1. Parse sentence with spaCy → dependency tree
  2. Build adjacency matrix (which nodes connect to which)
  3. Node features = 300-dim word vectors (spaCy has GloVe-like vectors)
  4. Also record which node indices correspond to the two drug entities
  5. GCN processes the graph → extract drug node embeddings → classify

CACHING:
  spaCy parsing 27,000 sentences takes ~5-10 minutes.
  We cache parsed results to  gcn_cache.pkl  so subsequent runs are instant.
"""

import os
import glob
import pickle
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# ── Label definitions (identical to BioBERT track) ────────────────────────────
LABEL2ID = {
    "negative":  0,
    "mechanism": 1,
    "effect":    2,
    "advise":    3,
    "int":       4,
}
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# ── Graph parameters ──────────────────────────────────────────────────────────
MAX_NODES  = 100   # sentences longer than 100 tokens are truncated
NODE_DIM   = 300   # spaCy en_core_web_md word vector dimension
CACHE_FILE = "gcn_cache.pkl"


# ── spaCy model loader ────────────────────────────────────────────────────────

def load_spacy():
    """
    Load the spaCy model with 300-dim word vectors.
    Downloads automatically if not already installed.
    """
    import spacy
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        print("Downloading spaCy model en_core_web_md ...")
        spacy.cli.download("en_core_web_md")
        nlp = spacy.load("en_core_web_md")
    return nlp


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(doc):
    """
    Convert a spaCy Doc into a padded adjacency matrix and node feature matrix.

    The dependency tree edges are made BIDIRECTIONAL so information can flow
    in both directions (child→parent and parent→child).
    Self-loops (each node connects to itself) are added so each node retains
    its own features after aggregation.

    Adjacency matrix is row-normalised: A_norm = D^{-1} * A
    This prevents nodes with many connections from dominating.

    Returns:
        adj          : (MAX_NODES, MAX_NODES) float32 — normalised adj matrix
        node_features: (MAX_NODES, NODE_DIM)  float32 — padded word vectors
    """
    n = min(len(doc), MAX_NODES)

    # Raw adjacency (before normalisation)
    adj_raw = np.zeros((n, n), dtype=np.float32)

    for token in doc[:n]:
        i = token.i
        j = min(token.head.i, n - 1)   # parent node (clamped)
        adj_raw[i][j] = 1.0             # child → parent
        adj_raw[j][i] = 1.0             # parent → child (bidirectional)
        adj_raw[i][i] = 1.0             # self-loop

    # Row normalisation: divide each row by its degree
    degree = adj_raw.sum(axis=1, keepdims=True)
    degree[degree == 0] = 1.0           # avoid division by zero
    adj_norm = adj_raw / degree

    # Pad to MAX_NODES × MAX_NODES
    adj_pad = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
    adj_pad[:n, :n] = adj_norm

    # Node features: 300-dim spaCy word vectors (GloVe-like, biomedical domain)
    # Tokens without a vector get a zero vector (padding)
    features = np.zeros((MAX_NODES, NODE_DIM), dtype=np.float32)
    for token in doc[:n]:
        features[token.i] = token.vector   # 300-dim

    return adj_pad, features


def char_offset_to_token_idx(doc, char_start):
    """
    Map a character offset to the index of the corresponding token.
    Falls back to 0 if no token is found.
    """
    for token in doc:
        if token.i >= MAX_NODES:
            break
        # token.idx is the start character of the token in the sentence
        if token.idx >= char_start:
            return min(token.i, MAX_NODES - 1)
    # If char_start is past all tokens, return last valid token
    return min(len(doc) - 1, MAX_NODES - 1)


# ── XML loading + graph building ──────────────────────────────────────────────

def load_xml_files_gcn(folder_path, nlp, cache_path=CACHE_FILE):
    """
    Parse all XML files in folder_path, build dependency graphs, and return
    a flat list of example dicts (one per entity pair per sentence).

    Results are cached to cache_path so spaCy parsing only runs once.

    Each example dict contains:
        adj          : (MAX_NODES, MAX_NODES)  normalised adjacency matrix
        node_features: (MAX_NODES, NODE_DIM)   word vectors
        e1_pos       : int — token index of drug 1
        e2_pos       : int — token index of drug 2
        label        : int (0-4)
        e1_text      : str (for debugging)
        e2_text      : str (for debugging)
    """
    # ── Check cache ────────────────────────────────────────────────────────────
    cache_key = os.path.abspath(folder_path)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if cache_key in cache:
            print(f"  Loaded from cache ({len(cache[cache_key])} examples)")
            return cache[cache_key]
    else:
        cache = {}

    # ── Parse XML files ────────────────────────────────────────────────────────
    xml_files = sorted(glob.glob(
        os.path.join(folder_path, "**", "*.xml"), recursive=True
    ))

    examples = []

    for xml_path in tqdm(xml_files, desc="  Parsing XML + spaCy"):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for sentence in root.iter("sentence"):
            sent_text = sentence.attrib.get("text", "").strip()
            if not sent_text:
                continue

            # Parse with spaCy (builds dependency tree + word vectors)
            doc = nlp(sent_text)

            # Build graph ONCE per sentence (reused for all pairs in it)
            adj, node_features = build_graph(doc)

            # Entity lookup: id → (char_start, char_end, text)
            entities = {}
            for ent in sentence.findall("entity"):
                eid    = ent.attrib["id"]
                offset = ent.attrib["charOffset"].split(";")[0]
                start, end = map(int, offset.split("-"))
                etext  = ent.attrib.get("text", sent_text[start:end + 1])
                entities[eid] = (start, end, etext)

            # One example per pair
            for pair in sentence.findall("pair"):
                e1_id = pair.attrib["e1"]
                e2_id = pair.attrib["e2"]

                if e1_id not in entities or e2_id not in entities:
                    continue

                ddi   = pair.attrib.get("ddi", "false").lower() == "true"
                itype = pair.attrib.get("type", "").lower() if ddi else "negative"
                label = LABEL2ID.get(itype, 0)

                e1_start, _, e1_text = entities[e1_id]
                e2_start, _, e2_text = entities[e2_id]

                # Map character offsets to token indices
                e1_pos = char_offset_to_token_idx(doc, e1_start)
                e2_pos = char_offset_to_token_idx(doc, e2_start)

                examples.append({
                    "adj":           adj,
                    "node_features": node_features,
                    "e1_pos":        e1_pos,
                    "e2_pos":        e2_pos,
                    "label":         label,
                    "e1_text":       e1_text,
                    "e2_text":       e2_text,
                })

    # ── Save cache ─────────────────────────────────────────────────────────────
    cache[cache_key] = examples
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  Cached {len(examples)} examples → {cache_path}")

    return examples


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class GCNDataset(Dataset):
    """
    PyTorch Dataset for the GCN DDI classifier.

    Takes examples from load_xml_files_gcn() and converts numpy arrays
    to PyTorch tensors ready for training.
    """

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "adj":           torch.tensor(ex["adj"],           dtype=torch.float),
            "node_features": torch.tensor(ex["node_features"], dtype=torch.float),
            "e1_pos":        torch.tensor(ex["e1_pos"],        dtype=torch.long),
            "e2_pos":        torch.tensor(ex["e2_pos"],        dtype=torch.long),
            "label":         torch.tensor(ex["label"],         dtype=torch.long),
        }
