# DDI Dual-Architecture Project — Complete Handoff Document
> Last updated: April 17, 2026
> Read this entire file before touching any code.
> This is the single source of truth for where this project stands and what is left to do.

---

## 1. WHAT THIS PROJECT IS

### The Big Picture
This is an **NLP (Natural Language Processing) machine learning project** for a university course.
The goal is to read medical sentences and automatically detect whether two drugs mentioned in that sentence have an interaction — and if so, what *type* of interaction.

### Real Example
Given the sentence:
> *"Grepafloxacin may inhibit the metabolism of theobromine."*

The model must identify:
- Drug 1: **Grepafloxacin**
- Drug 2: **theobromine**
- Interaction type: **mechanism** (one drug affects how the other is processed)

### Why It Matters
Doctors and pharmacists need to know which drug combinations are dangerous. There are thousands of drugs — manually tracking every pair is impossible. An NLP model that reads drug interaction texts and classifies them automatically can power drug safety alerts, EHR systems, and clinical decision support tools.

---

## 2. THE DATASET

**Name:** DDI Extraction 2013 Corpus (SemEval-2013 Task 9)
**Source:** DrugBank database + MedLine biomedical abstracts

**Location on disk:** `~/Desktop/DDICorpus/`
```
DDICorpus/
├── Train/
│   ├── DrugBank/   ← 572 XML files
│   └── MedLine/    ← 142 XML files
└── Test/
    └── Test for DDI Extraction task/
        ├── DrugBank/  ← 158 XML files
        └── MedLine/   ← 33 XML files
```

**Format:** XML files. Each file contains sentences. Each sentence contains drug entity tags and pair tags.
```xml
<sentence text="Aspirin may interact with Warfarin.">
    <entity id="s1.e1" charOffset="0-6"   type="drug" text="Aspirin"/>
    <entity id="s1.e2" charOffset="25-32" type="drug" text="Warfarin"/>
    <pair e1="s1.e1" e2="s1.e2" ddi="true" type="effect"/>
</sentence>
```

**The 5 Labels (what we're classifying):**

| Label | Meaning | Example |
|-------|---------|---------|
| `negative` | Two drugs in same sentence but NO interaction | Most pairs — ~85% of data |
| `mechanism` | One drug changes how the other is metabolised | "inhibit the metabolism of" |
| `effect` | Combined pharmacological effect | "potentiate the action of" |
| `advise` | Clinical recommendation / warning | "should not be used with" |
| `int` | Interaction mentioned but no detail given | "interacts with" |

**Class Imbalance (critical!):**
```
negative  : 23,772  (85.5%) ← hugely overrepresented
effect    :  1,687  ( 6.1%)
mechanism :  1,319  ( 4.7%)
advise    :    826  ( 3.0%)
int       :    188  ( 0.7%) ← very rare
Total     : 27,792 training pairs
Test      :  5,716 pairs
```
This is why we use **Focal Loss** instead of regular cross-entropy — to stop the model from just predicting "negative" for everything.

---

## 3. THE DUAL ARCHITECTURE

This project has **two completely independent tracks** that solve the same problem differently.
Both are fully implemented and tested in this repo.

### Track 1 — BioBERT (Semantic / Transformer)
Reads the sentence like a human — understanding the *meaning* of words in context.
BioBERT was pre-trained on 18 billion words of biomedical text (PubMed, PMC).

**Entity Marker Technique:**
```
Original : "Aspirin may interact with Warfarin."
Marked   : "[E1] Aspirin [/E1] may interact with [E2] Warfarin [/E2] ."
```
BioBERT encodes the whole sentence. We pull out the hidden states at `[E1]` and `[E2]`
(768-dim each), concatenate → 1536-dim → dropout → Linear → 5 classes.

Model: `dmis-lab/biobert-v1.1` — 108 million parameters.
Bottom 10 of 12 layers frozen on CPU → 14M trainable params.

### Track 2 — GCN (Syntactic / Graph)
Ignores word meaning entirely. Instead, it converts the sentence into a **dependency parse tree**
where each word is a node and grammatical relationships are edges.

```
"Aspirin  →(nsubj)→  inhibit  →(dobj)→  metabolism  →(prep_of)→  Warfarin"
```
This tree becomes an adjacency matrix fed into 3 stacked GCN layers.
We extract the embeddings at the two drug nodes → concatenate → MLP → 5 classes.

Uses: **spaCy `en_core_web_md`** (300-dim word vectors + dependency parsing).
Model: 342,021 parameters — 316× smaller than BioBERT, 100× faster per batch on CPU.

### The Key Comparison
| Property | BioBERT | GCN |
|---|---|---|
| Parameters | 108M | 342K |
| Approach | Semantic (meaning) | Syntactic (grammar) |
| Speed on CPU | ~7s/batch | ~0.04s/batch |
| Pre-trained weights | Yes (biomedical text) | No (trains from scratch) |
| Needs GPU for full training | Yes (~10 min on T4) | No (~30 min on CPU) |

---

## 4. COMPLETE FILE STRUCTURE

```
ddi nlp project/
│
├── HANDOFF.md              ← you are reading this
├── requirements.txt        ← pip dependencies
├── explore_data.py         ← one-time data inspection script
│
├── ── TRACK 1: BioBERT ──────────────────────────────
├── ddi_dataset.py          ← XML loader + entity marker tokenization
├── model.py                ← BioBERT + classification head
├── train.py                ← training loop + FocalLoss class
├── evaluate.py             ← test evaluation + metrics report
├── main.py                 ← BioBERT entry point
├── biobert_ddi_best.pt     ← saved BioBERT weights (demo run)
│
├── ── TRACK 2: GCN ──────────────────────────────────
├── gcn_dataset.py          ← XML loader + spaCy graph builder + cache
├── gcn_model.py            ← GCNLayer + DDIGCN classifier
├── gcn_train.py            ← GCN training loop (reuses FocalLoss from train.py)
├── gcn_evaluate.py         ← GCN test evaluation + metrics report
├── gcn_main.py             ← GCN entry point
├── gcn_ddi_best.pt         ← saved GCN weights (demo run)
├── gcn_cache.pkl           ← spaCy parse cache (27,792 train + 5,716 test graphs)
│
└── venv/                   ← Python 3.13 arm64 virtual environment
```

---

## 5. FILE-BY-FILE REFERENCE

### `ddi_dataset.py` — BioBERT Data Pipeline
- `load_xml_files(folder_path)` → list of `{text, label, e1_text, e2_text}` dicts
  - Parses XML, wraps drugs in `[E1]`/`[E2]` markers
- `DDIDataset(examples, tokenizer, max_length=128)` → PyTorch Dataset
  - Returns `{input_ids, attention_mask, e1_pos, e2_pos, label}`
- `get_tokenizer(model_name)` → adds `[E1]`, `[/E1]`, `[E2]`, `[/E2]` to vocab
- `LABEL2ID = {"negative":0, "mechanism":1, "effect":2, "advise":3, "int":4}`

### `model.py` — BioBERT Model
- `DDIClassifier(model_name, num_labels=5, dropout=0.1, vocab_size, freeze_layers=10)`
- `forward(input_ids, attention_mask, e1_pos, e2_pos)` → logits `(batch, 5)`
- Freezes bottom N encoder layers to speed up CPU training

### `train.py` — BioBERT Training + Focal Loss
- `FocalLoss(alpha, gamma=2.0)` — shared by both tracks
- `compute_class_weights(examples)` — shared by both tracks
- `train_model(model, train_ds, val_ds, config)` → trained model
  - AdamW with two LR groups (bert=2e-5, head=1e-4)
  - Linear warmup + linear decay, gradient clip at 1.0

### `evaluate.py` — BioBERT Evaluation
- `evaluate(model, test_dataset, config)` → `{macro_f1, precision, recall}`
- Official DDI metric: macro-F1 over positive classes only (excludes negative)

### `main.py` — BioBERT Entry Point
```python
CONFIG = {
    "model_name":      "dmis-lab/biobert-v1.1",
    "epochs":          3,
    "batch_size":      32,
    "bert_lr":         2e-5,
    "head_lr":         1e-4,
    "freeze_layers":   10,      # 0 = full fine-tune (GPU only)
    "max_length":      128,
    "val_frac":        0.1,
    "demo_mode":       True,    # ← CHANGE TO False FOR FULL TRAINING
    "demo_train_size": 1500,
    "demo_test_size":  300,
    "device":          "cuda" if cuda else "cpu",
    "seed":            42,
}
```

---

### `gcn_dataset.py` — GCN Data Pipeline
- `load_spacy()` → loads `en_core_web_md`, auto-downloads if missing
- `build_graph(doc)` → `(adj: MAX_NODESxMAX_NODES, features: MAX_NODESx300)`
  - Bidirectional edges + self-loops, row-normalised adjacency
  - Node features = 300-dim spaCy word vectors
- `load_xml_files_gcn(folder_path, nlp, cache_path)` → list of graph examples
  - Caches all parsed results to `gcn_cache.pkl` — only parses once
- `GCNDataset(examples)` → PyTorch Dataset
  - Returns `{adj, node_features, e1_pos, e2_pos, label}`
- Constants: `MAX_NODES=100`, `NODE_DIM=300`

### `gcn_model.py` — GCN Model
- `GCNLayer(in_features, out_features)` — single graph conv layer with BatchNorm
  - `forward(x, adj)`: aggregation = `adj @ x`, then linear + BN + ReLU
- `DDIGCN(input_dim=300, hidden_dim=256, num_layers=3, dropout=0.5)`
  - `forward(adj, node_features, e1_pos, e2_pos)` → logits `(batch, 5)`
  - Extracts drug node embeddings → concat → MLP head

### `gcn_train.py` — GCN Training
- `train_gcn(model, train_ds, val_ds, config)` → trained model
  - AdamW, lr=1e-3 (higher than BioBERT — no pre-trained weights)
  - ReduceLROnPlateau scheduler (halves LR after 3 epochs of no improvement)
  - Imports `FocalLoss`, `compute_class_weights` directly from `train.py`

### `gcn_evaluate.py` — GCN Evaluation
- `evaluate_gcn(model, test_dataset, config)` → `{macro_f1, precision, recall}`
- Identical output format to `evaluate.py` for direct comparison

### `gcn_main.py` — GCN Entry Point
```python
GCN_CONFIG = {
    "input_dim":       300,
    "hidden_dim":      256,
    "num_layers":      3,
    "dropout":         0.5,
    "epochs":          10,
    "batch_size":      64,
    "lr":              1e-3,
    "weight_decay":    5e-4,
    "val_frac":        0.1,
    "demo_mode":       True,    # ← CHANGE TO False FOR FULL TRAINING
    "demo_train_size": 1500,
    "demo_test_size":  300,
    "cache_file":      "gcn_cache.pkl",
    "device":          "cuda" if cuda else "cpu",
    "seed":            42,
}
```

---

## 6. WHAT HAS BEEN COMPLETED ✅

### Track 1 — BioBERT
| Task | Status | Notes |
|------|--------|-------|
| Dataset verified | ✅ | Local XML at `~/Desktop/DDICorpus/` confirmed authentic |
| XML data loader | ✅ | `load_xml_files()` in `ddi_dataset.py` |
| Entity marker tokenization | ✅ | `[E1]`/`[E2]` wrapping + token position finding |
| BioBERT model with layer freezing | ✅ | `model.py` |
| Focal Loss + class weights | ✅ | `train.py` |
| Full training loop | ✅ | LR warmup, grad clip, best checkpoint |
| Evaluation report | ✅ | Per-class F1/P/R + confusion matrix |
| Demo run — exit code 0 | ✅ | 1500 examples, 3 epochs |
| Model weights saved | ✅ | `biobert_ddi_best.pt` |

**BioBERT FULL training — IN PROGRESS (running in background as of April 18)**
```
Config  : demo_mode=False, freeze_layers=10, epochs=3, batch_size=32
ETA     : ~4 hours on CPU
Log     : biobert_training_log.txt  (in project folder)
```
Check progress: `tail -5 "/Users/meghnadh1/Desktop/ddi nlp project /biobert_training_log.txt"`
Expected final macro-F1: ~0.65–0.75

### Track 2 — GCN
| Task | Status | Notes |
|------|--------|-------|
| spaCy installed | ✅ | `en_core_web_md` in venv |
| XML loader + graph builder | ✅ | `gcn_dataset.py` |
| Parse cache built | ✅ | `gcn_cache.pkl` — 27,792 + 5,716 graphs |
| GCN model (3 layers) | ✅ | `gcn_model.py` |
| GCN training loop | ✅ | `gcn_train.py` |
| GCN evaluation | ✅ | `gcn_evaluate.py` |
| Demo run — exit code 0 | ✅ | 1500 examples, 10 epochs, ~15 seconds |
| Model weights saved | ✅ | `gcn_ddi_best.pt` |

**GCN FULL training results (27,792 examples — 100% of data):**
```
Epoch  1: loss=0.8202,  val_F1=0.3838  ← new best
Epoch  2: loss=0.6146,  val_F1=0.3290
Epoch  3: loss=0.5016,  val_F1=0.4172  ← new best
Epoch  4: loss=0.4595,  val_F1=0.3795
Epoch  5: loss=0.4247,  val_F1=0.3729
Epoch  6: loss=0.4101,  val_F1=0.4139
Epoch  7: loss=0.3688,  val_F1=0.4289  ← new best
Epoch  8: loss=0.3843,  val_F1=0.4320  ← best checkpoint saved
Epoch  9: loss=0.3703,  val_F1=0.4118
Epoch 10: loss=0.4496,  val_F1=0.3121

Test macro-F1 (positive classes only): 0.3116
Per-class breakdown:
  mechanism  P=0.19  R=0.67  F1=0.29
  effect     P=0.19  R=0.59  F1=0.29
  advise     P=0.21  R=0.73  F1=0.32
  int        P=0.26  R=0.51  F1=0.34
```
✅ Full training complete. Weights saved → `gcn_ddi_best.pt`

---

## 7. WHAT STILL NEEDS TO BE DONE ❌

### 7a. Wait for BioBERT Full Training to Finish ⏳
**Status:** Running in background since April 18
**Check log:** `tail -20 "/Users/meghnadh1/Desktop/ddi nlp project /biobert_training_log.txt"`
**Expected completion:** ~4 hours from start
**Expected macro-F1:** ~0.65–0.75 (with freeze_layers=10)

**Optional — for best possible BioBERT score on Colab:**
```python
# In main.py set:
"freeze_layers": 0   # unfreeze all 12 layers
# Then run on Colab T4 GPU (~10 min) → expected macro-F1 ~0.75–0.80
```

### 7b. GCN Full Training ✅ ALREADY DONE
Results: macro-F1 = **0.3116** on full test set
Weights saved: `gcn_ddi_best.pt`

### 7c. Presentation Analysis (April 18 deadline)
Once both full-training runs are done, compare the two models:

| Question to answer | How |
|---|---|
| Which model has higher overall macro-F1? | Compare headline numbers |
| Which model is better at `mechanism`? | Per-class F1 tables |
| Which model is better at `advise`? | Per-class F1 tables |
| Where does BioBERT fail that GCN catches? | Compare confusion matrices |
| Where does GCN fail that BioBERT catches? | Compare confusion matrices |
| What's the speed/accuracy trade-off? | 342K params vs 108M params |

### 7d. Stretch Goal — Ensemble (if time permits)
Late-fusion ensemble: take both trained models, concatenate their intermediate
representations before a final shared classifier.
- BioBERT output: 1536-dim vector (before its linear layer)
- GCN output: 512-dim vector (before its MLP head)
- Concatenate → 2048-dim → Linear → 5 classes

---

## 8. HOW TO RUN

### Always activate the virtual environment first
```bash
cd "/Users/meghnadh1/Desktop/ddi nlp project "
source venv/bin/activate
```

### BioBERT — demo mode (proves pipeline, ~25 min on CPU)
```bash
# demo_mode=True in main.py (default)
python3 main.py
```

### BioBERT — full training (Colab GPU recommended)
```bash
# Set demo_mode=False, freeze_layers=0 in main.py
python3 main.py
```

### GCN — demo mode (proves pipeline, ~2 min on CPU)
```bash
# demo_mode=True in gcn_main.py (default)
python3 gcn_main.py
```

### GCN — full training (CPU is fine, ~30 min)
```bash
# Set demo_mode=False in gcn_main.py
python3 gcn_main.py
# gcn_cache.pkl already exists — no re-parsing needed
```

---

## 9. KNOWN ISSUES & DECISIONS MADE

| Issue | Decision | Reason |
|-------|----------|--------|
| HuggingFace `bigbio/ddi_corpus` fails to load | Load from local XML | `datasets` v4+ dropped loading script support |
| MPS (Apple Silicon GPU) hangs at batch 2 | Force CPU (`device="cpu"`) | Known PyTorch 2.x MPS bug with large transformer backward pass |
| 108M BioBERT params too slow on CPU | Freeze bottom 10 layers | Reduces trainable params 108M→14M; ~5x faster |
| 85% negative class imbalance | Focal Loss with class weights | Prevents model predicting "negative" for everything |
| No official validation split | 90/10 random split of train | Standard practice for this dataset |
| `torch_geometric` not used for GCN | Manual dense adjacency matrix | Simpler install, equivalent for sentence-scale graphs |
| `ReduceLROnPlateau(verbose=True)` crashes | Removed `verbose` arg | PyTorch 2.11 dropped that parameter |

---

## 10. PROJECT DEADLINE & MILESTONES

| Date | Milestone | Status |
|------|-----------|--------|
| April 6  | Forward pass working, no tensor errors | ✅ Done |
| April 13 | Both models trained, F1 metrics generated | ✅ Demo runs done; full training on Colab pending |
| April 17 | Both tracks fully coded and demo-verified | ✅ Done |
| April 18 | Full GCN training done; BioBERT full training running | ⏳ BioBERT in background |
| April 21 | Final presentation | — |

---

## 11. ENVIRONMENT

```
Python        : 3.13.2 (arm64, Homebrew)
PyTorch       : 2.11.0
Transformers  : 5.5.3
spaCy         : 3.8.14  +  en_core_web_md
scikit-learn  : 1.8.0
datasets      : 4.8.4
Virtual env   : ./venv/  (activate with: source venv/bin/activate)
```

---

## 12. TEAMMATE CONTEXT

- **This repo:** Both tracks are implemented here
- **Original plan:** Rochan was supposed to do Track 2 (GCN) separately
- **Current state:** Both tracks are now in this repo, fully working
- **For presentation:** You can present both tracks and the comparison yourself,
  or share the GCN files (`gcn_*.py`) with Rochan to run on his machine
