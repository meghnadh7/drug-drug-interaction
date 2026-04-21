# Drug-Drug Interaction Extraction

NLP course project — Northeastern University

---

I built a system that takes a clinical sentence mentioning two drugs and classifies what kind of interaction exists between them (or whether there is one at all). The idea came from the SemEval-2013 Task 9 shared task on DDI extraction, which has a nicely labeled dataset I used for training and evaluation.

I implemented two different approaches to compare how a pre-trained language model stacks up against a graph-neural-network method on the same task.

---

## The problem

Doctors prescribe multiple drugs at the same time all the time. Some combinations are fine, others can be dangerous — one drug might make the other more toxic, reduce its effectiveness, or cause some unexpected side effect. There are tens of thousands of possible drug pairs so nobody can memorize all of them. The idea is to build a model that can read a sentence from a medical paper and automatically flag what kind of interaction is being described.

There are 5 classes in the dataset:

- **negative** — no interaction (this is ~85% of the data, which made things annoying)
- **mechanism** — one drug changes how the other is absorbed or metabolized
- **effect** — the combined effect changes, usually in a bad way
- **advise** — the text is giving a clinical warning about using these together
- **int** — an interaction is mentioned but they don't say what kind

---

## Two approaches

### Approach 1 — BioBERT

Fine-tuned BioBERT (dmis-lab/biobert-v1.1) on the DDI corpus. The model was pre-trained on PubMed so it already understands biomedical language before I even touch it, which helps a lot. The approach uses entity markers — you wrap the two drug names with special tokens like `[E1] warfarin [/E1]` and `[E2] aspirin [/E2]` before passing the sentence in, then extract the hidden states at those positions and classify from there.

The class imbalance (85% negative) was the main headache. Ended up using Focal Loss with inverse-frequency class weights, which helped quite a bit.

Results on the test set (full fine-tuning on GPU, all 12 layers trained):
- Macro F1 (positive classes only, which is the official metric): **0.75**
- Precision: 0.84 | Recall: 0.72
- Overall accuracy: 94%

Per-class breakdown:

| Class | F1 |
|---|---|
| mechanism | 0.84 |
| advise | 0.83 |
| effect | 0.79 |
| int | 0.54 |

The `int` class scores lower because it's the rarest and most ambiguous — it means "interaction mentioned without specifying type", which often looks identical to an effect sentence. Every published paper on this dataset has the same problem with it.

### Approach 2 — ChemBERTa-R-GAT

Instead of reading the sentence as plain text, this approach builds a graph from the sentence's grammatical structure (dependency parse tree) and runs a graph neural network over it. The key insight is that the **Shortest Dependency Path (SDP)** between the two drug tokens contains most of the interaction signal — words like "potentiates", "contraindicated", "synergistic" sit directly on that path and tell you the interaction type.

**How it works:**

- **Drug nodes** get ChemBERTa embeddings (a BERT model pre-trained on 77 million SMILES strings). This encodes the drug's actual molecular structure, not just its name as text.
- **Context nodes** (the words on the SDP between the two drugs) get frozen BioBERT embeddings. BioBERT was pre-trained on PubMed so it actually understands biomedical verbs and clinical terms.
- **Edges** carry dependency relation types (nsubj, dobj, prep, etc.) which get fed into the attention mechanism.
- **R-GAT** (Relational Graph Attention Network) runs 4 layers with 8 attention heads. The relational part means different grammatical connections get different attention weights — a subject relation and a prepositional relation mean different things.
- **Two-stage classifier:** a binary head first decides "is there any interaction?", then a type head decides "which type?". This stops the dominant negative class from contaminating the type decision.

This approach went through several iterations — the first version used plain GloVe word vectors for context nodes and scored 0.37. Replacing those with frozen BioBERT jumped it to 0.55 in one change, which confirmed that feature quality on the context nodes (the interaction verbs) was the main bottleneck.

Results on the test set:
- Macro F1 (positive classes only): **0.56**
- Precision: 0.54 | Recall: 0.54

| Class | F1 |
|---|---|
| advise | 0.63 |
| mechanism | 0.56 |
| effect | 0.52 |
| int | 0.42 |

Trains in about 30-40 min on CPU (much lighter than BioBERT). The main limitation is SMILES coverage — about 30% of drugs in the corpus couldn't be looked up in PubChem and get zero vectors.

---

## Dataset

SemEval-2013 Task 9 — DDI Extraction corpus. About 1,017 documents from DrugBank and MedLine, ~25,000 sentence pairs. You need to download this separately (it's not in the repo because of licensing).

Expected folder structure:
```
~/Desktop/DDICorpus/
    Train/
    Test/
        Test for DDI Extraction task/
```

---

## Setup

```bash
git clone https://github.com/meghnadh7/drug-drug-interaction.git
cd drug-drug-interaction

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# for the ChemBERTa-R-GAT approach you also need:
python -m spacy download en_core_web_md
```

---

## Running the models

**BioBERT training:**
```bash
python main.py
```
Runs for 3 epochs. Set `demo_mode=True` in `main.py` if you just want to verify the pipeline quickly (~5 min).

**ChemBERTa-R-GAT training:**
```bash
python gcn_main.py
```
First run parses all sentences with spaCy and caches them (takes ~10 min). After that it loads from cache and starts immediately.

---

## UI

There's a web interface you can use to test predictions on your own sentences.

```bash
# terminal 1 — start the model API
source venv/bin/activate
python api.py

# terminal 2 — start the frontend
cd ui
npm install
npm run dev
```

Go to http://localhost:3000. Type a sentence with two drug names, enter the drug names in the fields, hit Analyse. There are also example sentences you can click to try out each interaction type.

---

## File overview

```
main.py          — BioBERT training entry point
gcn_main.py      — ChemBERTa-R-GAT training entry point
ddi_dataset.py   — XML parsing + tokenization for BioBERT
gcn_dataset.py   — spaCy SDP parsing + graph construction for R-GAT
model.py         — DDIClassifier (BioBERT + entity-span head)
gcn_model.py     — ChemBERTa-R-GAT model (R-GAT + two-stage classifier)
train.py         — training loop for BioBERT (FocalLoss, warmup scheduler)
gcn_train.py     — training loop for R-GAT (dual FocalLoss, ReduceLROnPlateau)
evaluate.py      — test set evaluation (F1, confusion matrix)
gcn_evaluate.py  — same for R-GAT
api.py           — Flask API that wraps the trained BioBERT model
ui/              — React frontend
```

---

## Notes / known issues

BioBERT training on CPU is slow — use Colab with a T4 GPU, it takes 20-25 min there vs several hours on CPU.

The ChemBERTa-R-GAT approach has a SMILES coverage gap — about 30% of drugs in the corpus couldn't be found in PubChem and get zero vectors instead of ChemBERTa embeddings. A more complete drug-to-SMILES lookup would likely improve results.

The `int` class is the hardest for both approaches — it only has ~2% of the training data and the sentences look very similar to `effect` sentences. Both models struggle with it.

The `negative` class F1 is high but that's just because it's 85% of the data. The metric that matters for this task is macro-F1 over the 4 positive classes only, which is what all the numbers above reflect.
