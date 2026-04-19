# Drug-Drug Interaction Extraction

NLP course project — Northeastern University

---

I built a system that takes a clinical sentence mentioning two drugs and classifies what kind of interaction exists between them (or whether there is one at all). The idea came from the SemEval-2013 Task 9 shared task on DDI extraction, which has a nicely labeled dataset I used for training and evaluation.

I implemented two different approaches to compare how a pre-trained language model stacks up against a graph-based method on the same task.

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

Results on the test set:
- Macro F1 (positive classes only, which is the official metric): **0.63**
- Precision: 0.66 | Recall: 0.61

This is with `freeze_layers=10` (only training the top 2 BERT layers) because training the full 108M parameter model on CPU takes forever. Full fine-tuning on a GPU should push this closer to 0.75-0.80.

### Approach 2 — GCN

Builds a dependency parse tree for each sentence using spaCy, then runs a Graph Convolutional Network over it. The idea is that in a sentence like "rifampin induces CYP3A4 and thereby decreases clarithromycin levels", the shortest dependency path between the two drugs carries most of the interaction signal.

Trained from scratch (no pre-trained weights), so needed more epochs and a higher learning rate than BioBERT.

Results:
- Macro F1: **0.45**
- Much faster to train though (~20 min vs hours for BioBERT)

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

# for the GCN approach you also need:
python -m spacy download en_core_web_md
```

---

## Running the models

**BioBERT training:**
```bash
python main.py
```
Runs for 3 epochs. Set `demo_mode=True` in `main.py` if you just want to verify the pipeline quickly (~5 min).

**GCN training:**
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
gcn_main.py      — GCN training entry point
ddi_dataset.py   — XML parsing + tokenization for BioBERT
gcn_dataset.py   — spaCy parsing + graph construction for GCN
model.py         — DDIClassifier (BioBERT + classification head)
gcn_model.py     — DDIGCN (graph convolutional network)
train.py         — training loop for BioBERT (FocalLoss, etc.)
gcn_train.py     — training loop for GCN
evaluate.py      — test set evaluation (F1, confusion matrix)
gcn_evaluate.py  — same for GCN
api.py           — Flask API that wraps the trained model
ui/              — React frontend
```

---

## Notes / known issues

Training on CPU is slow. The BioBERT training took a few hours even with layer freezing. If you're replicating this, use Colab with a GPU — it's much more reasonable.

The GCN uses the full dependency tree rather than just the shortest path between the two drugs. The shortest path approach from the original DGCNN paper would probably work better but I ran out of time to implement it properly.

The `negative` class F1 is high but that's just because it dominates the dataset. The metric I care about for this task is macro-F1 over the 4 positive classes only, which is what the numbers above reflect.
