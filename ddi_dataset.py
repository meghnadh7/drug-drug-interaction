"""
ddi_dataset.py
──────────────
Loads the DDI 2013 Corpus from local XML files and converts it into
PyTorch tensors that BioBERT can consume.

WHY LOCAL XML (not HuggingFace):
  The newer datasets library (v4+) dropped support for dataset loading
  scripts, and bigbio/ddi_corpus still uses one. Loading directly from
  the local XML files we already have is more reliable and has zero
  external dependencies beyond the standard library's xml.etree module.

HOW IT WORKS:

The DDI task is RELATION EXTRACTION:
  - Input  : a sentence + two drug mentions inside it
  - Output : one of 5 labels (negative / mechanism / effect / advise / int)

The XML already contains explicit <pair> elements with their labels,
so we don't need to enumerate all pairs ourselves — we just read them.

ENTITY MARKER TECHNIQUE:
  Instead of feeding the plain sentence to BioBERT, we wrap the two drug
  entities in special tokens so the model knows which drugs to focus on.

  Original  : "Aspirin may interact with Warfarin."
  Marked    : "[E1] Aspirin [/E1] may interact with [E2] Warfarin [/E2] ."

  After BioBERT runs, we pull out the hidden states at the [E1] and [E2]
  token positions, concatenate them (768 + 768 = 1536-dim) → classify.
"""

import os
import glob
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# ── Label definitions ─────────────────────────────────────────────────────────
# XML uses: mechanism, effect, advise, int  (note: "advise" not "advice")
# "negative" is our label when ddi="false"
LABEL2ID = {
    "negative":  0,
    "mechanism": 1,
    "effect":    2,
    "advise":    3,
    "int":       4,
}
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


# ── XML parsing ───────────────────────────────────────────────────────────────

def _parse_char_offset(offset_str):
    """
    Parse a charOffset string like "45-52" into (start=45, end=52).
    The end index is INCLUSIVE in the DDI XML format.
    """
    parts = offset_str.split(";")[0]   # some offsets have multiple spans; use first
    start, end = parts.split("-")
    return int(start), int(end)


def load_xml_files(folder_path):
    """
    Parse all XML files in a folder (and subfolders) and return a flat
    list of example dicts, one per entity pair per sentence.

    Each dict:
        text    : sentence with [E1]/[E2] markers inserted
        label   : integer 0-4
        e1_text : raw drug 1 text (for debugging)
        e2_text : raw drug 2 text (for debugging)
    """
    examples = []

    xml_files = glob.glob(os.path.join(folder_path, "**", "*.xml"), recursive=True)
    xml_files.sort()

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for sentence in root.iter("sentence"):
            sent_text = sentence.attrib.get("text", "")

            # Build entity lookup: id → (start, end, text)
            entities = {}
            for ent in sentence.findall("entity"):
                eid    = ent.attrib["id"]
                start, end = _parse_char_offset(ent.attrib["charOffset"])
                etext  = ent.attrib.get("text", sent_text[start:end + 1])
                entities[eid] = (start, end, etext)

            # Each <pair> is one training example
            for pair in sentence.findall("pair"):
                e1_id = pair.attrib["e1"]
                e2_id = pair.attrib["e2"]
                ddi   = pair.attrib.get("ddi", "false").lower() == "true"
                itype = pair.attrib.get("type", "").lower() if ddi else "negative"

                # Skip pairs whose entities are missing from the lookup
                if e1_id not in entities or e2_id not in entities:
                    continue

                # Map to our label set
                label = LABEL2ID.get(itype, 0)

                e1_start, e1_end, e1_text = entities[e1_id]
                e2_start, e2_end, e2_text = entities[e2_id]

                marked = _mark_entities(
                    sent_text,
                    e1_start, e1_end,
                    e2_start, e2_end,
                )

                examples.append({
                    "text":    marked,
                    "label":   label,
                    "e1_text": e1_text,
                    "e2_text": e2_text,
                })

    return examples


def _mark_entities(text, e1_start, e1_end, e2_start, e2_end):
    """
    Insert [E1]/[/E1] and [E2]/[/E2] markers around two drug spans.

    We insert right-to-left so earlier insertions don't shift the offsets
    of later ones.

    Example:
      "Aspirin may interact with Warfarin."
      → "[E1] Aspirin [/E1] may interact with [E2] Warfarin [/E2] ."
    """
    if e1_start < e2_start:
        # e1 comes first — insert from rightmost to leftmost
        insertions = [
            (e2_end + 1, " [/E2]"),
            (e2_start,   "[E2] "),
            (e1_end + 1, " [/E1]"),
            (e1_start,   "[E1] "),
        ]
    else:
        # e2 comes first
        insertions = [
            (e1_end + 1, " [/E1]"),
            (e1_start,   "[E1] "),
            (e2_end + 1, " [/E2]"),
            (e2_start,   "[E2] "),
        ]

    for pos, marker in insertions:
        text = text[:pos] + marker + text[pos:]

    return text


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class DDIDataset(Dataset):
    """
    PyTorch Dataset for DDI classification.

    Args:
        examples   : list of dicts from load_xml_files()
        tokenizer  : HuggingFace tokenizer (BioBERT)
        max_length : max token sequence length (BERT hard limit = 512)
    """

    def __init__(self, examples, tokenizer, max_length=256):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        encoding = self.tokenizer(
            ex["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].squeeze(0)       # (max_length,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (max_length,)

        # Find token positions of [E1] and [E2] markers
        e1_token_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        e2_token_id = self.tokenizer.convert_tokens_to_ids("[E2]")

        e1_pos = (input_ids == e1_token_id).nonzero(as_tuple=True)[0]
        e2_pos = (input_ids == e2_token_id).nonzero(as_tuple=True)[0]

        # Fallback to [CLS] position if marker was truncated away
        e1_pos = e1_pos[0].item() if len(e1_pos) > 0 else 0
        e2_pos = e2_pos[0].item() if len(e2_pos) > 0 else 0

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "e1_pos":         torch.tensor(e1_pos,      dtype=torch.long),
            "e2_pos":         torch.tensor(e2_pos,      dtype=torch.long),
            "label":          torch.tensor(ex["label"], dtype=torch.long),
        }


def get_tokenizer(model_name="dmis-lab/biobert-v1.1"):
    """
    Load the BioBERT tokenizer and register our four entity marker tokens.
    These do not exist in BioBERT's vocabulary by default, so we add them.
    The model embedding table must be resized to match (done in model.py).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    )
    return tokenizer
