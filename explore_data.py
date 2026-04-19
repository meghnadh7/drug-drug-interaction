"""
explore_data.py
───────────────
Run this FIRST to understand the dataset before training.
Parses the local XML corpus and prints structure, a sample, and label counts.

Run with:
    python explore_data.py
"""

import os
import glob
import xml.etree.ElementTree as ET
from collections import Counter

CORPUS_ROOT = os.path.expanduser("~/Desktop/DDICorpus")
TRAIN_DIR   = os.path.join(CORPUS_ROOT, "Train")
TEST_DIR    = os.path.join(CORPUS_ROOT, "Test", "Test for DDI Extraction task")

# ── 1. Count files ────────────────────────────────────────────────────────────
train_files = glob.glob(os.path.join(TRAIN_DIR, "**", "*.xml"), recursive=True)
test_files  = glob.glob(os.path.join(TEST_DIR,  "**", "*.xml"), recursive=True)
print(f"Train XML files : {len(train_files)}")
print(f"Test  XML files : {len(test_files)}")

# ── 2. Print one raw sentence with its pairs ──────────────────────────────────
print("\n" + "="*65)
print("SAMPLE SENTENCE WITH A POSITIVE INTERACTION")
print("="*65)

for xml_path in sorted(train_files):
    tree = ET.parse(xml_path)
    for sentence in tree.getroot().iter("sentence"):
        pairs = sentence.findall("pair")
        if any(p.attrib.get("ddi", "false") == "true" for p in pairs):
            print(f"\nFile    : {os.path.basename(xml_path)}")
            print(f"Text    : {sentence.attrib['text']}")
            print(f"Entities:")
            for e in sentence.findall("entity"):
                print(f"  id={e.attrib['id']}  offset={e.attrib['charOffset']}  text={e.attrib['text']}")
            print(f"Pairs:")
            for p in pairs:
                print(f"  e1={p.attrib['e1']}  e2={p.attrib['e2']}  ddi={p.attrib['ddi']}  type={p.attrib.get('type','—')}")
            break
    else:
        continue
    break

# ── 3. Count label distribution in train set ─────────────────────────────────
print("\n" + "="*65)
print("LABEL DISTRIBUTION — TRAIN")
print("="*65)

train_counter = Counter()
for xml_path in train_files:
    tree = ET.parse(xml_path)
    for sentence in tree.getroot().iter("sentence"):
        for pair in sentence.findall("pair"):
            ddi   = pair.attrib.get("ddi", "false").lower() == "true"
            itype = pair.attrib.get("type", "").lower() if ddi else "negative"
            train_counter[itype if itype else "negative"] += 1

total = sum(train_counter.values())
for label, count in train_counter.most_common():
    print(f"  {label:12s}: {count:6d}  ({100*count/total:.1f}%)")
print(f"  {'TOTAL':12s}: {total:6d}")

print("\n" + "="*65)
print("LABEL DISTRIBUTION — TEST")
print("="*65)

test_counter = Counter()
for xml_path in test_files:
    tree = ET.parse(xml_path)
    for sentence in tree.getroot().iter("sentence"):
        for pair in sentence.findall("pair"):
            ddi   = pair.attrib.get("ddi", "false").lower() == "true"
            itype = pair.attrib.get("type", "").lower() if ddi else "negative"
            test_counter[itype if itype else "negative"] += 1

for label, count in test_counter.most_common():
    print(f"  {label:12s}: {count:6d}")

print("\nDone. Run  python main.py  to train the model.")
