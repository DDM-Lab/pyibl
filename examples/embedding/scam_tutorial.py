from pathlib import Path
import csv
import importlib

try:
    sentence_transformers = importlib.import_module("sentence_transformers")
except ImportError as exc:
    raise SystemExit(
        "This tutorial requires sentence-transformers. Install it with: "
        "pip install sentence-transformers"
    ) from exc

SentenceTransformer = sentence_transformers.SentenceTransformer

from pyibl import Agent

# Install dependency once: pip install sentence-transformers
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Build an agent that compares "title" values by embedding similarity.
agent = Agent(attributes=["title", "label"], mismatch_penalty=1)
agent.embedding(function=model)      # Register the embedding model once.
agent.embedding("title")            # Turn embedding-based similarity on for this attribute.

# Load a tiny labeled memory and populate positive/negative label utilities.
with open(Path(__file__).with_name("scam_titles.csv"), newline="") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        other = "safe" if row["label"] == "scam" else "scam"
        agent.populate([{"title": row["title"], "label": row["label"]}], 1)
        agent.populate([{"title": row["title"], "label": other}], 0)

# Predict label by choosing between two label options for the same title.
query = "Urgent: your mailbox storage is full, verify now"
choice = agent.choose([
    {"title": query, "label": "scam"},
    {"title": query, "label": "safe"},
])
print(f"Predicted label: {choice['label']}")
