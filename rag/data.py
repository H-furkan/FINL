"""Data loading, drug knowledge graph, and reverse indexes."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "drug_docs.csv"

SIDE_EFFECT_KEYWORDS = [
    "side effect",
    "may cause",
    "may experience",
    "include headache",
    "include nausea",
    "include drowsiness",
    "include fatigue",
    "include dizziness",
    "include stomach",
    "include joint",
    "include bleeding",
    "include redness",
    "include constipation",
    "include insomnia",
    "include chest",
    "include cough",
    "include weight",
    "cause allergic",
    "cause muscle",
    "cause bleeding",
    "cause mild",
    "cause next-day",
    "cause temporary",
    "cause mood",
    "cause infections",
    "cause dry",
]

_DRUG_NAME_RE = re.compile(r"\bDrug ([A-Z]{1,2})\b")


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load drug_docs.csv and return a clean DataFrame."""
    path = Path(path) if path else DATA_PATH
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def _extract_drug_name(text: str) -> str | None:
    """Extract the first Drug name (e.g. 'Drug A', 'Drug AX') from text."""
    match = _DRUG_NAME_RE.search(text)
    return f"Drug {match.group(1)}" if match else None


def _classify_doc_type(text: str) -> str:
    """Classify a document as 'side_effect' or 'usage'."""
    text_lower = text.lower()
    for kw in SIDE_EFFECT_KEYWORDS:
        if kw in text_lower:
            return "side_effect"
    return "usage"


def _extract_conditions(text: str) -> list[str]:
    """Extract medical conditions/purposes from a usage doc."""
    conditions = []
    text_lower = text.lower()

    condition_map = {
        "hypertension": ["hypertension"],
        "high blood pressure": ["hypertension"],
        "diabetes": ["diabetes"],
        "infection": ["infections"],
        "bacterial infection": ["infections"],
        "viral infection": ["infections"],
        "respiratory infection": ["infections"],
        "eye infection": ["infections"],
        "skin infection": ["infections"],
        "sinus infection": ["infections"],
        "fungal infection": ["infections"],
        "pneumonia": ["infections"],
        "bronchitis": ["infections"],
        "influenza": ["influenza"],
        "chronic pain": ["chronic pain"],
        "depression": ["depression"],
        "asthma": ["asthma"],
        "cholesterol": ["cholesterol"],
        "high cholesterol": ["cholesterol"],
        "migraine": ["migraines"],
        "severe headache": ["migraines"],
        "anxiety": ["anxiety"],
        "panic attack": ["anxiety"],
        "arthritis": ["arthritis"],
        "insomnia": ["insomnia"],
        "cancer": ["cancer"],
        "tumor": ["cancer"],
        "allergic": ["allergies"],
        "hay fever": ["allergies"],
        "epilepsy": ["epilepsy"],
        "heart failure": ["heart failure"],
        "thyroid": ["thyroid disorders"],
        "osteoporosis": ["osteoporosis"],
        "acid reflux": ["acid reflux"],
        "adhd": ["adhd"],
        "kidney disease": ["kidney disease"],
        "parkinson": ["parkinson disease"],
        "glaucoma": ["glaucoma"],
        "anemia": ["anemia"],
        "autoimmune": ["autoimmune diseases"],
        "acne": ["acne"],
        "obesity": ["obesity"],
        "bipolar": ["bipolar disorder"],
        "gout": ["gout"],
        "obstructive pulmonary": ["copd"],
        "copd": ["copd"],
        "hormonal imbalance": ["hormonal imbalances"],
        "ulcer": ["ulcers"],
        "chronic fatigue": ["chronic fatigue syndrome"],
        "multiple sclerosis": ["multiple sclerosis"],
        "blood clotting": ["blood clotting disorders"],
        "liver disease": ["liver diseases"],
        "skin inflammation": ["skin inflammation"],
        "metabolic disorder": ["metabolic disorders"],
        "cardiovascular": ["cardiovascular diseases"],
        "digestive disorder": ["digestive disorders"],
        "immune system deficien": ["immune deficiencies"],
    }

    for keyword, conds in condition_map.items():
        if keyword in text_lower:
            conditions.extend(conds)

    return sorted(set(conditions))


KNOWN_SIDE_EFFECTS = [
    "abdominal discomfort",
    "abdominal pain",
    "allergic reactions",
    "anxiety",
    "bleeding",
    "bloating",
    "blurred vision",
    "bruising",
    "burning sensation",
    "changes in appetite",
    "chest discomfort",
    "confusion",
    "constipation",
    "coordination problems",
    "cough",
    "decreased appetite",
    "dependency risks",
    "dependency",
    "diarrhea",
    "dizziness",
    "drowsiness",
    "dry cough",
    "dry mouth",
    "dry skin",
    "dryness",
    "electrolyte imbalance",
    "eye irritation",
    "fatigue",
    "fever",
    "gastrointestinal discomfort",
    "hair loss",
    "hallucinations",
    "headache",
    "impaired concentration",
    "increased heart rate",
    "infections",
    "insomnia",
    "irritation",
    "itching",
    "joint pain",
    "liver enzyme changes",
    "liver toxicity",
    "low blood pressure",
    "mild irritation",
    "mood changes",
    "mood swings",
    "muscle cramps",
    "muscle pain",
    "muscle weakness",
    "nasal irritation",
    "nausea",
    "nervousness",
    "next-day drowsiness",
    "potential dependency with long-term use",
    "rash",
    "redness",
    "risk of bleeding",
    "sensitivity to light",
    "sensitivity to sunlight",
    "skin rash",
    "sleep disturbances",
    "stomach pain",
    "swelling",
    "throat irritation",
    "tremors",
    "weakness",
    "weight changes",
    "weight gain",
]


def _extract_side_effects(text: str) -> list[str]:
    """Extract individual side effects from a side-effect doc by matching known terms."""
    text_lower = text.lower()
    found = []
    for effect in KNOWN_SIDE_EFFECTS:
        if effect in text_lower:
            # Avoid double-counting substrings (e.g. "irritation" in "mild irritation")
            is_substring = any(
                effect != other and effect in other and other in text_lower
                for other in KNOWN_SIDE_EFFECTS
            )
            if not is_substring:
                found.append(effect)
    return sorted(found)


def build_drug_index(df: pd.DataFrame) -> dict:
    """Build a structured drug knowledge graph from the DataFrame.

    Returns a dict like:
        {
            "Drug A": {
                "usage_docs": [0],
                "side_effect_docs": [1],
                "all_docs": [0, 1],
                "conditions": ["hypertension"],
                "side_effects": ["dizziness", "headache", "fatigue"],
                "usage_text": "Drug A is commonly used to...",
                "side_effect_text": "Drug A may cause...",
            },
            ...
        }
    """
    drug_index: dict = {}

    for idx, row in df.iterrows():
        text = row["text"]
        drug_name = _extract_drug_name(text)
        if not drug_name:
            continue

        if drug_name not in drug_index:
            drug_index[drug_name] = {
                "usage_docs": [],
                "side_effect_docs": [],
                "all_docs": [],
                "conditions": [],
                "side_effects": [],
                "usage_text": "",
                "side_effect_text": "",
            }

        entry = drug_index[drug_name]
        doc_type = _classify_doc_type(text)

        if doc_type == "side_effect":
            entry["side_effect_docs"].append(idx)
            entry["side_effects"] = _extract_side_effects(text)
            entry["side_effect_text"] = text
        else:
            entry["usage_docs"].append(idx)
            entry["conditions"] = _extract_conditions(text)
            entry["usage_text"] = text

        entry["all_docs"].append(idx)

    return drug_index


def build_condition_index(drug_index: dict) -> dict[str, list[str]]:
    """Build a reverse index: condition -> list of drug names."""
    condition_index: dict[str, list[str]] = {}
    for drug_name, info in drug_index.items():
        for condition in info["conditions"]:
            condition_index.setdefault(condition, []).append(drug_name)
    return condition_index


def build_side_effect_index(drug_index: dict) -> dict[str, list[str]]:
    """Build a reverse index: side_effect -> list of drug names."""
    se_index: dict[str, list[str]] = {}
    for drug_name, info in drug_index.items():
        for se in info["side_effects"]:
            se_index.setdefault(se, []).append(drug_name)
    return se_index


KG_CSV_PATH = Path(__file__).resolve().parent.parent / "drug_knowledge.csv"


def export_knowledge_csv(
    drug_index: dict, path: Path | str | None = None
) -> pd.DataFrame:
    """Export the drug knowledge graph to a structured CSV.

    Columns: drug_name, conditions, side_effects, usage_text, side_effect_text
    """
    rows = []
    for drug_name in sorted(drug_index.keys(), key=_drug_sort_key):
        info = drug_index[drug_name]
        rows.append(
            {
                "drug_name": drug_name,
                "conditions": "; ".join(info["conditions"]),
                "side_effects": "; ".join(info["side_effects"]),
                "usage_text": info["usage_text"],
                "side_effect_text": info["side_effect_text"],
            }
        )

    df = pd.DataFrame(rows)
    out = Path(path) if path else KG_CSV_PATH
    df.to_csv(out, index=False)
    return df


def load_knowledge_csv(path: Path | str | None = None) -> pd.DataFrame:
    """Load the structured drug knowledge CSV."""
    path = Path(path) if path else KG_CSV_PATH
    return pd.read_csv(path)


def _drug_sort_key(name: str) -> tuple[int, str]:
    """Sort drugs: single-letter first (A-Z), then double-letter (AA-AX)."""
    suffix = name.replace("Drug ", "")
    return (len(suffix), suffix)
