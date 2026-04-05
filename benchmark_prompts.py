"""Compare prompt strategies across multiple OpenRouter-hosted LLMs.

Runs the same hybrid retrieval + keyword/refusal scoring as eval_pipeline.py
for a 3x3 grid: three prompt styles × three models.

Usage:
    export OPENROUTER_API_KEY=sk-or-v1-...
    uv run python benchmark_prompts.py

Optional:
    uv run python benchmark_prompts.py --models "model1,model2,model3" --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass

from openai import OpenAI

from eval_pipeline import EVAL_CASES, FEW_SHOT_EXAMPLES, REFUSAL_PATTERNS
from rag.retrieval import HybridRetriever, classify_query

# ---------------------------------------------------------------------------
# Models (OpenRouter IDs — adjust if your account exposes different free tiers)
# ---------------------------------------------------------------------------

DEFAULT_MODELS = (
    "qwen/qwen3.6-plus:free",
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
)

# Short hints so one template still respects query-type routing from Phase 3
TYPE_HINTS: dict[str, str] = {
    "drug_specific": "Answer about the specific drug(s) named in the question.",
    "condition_lookup": "Name which drug(s) treat the condition and summarize mechanism briefly.",
    "comparison": "List every drug in the documents that matches the question; one line per drug.",
    "multi_hop": "Address both what the drug is used for AND side effects or risks.",
    "safety": "List all drugs in the documents relevant to the safety concern.",
    "general": "Answer concisely from the documents only.",
}


def build_prompt_vanilla(query: str, context: str, query_type: str) -> str:
    hint = TYPE_HINTS.get(query_type, TYPE_HINTS["general"])
    return f"""{hint}

Answer using ONLY the documents below. If the answer is not contained in them, reply exactly:
"I don't know based on the provided data."

Documents:
{context}

Question: {query}

Answer:"""


def build_prompt_few_shot(query: str, context: str, query_type: str) -> str:
    hint = TYPE_HINTS.get(query_type, TYPE_HINTS["general"])
    return f"""You are a medical knowledge assistant. {hint}
Use ONLY the documents below. If the answer is not there, say "I don't know based on the provided data."

{FEW_SHOT_EXAMPLES}

Now answer:

Documents:
{context}

Question: {query}

Use clear structure (bullets where helpful). Answer:"""


def build_prompt_expert(query: str, context: str, query_type: str) -> str:
    hint = TYPE_HINTS.get(query_type, TYPE_HINTS["general"])
    return f"""You are a clinical informatics assistant helping reviewers interpret internal drug documentation.

Operational rules:
1. Ground every factual claim in the provided documents; do not infer beyond them.
2. Prefer short bullet lists for side effects and multi-drug answers.
3. If the documents do not support a definitive answer (including "cure" or absolute claims), state that the information is not available in the provided data.
4. Do not give dosing, prescribing, or personal medical advice.

Task focus: {hint}

Documents:
{context}

Question: {query}

Structured answer:"""


PROMPT_BUILDERS = {
    "vanilla": build_prompt_vanilla,
    "few_shot": build_prompt_few_shot,
    "expert": build_prompt_expert,
}


@dataclass
class CellResult:
    model: str
    prompt_variant: str
    mean_score: float
    pass_count: int
    total: int
    errors: int


def call_llm(client: OpenAI, model: str, prompt: str, temperature: float = 0.2) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def score_answer(answer: str, tc: dict) -> tuple[float, str]:
    if tc["should_refuse"]:
        lower = answer.lower()
        ok = any(p in lower for p in REFUSAL_PATTERNS)
        return (1.0 if ok else 0.0, "PASS" if ok else "FAIL")
    kws = tc["expected_keywords"]
    if not kws:
        return 0.0, "FAIL"
    hits = [kw for kw in kws if kw.lower() in answer.lower()]
    score = len(hits) / len(kws)
    return score, "PASS" if score >= 0.5 else "FAIL"


def run_benchmark(
    models: tuple[str, ...],
    variants: tuple[str, ...],
    *,
    delay_s: float = 0.4,
    verbose: bool = False,
) -> list[CellResult]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set OPENROUTER_API_KEY in the environment (no key in this script)."
        )

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    retriever = HybridRetriever()
    results: list[CellResult] = []

    for model in models:
        for variant in variants:
            if variant not in PROMPT_BUILDERS:
                raise ValueError(f"Unknown variant: {variant}")

            builder = PROMPT_BUILDERS[variant]
            total_score = 0.0
            passes = 0
            errors = 0

            for tc in EVAL_CASES:
                query = tc["query"]
                qtype = classify_query(query)
                retrieved = retriever.retrieve(query)

                if not retrieved.attrs.get("is_answerable", True):
                    answer = "I don't know based on the provided data."
                else:
                    context = "\n".join(retrieved["text"].tolist())
                    prompt = builder(query, context, qtype)
                    try:
                        answer = call_llm(client, model, prompt)
                    except Exception as e:
                        errors += 1
                        answer = f"[API error: {e}]"
                        if verbose:
                            print(f"  ERROR {model} / {variant} Q{tc['id']}: {e}")

                score, status = score_answer(answer, tc)
                total_score += score
                if status == "PASS":
                    passes += 1

                if verbose:
                    print(
                        f"  {model} | {variant} | Q{tc['id']} | {score:.0%} | {status}"
                    )

                time.sleep(delay_s)

            n = len(EVAL_CASES)
            results.append(
                CellResult(
                    model=model,
                    prompt_variant=variant,
                    mean_score=total_score / n,
                    pass_count=passes,
                    total=n,
                    errors=errors,
                )
            )

    return results


def print_markdown_table(cells: list[CellResult], variants: tuple[str, ...]) -> None:
    models = sorted({c.model for c in cells})
    print("\n### Mean keyword / refusal score (0–100%)\n")
    header = "| Model | " + " | ".join(v.replace("_", " ") for v in variants) + " |"
    sep = "|" + "|".join(["---"] * (len(variants) + 1)) + "|"
    print(header)
    print(sep)
    by_model: dict[str, dict[str, float]] = {m: {} for m in models}
    for c in cells:
        by_model[c.model][c.prompt_variant] = c.mean_score
    for m in models:
        row = f"| `{m}` |"
        for v in variants:
            s = by_model[m].get(v, float("nan"))
            row += f" {s:.0%} |" if s == s else " — |"
        print(row)

    print("\n### Pass rate (≥50% keywords or correct refusal)\n")
    print(header)
    print(sep)
    for m in models:
        row = f"| `{m}` |"
        for v in variants:
            cell = next(
                (c for c in cells if c.model == m and c.prompt_variant == v), None
            )
            if cell:
                row += f" {cell.pass_count}/{cell.total}"
                if cell.errors:
                    row += f" ({cell.errors} err)"
                row += " |"
            else:
                row += " — |"
        print(row)


def write_csv(path: str, cells: list[CellResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "prompt_variant",
                "mean_score",
                "pass_count",
                "total",
                "errors",
            ]
        )
        for c in cells:
            w.writerow(
                [
                    c.model,
                    c.prompt_variant,
                    f"{c.mean_score:.4f}",
                    c.pass_count,
                    c.total,
                    c.errors,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt × model benchmark on OpenRouter"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated OpenRouter model IDs",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="vanilla,few_shot,expert",
        help="Comma-separated: vanilla,few_shot,expert",
    )
    parser.add_argument("--csv", type=str, default="benchmark_results.csv")
    parser.add_argument(
        "--delay", type=float, default=0.4, help="Seconds between API calls"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())

    print(
        f"Benchmark: {len(models)} models × {len(variants)} prompt variants × {len(EVAL_CASES)} questions = {len(models) * len(variants) * len(EVAL_CASES)} LLM calls"
    )
    print("Retriever: HybridRetriever (fixed across all runs)")
    print(
        f"Questions: {len(EVAL_CASES)} held-out eval cases (same as eval_pipeline.py)"
    )

    cells = run_benchmark(models, variants, delay_s=args.delay, verbose=args.verbose)
    print_markdown_table(cells, variants)
    write_csv(args.csv, cells)
    print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
