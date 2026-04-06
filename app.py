"""Gradio UI for the Drug Knowledge Assistant."""

from __future__ import annotations

import itertools
import os

import gradio as gr

from rag.generation import RAGPipeline

API_KEYS = [
    "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",
    "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",
    "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",
    "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",
    "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",
]
_key_cycle = itertools.cycle(API_KEYS)

DEFAULT_MODELS = (
    "qwen/qwen3.6-plus:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
)
NO_LLM = "No LLM (Retrieval Only)"


def _get_pipeline(model_choice: str) -> RAGPipeline:
    if model_choice == NO_LLM:
        return RAGPipeline(api_key="")
    return RAGPipeline(api_key=next(_key_cycle), model=model_choice)


def ask(query: str, model_choice: str) -> tuple[str, str, str]:
    """Process a query and return (answer, sources, metadata)."""
    if not query.strip():
        return "Please enter a question.", "", ""

    pipeline = _get_pipeline(model_choice)
    result = pipeline.answer(query)

    mode_label = {
        "llm": f"LLM ({model_choice})",
        "fallback": "Fallback (no LLM)",
        "refused": "Refused (unanswerable)",
    }
    meta_lines = [
        f"Model: {model_choice}",
        f"Query type: {result['query_type']}",
        f"Mode: {mode_label.get(result['mode'], result['mode'])}",
        f"Top retrieval score: {result['max_score']:.3f}",
    ]
    metadata = "\n".join(meta_lines)

    return result["answer"], result["sources"], metadata


with gr.Blocks(
    title="Drug Knowledge Assistant",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # Drug Knowledge Assistant
        ### RAG System | EPAM Mini Datathon

        Ask any question about drugs -- usage, side effects, safety, comparisons.
        The system uses **hybrid 3-layer retrieval** with drug-aware document linking.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What are the side effects of Drug A?",
                lines=2,
            )
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary", scale=2)
                model_selector = gr.Dropdown(
                    choices=[NO_LLM] + list(DEFAULT_MODELS),
                    value=DEFAULT_MODELS[0],
                    label="Model",
                    scale=1,
                )

        with gr.Column(scale=1):
            metadata_output = gr.Textbox(label="Query Info", lines=5, interactive=False)

    answer_output = gr.Textbox(label="Answer", lines=8, interactive=False)
    sources_output = gr.Textbox(label="Source Documents", lines=6, interactive=False)

    gr.Markdown("### Try these examples:")
    gr.Examples(
        examples=[
            ["What are the side effects of Drug A?"],
            ["Which drug is used for diabetes?"],
            ["What treats hypertension?"],
            ["Which drug is used for asthma and what are its side effects?"],
            ["Which drugs have dependency risks?"],
            ["Which drugs are used for infections?"],
            ["Which drugs affect the brain?"],
            ["Which drug cures cancer completely?"],
            ["Which drug is used for cholesterol and what risks does it have?"],
            ["Which drugs can cause liver problems?"],
        ],
        inputs=query_input,
    )

    submit_btn.click(
        fn=ask,
        inputs=[query_input, model_selector],
        outputs=[answer_output, sources_output, metadata_output],
    )
    query_input.submit(
        fn=ask,
        inputs=[query_input, model_selector],
        outputs=[answer_output, sources_output, metadata_output],
    )

if __name__ == "__main__":
    demo.launch(share=True)
