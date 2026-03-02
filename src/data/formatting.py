"""Dataset-specific prompt formatters.

Each formatter is a callable that takes a batch dict (lists of column values)
and returns a list of formatted prompt strings.

To add a new dataset, register a function in ``FORMATTERS``.
"""

from __future__ import annotations

from typing import Callable


FormatterFn = Callable[[dict[str, list]], list[str]]


def _format_medqa(examples: dict[str, list], template: str) -> list[str]:
    """Format bigbio/med_qa examples into instruction-style prompts."""
    texts = []
    for i in range(len(examples[next(iter(examples))])):
        question = examples.get("question", examples.get("sent1", [""]))[i]
        options_raw = examples.get("options", [[]])[i]

        if isinstance(options_raw, dict):
            options_str = "\n".join(
                f"  {k}. {v}" for k, v in options_raw.items()
            )
        elif isinstance(options_raw, list):
            letters = "ABCDEFGHIJ"
            options_str = "\n".join(
                f"  {letters[j]}. {opt}" for j, opt in enumerate(options_raw)
            )
        else:
            options_str = str(options_raw)

        answer = examples.get("answer", examples.get("answer_idx", [""]))[i]
        if isinstance(answer, int) and isinstance(options_raw, list):
            answer = options_raw[answer]

        text = template.format(question=question, options=options_str, answer=answer)
        texts.append(text)
    return texts


def _format_pubmedqa(examples: dict[str, list], template: str) -> list[str]:
    """Format qiaojin/PubMedQA examples."""
    texts = []
    for i in range(len(examples[next(iter(examples))])):
        context_parts = examples.get("context", examples.get("CONTEXTS", [[]]))[i]
        if isinstance(context_parts, list):
            context = " ".join(context_parts)
        else:
            context = str(context_parts)

        question = examples.get("question", examples.get("QUESTION", [""]))[i]
        answer = examples.get("final_decision", examples.get("LONG_ANSWER", [""]))[i]
        text = template.format(context=context, question=question, answer=answer)
        texts.append(text)
    return texts


def _format_generic(examples: dict[str, list], template: str) -> list[str]:
    """Fallback: try to fill template from column names."""
    n = len(examples[next(iter(examples))])
    texts = []
    for i in range(n):
        row = {k: v[i] for k, v in examples.items()}
        try:
            texts.append(template.format(**row))
        except KeyError:
            texts.append(str(row))
    return texts


FORMATTERS: dict[str, Callable] = {
    "bigbio/med_qa": _format_medqa,
    "qiaojin/PubMedQA": _format_pubmedqa,
    "GBaker/MedQA-USMLE-4-options": _format_medqa,
}


def build_formatter(dataset_name: str, template: str | None = None) -> FormatterFn:
    """Return a batch formatter function for the given dataset."""
    fn = FORMATTERS.get(dataset_name, _format_generic)

    default_template = (
        "### Question:\n{question}\n\n### Answer:\n{answer}"
    )
    tpl = template or default_template

    def _format(examples: dict[str, list]) -> list[str]:
        return fn(examples, tpl)

    return _format
