"""Stage 16: generate grounded answers from retrieved video evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Protocol

from video_rag.index.embedding_providers import PROVIDERS as EMBEDDING_PROVIDERS
from video_rag.schemas import AnswerCitation, AnswerResult, RetrievalResult
from video_rag.search.retrieve import retrieve


class AnswerProvider(Protocol):
    """Adapter interface for an answer-generation backend."""

    name: str
    model: str

    def answer(self, question: str, context: str) -> str: ...


class MockAnswerProvider:
    """Deterministic offline provider for tests and smoke checks."""

    name = "mock"
    model = "mock-answer"

    def answer(self, question: str, context: str) -> str:
        if not context.strip():
            return "The answer was not found in the indexed video evidence."
        first_line = next(
            (line.strip() for line in context.splitlines() if line.strip()),
            "",
        )
        return f"Mock answer for: {question.strip()} Evidence: {first_line}"


class OpenAIAnswerProvider:
    """Real provider backed by the OpenAI Responses API."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def answer(self, question: str, context: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "install the 'answer' extra to use the openai answer provider: "
                "pip install -e .[answer]"
            ) from e

        try:
            from dotenv import load_dotenv
        except ImportError as e:
            raise RuntimeError(
                "install the 'answer' extra to load OPENAI_API_KEY from .env: "
                "pip install -e .[answer]"
            ) from e

        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; cannot use the openai answer provider"
            )

        client = OpenAI(api_key=api_key)
        prompt = _build_answer_prompt(question, context)
        try:
            response = client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            raise RuntimeError(f"openai answer generation failed: {e}") from e
        return response.output_text.strip()


PROVIDERS: dict[str, type[AnswerProvider]] = {
    "mock": MockAnswerProvider,
    "openai": OpenAIAnswerProvider,
}


def get_provider(name: str, *, model: str | None = None) -> AnswerProvider:
    cls = PROVIDERS.get(name)
    if cls is None:
        known = ", ".join(sorted(PROVIDERS)) or "(none)"
        raise ValueError(f"unknown answer provider {name!r} (known: {known})")
    if name == "openai":
        return cls(model=model) if model else cls()
    return cls()


def _format_time(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _result_block(index: int, result: RetrievalResult) -> str:
    parts = [
        f"[{index}] chunk_id={result.chunk_id}",
        f"time={_format_time(result.start_time)}-{_format_time(result.end_time)}",
    ]
    if result.transcript_text:
        parts.append(f"Transcript: {result.transcript_text}")
    if result.ocr_text:
        parts.append(f"On-screen text: {result.ocr_text}")
    if result.vlm_caption:
        parts.append(f"Visual caption: {result.vlm_caption}")
    if result.combined_text and not (
        result.transcript_text or result.ocr_text or result.vlm_caption
    ):
        parts.append(f"Indexed text: {result.combined_text}")
    if result.frame_paths:
        parts.append("Frames: " + ", ".join(result.frame_paths))
    return "\n".join(parts)


def format_context(results: list[RetrievalResult]) -> str:
    """Format retrieved evidence for answer generation."""
    return "\n\n".join(
        _result_block(index, result) for index, result in enumerate(results, start=1)
    )


def _build_answer_prompt(question: str, context: str) -> str:
    return (
        "Answer the question using only the video evidence below.\n"
        "Cite timestamps from the evidence for factual claims, using forms like "
        "[03:30-04:00].\n"
        "When useful, distinguish transcript, on-screen text, and visual captions.\n"
        "If the evidence is insufficient, say the answer was not found in the "
        "indexed video evidence.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Evidence:\n{context.strip()}"
    )


def _citations(results: list[RetrievalResult]) -> list[AnswerCitation]:
    return [
        AnswerCitation(
            chunk_id=result.chunk_id,
            video_id=result.video_id,
            start_time=result.start_time,
            end_time=result.end_time,
            frame_paths=result.frame_paths,
        )
        for result in results
    ]


def answer_from_results(
    video_id: str,
    question: str,
    results: list[RetrievalResult],
    *,
    provider: AnswerProvider | str = "openai",
    model: str = "gpt-4o-mini",
) -> AnswerResult:
    """Generate an answer from already-retrieved evidence."""
    if not video_id or not video_id.strip():
        raise ValueError("video_id must be non-empty")
    if not question or not question.strip():
        raise ValueError("question must be non-empty")

    resolved = get_provider(provider, model=model) if isinstance(provider, str) else provider
    context = format_context(results)
    if not results:
        answer_text = "The answer was not found in the indexed video evidence."
    else:
        answer_text = resolved.answer(question, context).strip()
    if not answer_text:
        raise ValueError("answer provider returned an empty answer")

    return AnswerResult(
        video_id=video_id,
        question=question.strip(),
        answer=answer_text,
        citations=_citations(results),
        retrieval_results=results,
        model=resolved.model,
        provider=resolved.name,
    )


def answer_video(
    video_id: str,
    question: str,
    *,
    data_dir: str | os.PathLike[str] = "data",
    chunk_seconds: int | float = 30,
    variant: str = "transcript_ocr_vlm",
    retrieval_provider: str = "openai",
    retrieval_model: str = "text-embedding-3-small",
    answer_provider: AnswerProvider | str = "openai",
    answer_model: str = "gpt-4o-mini",
    top_k: int = 5,
) -> AnswerResult:
    """Retrieve evidence for a question and generate a grounded answer."""
    results = retrieve(
        video_id=video_id,
        question=question,
        data_dir=data_dir,
        chunk_seconds=chunk_seconds,
        variant=variant,
        provider=retrieval_provider,
        model=retrieval_model,
        top_k=top_k,
    )
    return answer_from_results(
        video_id,
        question,
        results,
        provider=answer_provider,
        model=answer_model,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_rag.search.answer",
        description="Answer a question using retrieved video evidence.",
    )
    p.add_argument("--video-id", required=True, help="Registered video_id.")
    p.add_argument("--question", required=True, help="Question to answer.")
    p.add_argument("--data-dir", default="data", help="Artifact root (default: data).")
    p.add_argument(
        "--chunk-seconds",
        type=float,
        default=30,
        help="Chunk window size used by the index (default: 30).",
    )
    p.add_argument(
        "--variant",
        default="transcript_ocr_vlm",
        help="Embedding/search-text variant (default: transcript_ocr_vlm).",
    )
    p.add_argument(
        "--retrieval-provider",
        required=True,
        choices=sorted(EMBEDDING_PROVIDERS.keys()),
        help="Embedding provider used for query retrieval.",
    )
    p.add_argument(
        "--retrieval-model",
        default="text-embedding-3-small",
        help="Embedding model used for query retrieval.",
    )
    p.add_argument(
        "--answer-provider",
        required=True,
        choices=sorted(PROVIDERS.keys()),
        help="Answer-generation provider.",
    )
    p.add_argument(
        "--answer-model",
        default="gpt-4o-mini",
        help="Answer-generation model.",
    )
    p.add_argument("--top-k", type=int, default=5, help="Number of hits to use.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = answer_video(
            video_id=args.video_id,
            question=args.question,
            data_dir=args.data_dir,
            chunk_seconds=args.chunk_seconds,
            variant=args.variant,
            retrieval_provider=args.retrieval_provider,
            retrieval_model=args.retrieval_model,
            answer_provider=args.answer_provider,
            answer_model=args.answer_model,
            top_k=args.top_k,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
        print(f"FAIL  {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1

    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
