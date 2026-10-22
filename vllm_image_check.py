#!/usr/bin/env python3
"""Single-image Qwen3-VL smoke check using vLLM's Python API.

Run in a dedicated vLLM environment. Timings are CPU wall-clock intervals,
not TTFT, decode-only throughput, or isolated GPU execution time.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time


DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "Qwen3-VL-2B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Local model directory or Hugging Face model ID (default: project models directory)",
    )
    parser.add_argument("--revision", help="Optional Hugging Face revision/commit")
    parser.add_argument(
        "--question",
        default="请描述图片中可见的物品及其大致位置。无法确认的内容请说明无法确认。",
    )
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--report", type=Path, help="Optional JSON result path")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if args.max_side < 32:
        parser.error("--max-side must be at least 32")
    if not 0 < args.max_tokens < args.max_model_len:
        parser.error("Require 0 < --max-tokens < --max-model-len")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if not args.question.strip():
        parser.error("--question must not be empty")
    return args


def main():
    args = parse_args()
    # Keep worker startup safe if the parent initializes CUDA.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from PIL import Image, ImageOps
    from vllm import LLM, SamplingParams

    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    original_size = image.size
    image.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)

    engine_args = {
        "model": args.model,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "max_num_seqs": 1,
        "max_num_batched_tokens": min(1024, args.max_model_len),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "seed": 0,
        "limit_mm_per_prompt": {"image": 1, "video": 0},
        # Bound both real inputs and multimodal profiling allocations.
        "mm_processor_kwargs": {
            "min_pixels": 32 * 32,
            "max_pixels": args.max_side * args.max_side,
        },
    }
    if args.revision:
        engine_args["revision"] = args.revision
        engine_args["tokenizer_revision"] = args.revision

    print(f"Image: {args.image.resolve()} {original_size} -> {image.size}", flush=True)
    if Path(args.model).is_dir():
        print(f"Initializing vLLM with local model: {Path(args.model).resolve()}", flush=True)
    else:
        print("Initializing vLLM; first run may download weights and compile kernels.", flush=True)
    started = time.perf_counter()
    llm = LLM(**engine_args)
    initialization_seconds = time.perf_counter() - started

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": image},
            {"type": "text", "text": args.question},
        ],
    }]
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    started = time.perf_counter()
    result = llm.chat(messages, sampling_params=sampling, use_tqdm=False)[0]
    generate_seconds = time.perf_counter() - started
    completion = result.outputs[0]
    if not completion.text.strip():
        raise RuntimeError("Model returned an empty answer; smoke check failed")

    report = {
        "image": str(args.image.resolve()),
        "original_image_size": original_size,
        "submitted_image_size": image.size,
        "question": args.question,
        "engine_args": engine_args,
        "max_output_tokens": args.max_tokens,
        "temperature": 0.0,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "transformers", "pillow")
        },
        "initialization_wall_seconds": initialization_seconds,
        "chat_wall_seconds": generate_seconds,
        "timing_note": "First request, no warmup; chat wall time includes input processing and generation. Not TTFT or decode-only timing.",
        "output_tokens": len(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "answer": completion.text,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if completion.finish_reason == "length":
        print("Output reached --max-tokens and may be incomplete.")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {args.report.resolve()}")


if __name__ == "__main__":
    main()
