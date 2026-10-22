#!/usr/bin/env python3
"""Load Qwen3-VL once and sequentially describe a fixed list of images.

Each llm.chat() call is a separate conversation. Timings are wall-clock
intervals; they are not TTFT or decode-only throughput.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time

from vllm_image_check import DEFAULT_MODEL


PROJECT = Path(__file__).resolve().parent
DEFAULT_IMAGES = Path(
    "/home/smisei/project/RTDETR-ONNX-TensorRT/ultralytics/coco/images/calibration"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--question",
        default="请简要描述图片中主要物品及其大致位置。无法确认的内容请说明无法确认。",
    )
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--list-only", action="store_true", help="List inputs without loading vLLM")
    args = parser.parse_args()
    if not args.image_dir.is_dir():
        parser.error(f"Image directory does not exist: {args.image_dir}")
    if args.limit < 1 or args.warmup < 0 or args.max_side < 32:
        parser.error("Require limit >= 1, warmup >= 0, max-side >= 32")
    if not 0 < args.max_tokens < args.max_model_len:
        parser.error("Require 0 < max-tokens < max-model-len")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")
    if not args.question.strip():
        parser.error("Question must not be empty")
    return args


def select_images(directory, limit):
    images = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )
    if len(images) < limit:
        raise ValueError(f"Requested {limit} images, but found only {len(images)}")
    return images[:limit], images


def prepare_messages(path, max_side, question):
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    original_size = image.size
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": image},
            {"type": "text", "text": question},
        ],
    }]
    return messages, original_size, image.size


def latency_summary(values):
    if not values:
        return None
    percentiles = statistics.quantiles(values, n=100, method="inclusive") if len(values) > 1 else [values[0]] * 99
    return {
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": percentiles[94],
        "p99": percentiles[98],
        "min": min(values),
        "max": max(values),
    }


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    selected, all_images = select_images(args.image_dir, args.limit)
    print(f"Selected {len(selected)} of {len(all_images)} images, sorted by filename.", flush=True)
    print(f"First: {selected[0].name}; last: {selected[-1].name}", flush=True)
    if args.list_only:
        for i, path in enumerate(selected, 1):
            print(f"{i:02d} {path.name}")
        return

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    output = args.output_dir or PROJECT / "runs" / (
        "vllm_batch_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    engine_args = {
        "model": args.model,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "max_num_seqs": 1,
        "max_num_batched_tokens": min(1024, args.max_model_len),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "seed": 0,
        "limit_mm_per_prompt": {"image": 1, "video": 0},
        # Same pixel bounds as the single-image script, using the current API.
        "mm_processor_kwargs": {
            "size": {"shortest_edge": 32 * 32, "longest_edge": args.max_side * args.max_side},
        },
    }
    # Prefer an image outside the measured set, to avoid warming its image cache.
    warmup_image = all_images[args.limit] if len(all_images) > args.limit else selected[0]
    manifest = {
        "engine_args": engine_args,
        "question": args.question,
        "max_output_tokens": args.max_tokens,
        "temperature": 0.0,
        "warmup_requests": args.warmup,
        "warmup_image": str(warmup_image.resolve()) if args.warmup else None,
        "images": [str(p.resolve()) for p in selected],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "transformers", "pillow")
        },
    }
    source_manifest = Path(args.model) / "source_manifest.json"
    if source_manifest.is_file():
        manifest["model_source"] = json.loads(source_manifest.read_text())
    save_json(output / "manifest.json", manifest)
    print(f"Results: {output.resolve()}", flush=True)
    records = []
    summary = {
        "status": "running", "selected_images": len(selected),
        "initialization_wall_seconds": None, "warmup_wall_seconds": [],
        "timing_note": "Per-image chat wall time includes vLLM input processing, vision, prefill, and generation. Image file preparation is separate. Warmup and model initialization are excluded from measured requests. Not TTFT or decode-only throughput.",
    }
    current_image = None
    try:
        started = time.perf_counter()
        llm = LLM(**engine_args)  # Created once, outside both loops.
        summary["initialization_wall_seconds"] = time.perf_counter() - started
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

        for i in range(args.warmup):
            messages, _, _ = prepare_messages(warmup_image, args.max_side, args.question)
            started = time.perf_counter()
            llm.chat(messages, sampling_params=sampling, use_tqdm=False)
            elapsed = time.perf_counter() - started
            summary["warmup_wall_seconds"].append(elapsed)
            print(f"Warmup {i + 1}/{args.warmup}: {elapsed:.3f}s", flush=True)

        with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
            for i, path in enumerate(selected, 1):
                current_image = str(path.resolve())
                started = time.perf_counter()
                messages, original_size, submitted_size = prepare_messages(path, args.max_side, args.question)
                prepare_seconds = time.perf_counter() - started
                started = time.perf_counter()
                result = llm.chat(messages, sampling_params=sampling, use_tqdm=False)[0]
                chat_seconds = time.perf_counter() - started
                completion = result.outputs[0]
                if not completion.text.strip():
                    raise RuntimeError(f"Empty answer for {path.name}")
                record = {
                    "index": i, "image": current_image,
                    "original_image_size": original_size, "submitted_image_size": submitted_size,
                    "image_prepare_seconds": prepare_seconds,
                    "chat_wall_seconds": chat_seconds,
                    "output_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "answer": completion.text,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()  # Preserve each completed answer if a later request fails.
                records.append(record)
                print(f"[{i:02d}/{len(selected)}] {path.name}: {chat_seconds:.3f}s, "
                      f"tokens={record['output_tokens']}, finish={completion.finish_reason}", flush=True)
        summary["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       error=f"{type(exc).__name__}: {exc}", current_image=current_image)
        raise
    finally:
        summary.update(
            completed_images=len(records),
            truncated_answers=sum(r["finish_reason"] == "length" for r in records),
            chat_wall_seconds=latency_summary([r["chat_wall_seconds"] for r in records]),
            total_output_tokens=sum(r["output_tokens"] for r in records),
        )
        save_json(output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
