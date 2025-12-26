"""
Benchmark text generation performance for Gemma models.

Usage:
    python utils/benchmark_generation.py
    python utils/benchmark_generation.py --model /path/to/model
    python utils/benchmark_generation.py --attn eager    # Compare attention implementations
    python utils/benchmark_generation.py --attn sdpa
    python utils/benchmark_generation.py --attn flash_attention_2
"""

import os
import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def benchmark_generation(
    model_path: str,
    prompt: str = "Explain the theory of relativity in simple terms.",
    max_tokens: int = 100,
    num_runs: int = 3,
    attn_impl: str = "sdpa",
    warmup: bool = True,
):
    """Run generation benchmark and report performance metrics."""

    print(f"{'='*60}")
    print(f"Generation Benchmark")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Attention: {attn_impl}")
    print(f"Max tokens: {max_tokens}")
    print(f"Num runs: {num_runs}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    print(f"Loading model with attn_implementation='{attn_impl}'...")
    load_start = time.perf_counter()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
    except ValueError as e:
        if "flash" in str(e).lower():
            print(f"\nError: Flash Attention 2 not available. Install with:")
            print(f"  pip install flash-attn --no-build-isolation")
            return
        raise

    model.eval()
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s\n")

    # Prepare input
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer.encode(
        formatted,
        return_tensors="pt",
        add_special_tokens=True
    ).to(model.device)

    input_len = inputs.shape[1]
    print(f"Prompt: \"{prompt[:50]}...\"")
    print(f"Input tokens: {input_len}\n")

    # Warmup run
    if warmup:
        print("Warmup run...")
        with torch.no_grad():
            _ = model.generate(
                input_ids=inputs,
                max_new_tokens=10,
                do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("Warmup complete.\n")

    # Benchmark runs
    times = []
    tokens_generated = []

    print(f"Running {num_runs} benchmark iterations...")
    for i in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        num_new_tokens = outputs.shape[1] - input_len

        times.append(elapsed)
        tokens_generated.append(num_new_tokens)

        tok_per_sec = num_new_tokens / elapsed
        print(f"  Run {i+1}: {num_new_tokens} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")

    # Results
    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    avg_tok_per_sec = avg_tokens / avg_time

    print(f"\n{'='*60}")
    print(f"Results ({attn_impl})")
    print(f"{'='*60}")
    print(f"Average: {avg_tokens:.0f} tokens in {avg_time:.2f}s")
    print(f"Throughput: {avg_tok_per_sec:.1f} tokens/second")
    print(f"Latency per token: {1000 * avg_time / avg_tokens:.1f} ms")
    print(f"{'='*60}\n")

    # Show sample output
    output_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    print("Sample output (truncated):")
    print(f"  {output_text[:200]}...")

    # Memory usage
    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    return {
        "model": model_path,
        "attn_implementation": attn_impl,
        "avg_time": avg_time,
        "avg_tokens": avg_tokens,
        "tokens_per_second": avg_tok_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Gemma generation performance")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=os.getenv("MODEL_PATH", "google/gemma-3-12b-it"),
        help="Path to model (default: $MODEL_PATH or google/gemma-3-12b-it)"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Explain the theory of relativity in simple terms.",
        help="Prompt to generate from"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)"
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=3,
        help="Number of benchmark runs (default: 3)"
    )
    parser.add_argument(
        "--attn",
        type=str,
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
        help="Attention implementation (default: sdpa)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all available attention implementations"
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup run"
    )

    args = parser.parse_args()

    if args.compare:
        # Compare different attention implementations
        results = []
        for attn in ["eager", "sdpa"]:
            print(f"\n{'#'*60}")
            print(f"Testing: {attn}")
            print(f"{'#'*60}\n")

            result = benchmark_generation(
                model_path=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                num_runs=args.runs,
                attn_impl=attn,
                warmup=not args.no_warmup,
            )
            if result:
                results.append(result)

            # Clear GPU memory between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Try flash attention if available
        try:
            print(f"\n{'#'*60}")
            print(f"Testing: flash_attention_2")
            print(f"{'#'*60}\n")

            result = benchmark_generation(
                model_path=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                num_runs=args.runs,
                attn_impl="flash_attention_2",
                warmup=not args.no_warmup,
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"Flash Attention 2 not available: {e}")

        # Summary
        if results:
            print(f"\n{'='*60}")
            print("Comparison Summary")
            print(f"{'='*60}")
            for r in results:
                print(f"  {r['attn_implementation']:20s}: {r['tokens_per_second']:.1f} tok/s")

            if len(results) > 1:
                baseline = results[0]['tokens_per_second']
                for r in results[1:]:
                    speedup = r['tokens_per_second'] / baseline
                    print(f"    → {r['attn_implementation']} is {speedup:.2f}x vs {results[0]['attn_implementation']}")
    else:
        benchmark_generation(
            model_path=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            num_runs=args.runs,
            attn_impl=args.attn,
            warmup=not args.no_warmup,
        )


if __name__ == "__main__":
    main()
