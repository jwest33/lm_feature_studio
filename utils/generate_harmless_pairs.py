"""
Generate harmless prompt variants from harmful prompts using llama-server.

For each harmful prompt (e.g., "How can I hack a website?"), generates a
semantically related but harmless version (e.g., "How can I build a website?").

Uses llama-server's OpenAI-compatible API for fast inference.

  # Basic usage (assumes llama-server running on port 8080)
  python utils/generate_harmless_pairs.py --prompts prompts/harmful.txt

  # Custom server URL
  python utils/generate_harmless_pairs.py --prompts prompts/harmful.txt --url http://localhost:8081

  # Test with just first 10 prompts
  python utils/generate_harmless_pairs.py --prompts prompts/harmful.txt --limit 10

  # Parallel requests for faster processing
  python utils/generate_harmless_pairs.py --prompts prompts/harmful.txt --parallel 4
"""

import argparse
import json
import sys
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime
from typing import Optional


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SERVER_URL = "http://127.0.0.1:8080"
DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_PARALLEL = 4  # Number of parallel requests


# =============================================================================
# Prompt Building (Gemma 3 format)
# =============================================================================

def build_gemma3_prompt(harmful_prompt: str) -> str:
    """
    Build a prompt using Gemma 3 chat template.
    """
    system_message = (
        "You are a helpful assistant that transforms harmful prompts into harmless ones. "
        "You output ONLY the transformed prompt with no explanation or preamble."
    )

    user_message = f"""Transform this harmful prompt into a harmless, constructive version covering a similar topic:

Harmful: {harmful_prompt}

Output only the harmless version:"""

    # Gemma 3 chat template format
    prompt = f"""<start_of_turn>user
{system_message}

{user_message}<end_of_turn>
<start_of_turn>model
"""
    return prompt


# =============================================================================
# Generation via llama-server
# =============================================================================

async def generate_harmless_variant(
    session: aiohttp.ClientSession,
    server_url: str,
    harmful_prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Generate a harmless variant of a harmful prompt via llama-server.
    """
    try:
        prompt = build_gemma3_prompt(harmful_prompt)

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop": ["<end_of_turn>", "<start_of_turn>"],
        }

        async with session.post(
            f"{server_url}/completion",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                return {
                    "success": False,
                    "harmful": harmful_prompt,
                    "harmless": "",
                    "error": f"HTTP {response.status}: {error_text}",
                }

            result = await response.json()
            harmless_prompt = result.get("content", "").strip()

            # Clean up the output - take first line if multiple
            harmless_prompt = harmless_prompt.split("\n")[0].strip()

            # Remove common prefixes the model might add
            prefixes_to_remove = [
                "Harmless:", "Harmless prompt:", "Safe:", "Safe prompt:",
                "Here's", "Here is", "The harmless version:",
            ]
            for prefix in prefixes_to_remove:
                if harmless_prompt.lower().startswith(prefix.lower()):
                    harmless_prompt = harmless_prompt[len(prefix):].strip()

            return {
                "success": True,
                "harmful": harmful_prompt,
                "harmless": harmless_prompt,
            }

    except asyncio.TimeoutError:
        return {
            "success": False,
            "harmful": harmful_prompt,
            "harmless": "",
            "error": "Request timed out",
        }
    except Exception as e:
        return {
            "success": False,
            "harmful": harmful_prompt,
            "harmless": "",
            "error": str(e),
        }


async def process_batch(
    session: aiohttp.ClientSession,
    server_url: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    start_idx: int,
    total: int,
) -> list[dict]:
    """Process a batch of prompts concurrently."""
    tasks = []
    for i, prompt in enumerate(prompts):
        tasks.append(generate_harmless_variant(
            session=session,
            server_url=server_url,
            harmful_prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ))

    results = await asyncio.gather(*tasks)

    # Print progress
    for i, result in enumerate(results):
        idx = start_idx + i + 1
        if result["success"]:
            print(f"[{idx}/{total}] {result['harmful'][:50]}...")
            print(f"    -> {result['harmless'][:60]}...")
        else:
            print(f"[{idx}/{total}] {result['harmful'][:50]}...")
            print(f"    -> ERROR: {result.get('error', 'Unknown')}")

    return results


async def process_prompts(
    server_url: str,
    harmful_prompts: list[str],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    parallel: int = DEFAULT_PARALLEL,
) -> list[dict]:
    """
    Process all harmful prompts and generate harmless variants.
    """
    results = []
    total = len(harmful_prompts)

    async with aiohttp.ClientSession() as session:
        # Process in batches
        for i in range(0, total, parallel):
            batch = harmful_prompts[i:i + parallel]
            batch_results = await process_batch(
                session=session,
                server_url=server_url,
                prompts=batch,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                start_idx=i,
                total=total,
            )
            results.extend(batch_results)

    return results


# =============================================================================
# I/O
# =============================================================================

def load_prompts(prompts_file: Path) -> list[str]:
    """Load prompts from a text file (one per line)."""
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def save_results(
    results: list[dict],
    output_path: Path,
    config: dict,
) -> None:
    """Save results to a JSON file."""
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "total_prompts": len(results),
            "successful": sum(1 for r in results if r["success"]),
        },
        "pairs": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Results saved to: {output_path}")


def save_prompts_txt(results: list[dict], output_path: Path) -> None:
    """Save just the harmless prompts to a text file (one per line)."""
    txt_path = output_path.with_suffix(".txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            if r["success"] and r["harmless"]:
                f.write(r["harmless"] + "\n")

    print(f"[INFO] Harmless prompts saved to: {txt_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate harmless prompt variants from harmful prompts using llama-server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (llama-server on default port 8080)
  python generate_harmless_pairs.py --prompts ../prompts/harmful.txt

  # Custom server URL
  python generate_harmless_pairs.py --prompts ../prompts/harmful.txt --url http://localhost:8081

  # Process only first N prompts (for testing)
  python generate_harmless_pairs.py --prompts ../prompts/harmful.txt --limit 10

  # More parallel requests for faster processing
  python generate_harmless_pairs.py --prompts ../prompts/harmful.txt --parallel 8
        """
    )

    parser.add_argument(
        "--prompts", "-p",
        type=Path,
        required=True,
        help="Path to harmful prompts file (one per line)"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON file path (default: harmless_pairs.json)"
    )

    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_SERVER_URL,
        help=f"llama-server URL (default: {DEFAULT_SERVER_URL})"
    )

    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Process only first N prompts (for testing)"
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL,
        help=f"Number of parallel requests (default: {DEFAULT_PARALLEL})"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum new tokens to generate (default: {DEFAULT_MAX_TOKENS})"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})"
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Top-p sampling (default: {DEFAULT_TOP_P})"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Top-k sampling (default: {DEFAULT_TOP_K})"
    )

    return parser.parse_args()


async def async_main():
    args = parse_args()

    # Validate inputs
    if not args.prompts.exists():
        print(f"[ERROR] Prompts file not found: {args.prompts}")
        sys.exit(1)

    # Set default output path
    if args.output is None:
        args.output = Path("harmless_pairs.json")

    # Load prompts
    print(f"[INFO] Loading prompts from: {args.prompts}")
    harmful_prompts = load_prompts(args.prompts)

    if args.limit:
        harmful_prompts = harmful_prompts[:args.limit]

    print(f"[INFO] Loaded {len(harmful_prompts)} prompts")

    print("\n" + "=" * 60)
    print("Harmless Prompt Generator (llama-server)")
    print("=" * 60)
    print(f"Server: {args.url}")
    print(f"Prompts: {args.prompts}")
    print(f"Count: {len(harmful_prompts)}")
    print(f"Parallel: {args.parallel}")
    print(f"Output: {args.output}")
    print("=" * 60 + "\n")

    # Check server is reachable
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print(f"[WARN] Server health check returned {resp.status}")
    except Exception as e:
        print(f"[ERROR] Cannot connect to llama-server at {args.url}: {e}")
        print("[INFO] Make sure llama-server is running:")
        print(f"  llama-server -m model.gguf --port 8080")
        sys.exit(1)

    print("[INFO] Generating harmless variants...\n")

    results = await process_prompts(
        server_url=args.url,
        harmful_prompts=harmful_prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        parallel=args.parallel,
    )

    # Save results
    config = {
        "server_url": args.url,
        "prompts_file": str(args.prompts),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "parallel": args.parallel,
    }

    save_results(results, args.output, config)
    save_prompts_txt(results, args.output)

    # Summary
    successful = sum(1 for r in results if r["success"])

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print(f"Successful: {successful}/{len(results)}")
    print("=" * 60)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
