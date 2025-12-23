#!/usr/bin/env python
"""Inspect saved residual activation files."""

import torch
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Inspect residual activation files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_residuals.py residuals.pt              # Show summary
  python inspect_residuals.py residuals.pt --list       # List all prompts
  python inspect_residuals.py residuals.pt --index 5    # Show details for index 5
  python inspect_residuals.py residuals.pt --stats      # Show tensor statistics
        """
    )
    parser.add_argument("file", type=str, help="Path to .pt residuals file")
    parser.add_argument("--list", "-l", action="store_true", help="List all prompts and responses")
    parser.add_argument("--index", "-i", type=int, help="Show details for specific index")
    parser.add_argument("--stats", "-s", action="store_true", help="Show tensor statistics")
    parser.add_argument("--full", "-f", action="store_true", help="Show full text (don't truncate)")
    args = parser.parse_args()

    # Load file
    try:
        data = torch.load(args.file, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    residuals = data.get("residuals")
    prompts = data.get("prompts", [])
    responses = data.get("responses", [])
    indices = data.get("indices", [])

    # Summary (always shown)
    print("=" * 60)
    print("RESIDUALS FILE SUMMARY")
    print("=" * 60)
    print(f"Run ID:        {data.get('run_id', 'N/A')}")
    print(f"Timestamp:     {data.get('timestamp', 'N/A')}")
    print(f"Model:         {data.get('model_path', 'N/A')}")
    print(f"Capture point: {data.get('capture_point', 'N/A')}")
    print(f"Responses file:{data.get('responses_file', 'N/A')}")
    print(f"Num samples:   {len(prompts)}")
    if residuals is not None:
        print(f"Residual shape:{residuals.shape}")
        print(f"Dtype:         {residuals.dtype}")
    print()

    # Stats mode
    if args.stats and residuals is not None:
        print("=" * 60)
        print("TENSOR STATISTICS")
        print("=" * 60)
        print(f"Mean:          {residuals.mean().item():.6f}")
        print(f"Std:           {residuals.std().item():.6f}")
        print(f"Min:           {residuals.min().item():.6f}")
        print(f"Max:           {residuals.max().item():.6f}")
        print(f"L2 norm (avg): {residuals.norm(dim=-1).mean().item():.6f}")

        # Per-sample norms
        norms = residuals.norm(dim=-1)
        print(f"\nPer-sample L2 norms:")
        print(f"  Min:  {norms.min().item():.4f}")
        print(f"  Max:  {norms.max().item():.4f}")
        print(f"  Mean: {norms.mean().item():.4f}")
        print(f"  Std:  {norms.std().item():.4f}")
        print()

    # List mode
    if args.list:
        print("=" * 60)
        print("PROMPTS AND RESPONSES")
        print("=" * 60)
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            idx = indices[i] if i < len(indices) else i
            if args.full:
                print(f"\n[{idx}] PROMPT:\n{prompt}")
                print(f"\n[{idx}] RESPONSE:\n{response}")
            else:
                prompt_preview = prompt[:100].replace('\n', ' ')
                response_preview = response[:150].replace('\n', ' ')
                print(f"[{idx}] {prompt_preview}...")
                print(f"    -> {response_preview}...")
            print("-" * 40)

    # Single index mode
    if args.index is not None:
        idx = args.index
        if idx < 0 or idx >= len(prompts):
            print(f"Error: Index {idx} out of range (0-{len(prompts)-1})")
            sys.exit(1)

        print("=" * 60)
        print(f"DETAILS FOR INDEX {idx}")
        print("=" * 60)
        print(f"\nPROMPT:\n{prompts[idx]}")
        print(f"\nRESPONSE:\n{responses[idx]}")

        if residuals is not None:
            r = residuals[idx]
            print(f"\nRESIDUAL:")
            print(f"  Shape: {r.shape}")
            print(f"  L2 norm: {r.norm().item():.6f}")
            print(f"  Mean: {r.mean().item():.6f}")
            print(f"  Std: {r.std().item():.6f}")
            print(f"  Top 5 values: {r.topk(5).values.tolist()}")
            print(f"  Top 5 indices: {r.topk(5).indices.tolist()}")


if __name__ == "__main__":
    main()
