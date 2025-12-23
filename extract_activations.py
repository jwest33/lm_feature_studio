import torch
import json
import argparse
import uuid
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

# ==========================================
# CONFIGURATION (defaults, can be overridden via CLI)
# ==========================================
MODEL_PATH = "D:\\models\\gemma-3-4b-it"
PROMPTS_FILE = "harmful.txt"
OUTPUT_DIR = "activations_output"
MAX_NEW_TOKENS = 512
BATCH_SIZE = 1
LIMIT = None  # None = no limit


def parse_args():
    parser = argparse.ArgumentParser(description="Extract model responses and residual activations")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to model")
    parser.add_argument("--prompts", type=str, default=PROMPTS_FILE, help="Path to prompts file")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS, help="Max tokens to generate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for processing")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Limit number of prompts to process")
    return parser.parse_args()


def load_prompts(filepath, limit=None):
    """Load prompts from a text file (one per line)."""
    with open(filepath, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def find_final_norm(model):
    """Find the final layer norm before lm_head."""
    # Try common architecture patterns
    # Gemma 3 multimodal (Gemma3ForConditionalGeneration)
    if hasattr(model, 'model') and hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'norm'):
        return model.model.language_model.norm, "model.model.language_model.norm"
    # Standard Llama/Gemma/Mistral
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        return model.model.norm, "model.norm"
    # GPT-2 style
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        return model.transformer.ln_f, "transformer.ln_f"
    # Other patterns
    elif hasattr(model, 'model') and hasattr(model.model, 'final_layernorm'):
        return model.model.final_layernorm, "model.final_layernorm"
    return None, None


def process_batch(model, tokenizer, final_norm, batch_prompts, batch_indices, max_new_tokens):
    """Process a batch of prompts and return results and residuals."""
    batch_results = []
    batch_residuals = []

    # Format all prompts
    formatted_texts = []
    for prompt in batch_prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_texts.append(formatted)

    # Tokenize with padding for batched processing
    tokenizer.padding_side = "left"  # Left-pad for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        formatted_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    # Track input lengths (excluding padding) for each item
    attention_mask = inputs["attention_mask"]
    input_lengths = attention_mask.sum(dim=1).tolist()

    # ============================================
    # STEP 1: Capture residuals at decision point
    # ============================================
    captured_residuals = None

    def capture_hook(module, input, output):
        nonlocal captured_residuals
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # Get last non-padded position for each sequence
        residuals = []
        for i in range(hidden.shape[0]):
            last_pos = input_lengths[i] - 1
            residuals.append(hidden[i, last_pos, :].detach().cpu())
        captured_residuals = residuals

    hook_handle = final_norm.register_forward_hook(capture_hook)

    with torch.no_grad():
        _ = model(**inputs)

    hook_handle.remove()

    # ============================================
    # STEP 2: Generate responses
    # ============================================
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode each response
    for i, (prompt, idx) in enumerate(zip(batch_prompts, batch_indices)):
        # Find where the actual input ends (account for left padding)
        pad_length = (attention_mask[i] == 0).sum().item()
        input_end = inputs["input_ids"].shape[1]
        generated_ids = outputs[i, input_end:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        batch_results.append({
            "index": idx,
            "prompt": prompt,
            "response": response,
        })

        if captured_residuals is not None:
            batch_residuals.append({
                "index": idx,
                "prompt": prompt,
                "residual": captured_residuals[i],
            })

    return batch_results, batch_residuals


def main():
    args = parse_args()

    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Prompts: {args.prompts}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Limit: {args.limit or 'None'}")
    print(f"  Max tokens: {args.max_tokens}")
    print()

    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Find the final norm layer
    final_norm, final_norm_name = find_final_norm(model)
    if final_norm is None:
        print("Could not find final layer norm. Model structure:")
        print(model)
        raise ValueError("Please identify the final norm layer and update find_final_norm().")

    print(f"Found final norm: {final_norm_name}")

    # Load prompts
    prompts = load_prompts(args.prompts, limit=args.limit)
    print(f"Loaded {len(prompts)} prompts")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)

    results = []
    all_residuals = []

    # Process in batches
    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    print(f"\nProcessing {len(prompts)} prompts in {num_batches} batches...")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        batch_indices = list(range(start_idx, end_idx))

        print(f"\n[Batch {batch_idx+1}/{num_batches}] Processing {len(batch_prompts)} prompts...")
        for i, p in enumerate(batch_prompts):
            print(f"  {start_idx + i + 1}. {p[:60]}...")

        batch_results, batch_residuals = process_batch(
            model, tokenizer, final_norm,
            batch_prompts, batch_indices, args.max_tokens
        )

        results.extend(batch_results)
        all_residuals.extend(batch_residuals)

        # Show responses
        for r in batch_results:
            preview = r["response"][:80].replace('\n', ' ')
            print(f"  [{r['index']+1}] -> {preview}...")

    # Save results with shared run ID for correlation
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_{run_id}"

    # Save text responses
    responses_file = output_path / f"responses_{base_name}.json"
    json_output = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_path": args.model,
        "num_prompts": len(results),
        "results": results,
    }
    with open(responses_file, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved responses to {responses_file}")

    # Save residuals
    if all_residuals:
        residuals_file = output_path / f"residuals_{base_name}.pt"
        residual_tensor = torch.stack([r["residual"] for r in all_residuals])

        torch.save({
            "run_id": run_id,
            "timestamp": timestamp,
            "model_path": args.model,
            "capture_point": "last_prompt_token_before_generation",
            "responses_file": responses_file.name,
            "residuals": residual_tensor,  # Shape: [num_prompts, hidden_dim]
            "prompts": [r["prompt"] for r in all_residuals],
            "responses": [results[r["index"]]["response"] for r in all_residuals],
            "indices": [r["index"] for r in all_residuals],
        }, residuals_file)

        print(f"Saved residuals to {residuals_file}")
        print(f"Residual tensor shape: {residual_tensor.shape}")
        print(f"Run ID: {run_id} (shared between both files)")

    print("\nDone!")


if __name__ == "__main__":
    main()
