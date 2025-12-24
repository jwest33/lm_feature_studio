"""
Category-based Prompt Generator using Qwen3-VL via llama.cpp server

Generates N samples per category with different seeds for variety.
Manages llama-server lifecycle (starts if not running, optionally shuts down after).
"""

import argparse
import json
import subprocess
import sys
import time
import socket
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional


# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = Path(r"D:\huggingface\qwen3-vl-8b-instruct-null-space-abliterated-GGUF\qwen3-vl-8b-instruct-null-space-abliterated-q8_0.gguf")
MMPROJ_PATH = Path(r"D:\huggingface\qwen3-vl-8b-instruct-null-space-abliterated-GGUF\mmproj-qwen3-vl-8b-instruct-f16.gguf")

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# Generation defaults
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40
DEFAULT_MAX_TOKENS = 512
DEFAULT_SAMPLES_PER_CATEGORY = 5

# Server startup
SERVER_STARTUP_TIMEOUT = 120  # seconds to wait for server to become ready
SERVER_HEALTH_CHECK_INTERVAL = 2  # seconds between health checks


# =============================================================================
# Server Management
# =============================================================================

def is_port_in_use(host: str, port: int) -> bool:
    """Check if a port is currently in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        return result == 0


def is_server_healthy(url: str) -> bool:
    """Check if llama-server is responding to health checks."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def find_server_executable() -> Optional[str]:
    """Find llama-server executable (assumes it's on PATH)."""
    import shutil
    if shutil.which("llama-server"):
        return "llama-server"
    return None


def start_server(
    n_gpu_layers: int = -1,
    ctx_size: int = 8192,
    n_parallel: int = 1
) -> Optional[subprocess.Popen]:
    """
    Start llama-server if not already running.
    
    Returns:
        Popen object if server was started, None if already running.
    """
    # Check if server is already running
    if is_server_healthy(SERVER_URL):
        print(f"[INFO] llama-server already running at {SERVER_URL}")
        return None
    
    if is_port_in_use(SERVER_HOST, SERVER_PORT):
        print(f"[WARNING] Port {SERVER_PORT} is in use but server not responding to health checks")
        print("[WARNING] Attempting to use existing server anyway...")
        return None
    
    # Find executable
    server_exe = find_server_executable()
    if server_exe is None:
        print("[ERROR] Could not find llama-server on PATH")
        print("[ERROR] Make sure llama-server is installed and available in your PATH")
        sys.exit(1)
    
    # Validate model files exist
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        sys.exit(1)
    
    if not MMPROJ_PATH.exists():
        print(f"[ERROR] mmproj file not found: {MMPROJ_PATH}")
        sys.exit(1)
    
    # Build command
    cmd = [
        server_exe,
        "--model", str(MODEL_PATH),
        "--mmproj", str(MMPROJ_PATH),
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--n-gpu-layers", str(n_gpu_layers),
        "--ctx-size", str(ctx_size),
        "--parallel", str(n_parallel),
    ]
    
    print(f"[INFO] Starting llama-server...")
    print(f"[INFO] Command: {' '.join(cmd)}")
    
    # Start server process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # Windows-specific
    )
    
    # Wait for server to become ready
    print(f"[INFO] Waiting for server to start (timeout: {SERVER_STARTUP_TIMEOUT}s)...")
    start_time = time.time()
    
    while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
        if process.poll() is not None:
            # Process exited unexpectedly
            stdout, _ = process.communicate()
            print(f"[ERROR] Server process exited with code {process.returncode}")
            print(f"[ERROR] Output: {stdout}")
            sys.exit(1)
        
        if is_server_healthy(SERVER_URL):
            elapsed = time.time() - start_time
            print(f"[INFO] Server is ready! (took {elapsed:.1f}s)")
            return process
        
        time.sleep(SERVER_HEALTH_CHECK_INTERVAL)
    
    # Timeout
    print(f"[ERROR] Server failed to start within {SERVER_STARTUP_TIMEOUT} seconds")
    process.terminate()
    sys.exit(1)


def stop_server(process: Optional[subprocess.Popen]) -> None:
    """Stop the server if we started it."""
    if process is None:
        return
    
    print("[INFO] Shutting down llama-server...")
    process.terminate()
    try:
        process.wait(timeout=10)
        print("[INFO] Server stopped gracefully")
    except subprocess.TimeoutExpired:
        print("[WARNING] Server did not stop gracefully, killing...")
        process.kill()
        process.wait()


# =============================================================================
# Generation
# =============================================================================

def generate_completion(
    prompt: str,
    seed: int,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """
    Generate a completion using the llama-server API.
    
    Returns:
        dict with 'success', 'output', 'seed', and optionally 'error' keys.
    """
    payload = {
        "prompt": prompt,
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "n_predict": max_tokens,
        "stream": False,
    }
    
    try:
        response = requests.post(
            f"{SERVER_URL}/completion",
            json=payload,
            timeout=120  # 2 minute timeout per generation
        )
        response.raise_for_status()
        
        data = response.json()
        return {
            "success": True,
            "seed": seed,
            "output": data.get("content", ""),
            "tokens_predicted": data.get("tokens_predicted", 0),
            "tokens_evaluated": data.get("tokens_evaluated", 0),
        }
    
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "seed": seed,
            "output": "",
            "error": str(e),
        }


def build_prompt(category: str, instruction: str) -> str:
    """
    Build a prompt for generating samples in a specific category.
    
    Uses Qwen3 chat format.
    """
    system_message = (
        "You are a helpful assistant that generates diverse, high-quality examples. "
        "Generate creative and varied content based on the user's request."
    )
    
    user_message = f"{instruction}\n\nCategory/Domain: {category}"
    
    # Qwen3 chat format
    prompt = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    return prompt


def generate_for_categories(
    categories: list[str],
    instruction: str,
    samples_per_category: int,
    base_seed: int = 1,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """
    Generate samples for all categories.
    
    Returns:
        dict mapping category names to lists of generation results.
    """
    results = {}
    total_samples = len(categories) * samples_per_category
    current_sample = 0
    
    for category in categories:
        print(f"\n[INFO] Generating {samples_per_category} samples for category: {category}")
        results[category] = []
        
        prompt = build_prompt(category, instruction)
        
        for i in range(samples_per_category):
            seed = base_seed + i
            current_sample += 1
            
            print(f"  [{current_sample}/{total_samples}] Seed {seed}...", end=" ", flush=True)
            
            result = generate_completion(
                prompt=prompt,
                seed=seed,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )
            
            if result["success"]:
                print(f"OK ({result['tokens_predicted']} tokens)")
            else:
                print(f"FAILED: {result.get('error', 'Unknown error')}")
            
            results[category].append(result)
    
    return results


# =============================================================================
# Output
# =============================================================================

def save_results(results: dict, output_path: Path, instruction: str, config: dict) -> None:
    """Save generation results to a JSON file."""
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "instruction": instruction,
            "config": config,
            "total_samples": sum(len(samples) for samples in results.values()),
            "successful_samples": sum(
                sum(1 for s in samples if s["success"])
                for samples in results.values()
            ),
        },
        "results": results,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[INFO] Results saved to: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate prompt examples across different knowledge domains using Qwen3-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5 samples each for physics, biology, and history
  python category_prompt_generator.py -c physics biology history -n 5 \\
      -i "Generate a challenging exam question with its answer"

  # Generate 10 samples per category with custom output file
  python category_prompt_generator.py -c math chemistry -n 10 \\
      -i "Create a real-world problem that requires this domain knowledge" \\
      -o my_prompts.json

  # Use specific generation parameters
  python category_prompt_generator.py -c programming -n 3 \\
      -i "Write a coding challenge" \\
      --temperature 1.0 --max-tokens 1024
        """
    )
    
    parser.add_argument(
        "-c", "--categories",
        nargs="+",
        required=True,
        help="List of categories/domains to generate samples for"
    )
    
    parser.add_argument(
        "-n", "--samples",
        type=int,
        default=DEFAULT_SAMPLES_PER_CATEGORY,
        help=f"Number of samples to generate per category (default: {DEFAULT_SAMPLES_PER_CATEGORY})"
    )
    
    parser.add_argument(
        "-i", "--instruction",
        required=True,
        help="Instruction/prompt template for generation"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("generated_samples.json"),
        help="Output JSON file path (default: generated_samples.json)"
    )
    
    parser.add_argument(
        "--base-seed",
        type=int,
        default=1,
        help="Starting seed value (default: 1)"
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
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})"
    )
    
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=8192,
        help="Context size for the server (default: 8192)"
    )
    
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 for all, default: -1)"
    )
    
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Keep the server running after generation completes"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Category-based Prompt Generator")
    print("=" * 60)
    print(f"Categories: {', '.join(args.categories)}")
    print(f"Samples per category: {args.samples}")
    print(f"Total samples to generate: {len(args.categories) * args.samples}")
    print(f"Instruction: {args.instruction}")
    print(f"Output file: {args.output}")
    print("=" * 60)
    
    # Start server if needed
    server_process = start_server(
        n_gpu_layers=args.n_gpu_layers,
        ctx_size=args.ctx_size,
    )
    
    we_started_server = server_process is not None
    
    try:
        # Generate samples
        results = generate_for_categories(
            categories=args.categories,
            instruction=args.instruction,
            samples_per_category=args.samples,
            base_seed=args.base_seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
        )
        
        # Save results
        config = {
            "base_seed": args.base_seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "model": str(MODEL_PATH),
            "samples_per_category": args.samples,
        }
        
        save_results(results, args.output, args.instruction, config)
        
        # Summary
        total = sum(len(samples) for samples in results.values())
        successful = sum(
            sum(1 for s in samples if s["success"])
            for samples in results.values()
        )
        
        print("\n" + "=" * 60)
        print("Generation Complete!")
        print(f"Total: {successful}/{total} samples generated successfully")
        print("=" * 60)
        
    finally:
        # Cleanup
        if we_started_server and not args.keep_server:
            stop_server(server_process)
        elif we_started_server and args.keep_server:
            print(f"\n[INFO] Server left running at {SERVER_URL}")
            print("[INFO] To stop it manually, press Ctrl+C in the server window or kill the process")


if __name__ == "__main__":
    main()
