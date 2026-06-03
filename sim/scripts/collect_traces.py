"""
CAPIM Trace Collection Script

Downloads models and datasets, runs instrumented EAGLE-2 inference, and
saves a TraceDataset JSON for each evaluation scenario.

Requirements:
  - A CUDA GPU with at least 16 GB VRAM (for Qwen2.5-7B in FP16)
    OR 10 GB VRAM (with 4-bit quantization via bitsandbytes)
  - ~20 GB free disk space for models
  - The packages listed in requirements_collection.txt

Usage:
    # Sanity check: 20 built-in prompts, no dataset download needed (~30 min on T4)
    python sim/scripts/collect_traces.py --sanity

    # Collect Alpaca traces (matches LP-Spec evaluation conditions)
    python sim/scripts/collect_traces.py --dataset alpaca --n-prompts 200

    # Collect GSM8K traces (math reasoning)
    python sim/scripts/collect_traces.py --dataset gsm8k --n-prompts 200

    # Dry run: check everything loads without running full inference
    python sim/scripts/collect_traces.py --dataset alpaca --n-prompts 3 --dry-run

Models downloaded automatically from HuggingFace on first run:
    Target : Qwen/Qwen2.5-7B-Instruct          (~15 GB FP16, ~8 GB INT8/4-bit)
    Draft  : yuhuili/EAGLE2-Qwen2.5-7B-Instruct (~1 GB)

Output:
    traces/qwen25_sanity.json   (--sanity)
    traces/qwen25_alpaca.json   (--dataset alpaca)
    traces/qwen25_gsm8k.json    (--dataset gsm8k)
"""

import argparse
import json
import os
import sys

# Add capim/ to path so sim.* imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
capim_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, capim_dir)

# Add the EAGLE repo to path
eagle_dir = os.path.join(capim_dir, "releted-repos", "EAGLE")
sys.path.insert(0, eagle_dir)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_alpaca_prompts(n: int) -> list[str]:
    """
    Load prompts from the Stanford Alpaca dataset.
    Downloads automatically via HuggingFace datasets.
    Returns instruction strings (no input field for simplicity).
    """
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = []
    for row in ds:
        if len(prompts) >= n:
            break
        instruction = row["instruction"].strip()
        inp = row.get("input", "").strip()
        if inp:
            prompt = f"{instruction}\n\n{inp}"
        else:
            prompt = instruction
        if prompt:
            prompts.append(prompt)
    print(f"Loaded {len(prompts)} Alpaca prompts")
    return prompts


def load_gsm8k_prompts(n: int) -> list[str]:
    """
    Load math word problems from GSM8K.
    Downloads automatically via HuggingFace datasets.
    """
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    prompts = [row["question"].strip() for row in ds][:n]
    print(f"Loaded {len(prompts)} GSM8K prompts")
    return prompts


DATASET_LOADERS = {
    "alpaca": load_alpaca_prompts,
    "gsm8k": load_gsm8k_prompts,
}

# ---------------------------------------------------------------------------
# Sanity prompts (no dataset download required)
# ---------------------------------------------------------------------------

# 20 prompts spanning factual, instruction, math, code, and creative tasks.
# Diversity in expected confidence is intentional: factual/formulaic prompts
# should produce high-confidence draft tokens; open-ended creative prompts
# should produce lower-confidence ones. This spread is what makes the sanity
# run useful for validating the confidence-acceptance correlation.
SANITY_PROMPTS = [
    # Factual / high confidence
    "What is the capital of France?",
    "List the planets of the solar system in order from the Sun.",
    "Translate 'Hello, how are you?' into Spanish.",
    "What is the chemical formula for water?",
    "Name the four seasons in order, starting from spring.",
    # Instruction / medium confidence
    "Write a Python function that returns the factorial of a non-negative integer n.",
    "Explain the difference between supervised and unsupervised learning in two sentences.",
    "What are the three laws of motion formulated by Isaac Newton?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "How does photosynthesis work? Give a brief explanation.",
    # Math / reasoning
    "If a train travels at 60 mph for 2.5 hours, how far does it travel?",
    "What is the sum of the first 100 positive integers?",
    "A store sells apples for $0.50 each and oranges for $0.75 each. If I buy 3 apples and 4 oranges, how much do I spend in total?",
    "Explain why 0.999... (repeating) equals 1.",
    "A rectangle has a length of 8 cm and a width of 5 cm. What is its area and perimeter?",
    # Creative / open-ended / lower confidence
    "Write a haiku about a rainy autumn evening.",
    "Write a short story (3-4 sentences) about a robot who discovers it can dream.",
    "Describe the colour blue to someone who has been blind from birth.",
    "If you could add one subject to every school's curriculum, what would it be and why?",
    "What might everyday life look like if humans required 12 hours of sleep per night?",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_eagle_model(
    base_model_id: str,
    ea_model_id: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    device: str = "cuda",
):
    """
    Load the EAGLE-2 EaModel from HuggingFace.

    Args:
        base_model_id: HuggingFace repo for the target model.
        ea_model_id: HuggingFace repo for the EAGLE draft model.
        load_in_4bit: Use bitsandbytes 4-bit quantization (needs bnb).
        load_in_8bit: Use bitsandbytes 8-bit quantization (needs bnb).
        device: "cuda" or "cpu" (cpu is very slow, for testing only).

    Returns:
        (ea_model, tokenizer) tuple.
    """
    import torch
    from eagle.model.ea_model import EaModel

    print(f"Loading base model: {base_model_id}")
    print(f"Loading EAGLE model: {ea_model_id}")

    kwargs = {
        "base_model_path": base_model_id,
        "ea_model_path": ea_model_id,
        "total_token": 60,       # draft tree size (matches EAGLE-2 paper default)
        "depth": 5,              # max draft depth
        "top_k": 10,             # top-k per draft step
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "use_eagle3": False,     # use EAGLE-2 (not EAGLE-3)
    }

    if load_in_4bit:
        kwargs["load_in_4bit"] = True
        print("Using 4-bit quantization")
    elif load_in_8bit:
        kwargs["load_in_8bit"] = True
        print("Using 8-bit quantization")

    model = EaModel.from_pretrained(**kwargs)
    model.eval()
    tokenizer = model.get_tokenizer()
    print("Model loaded successfully")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Instrumented inference loop
# ---------------------------------------------------------------------------

def run_collection(
    model,
    tokenizer,
    prompts: list[str],
    dataset: str,
    max_new_tokens: int = 200,
    dry_run: bool = False,
) -> "TraceDataset":
    """
    Run EAGLE-2 inference with the CAPIM collector attached.

    Returns a TraceDataset with per-token confidence scores logged.
    """
    import torch
    from sim.trace.collector import Collector
    from sim.trace.schema import TraceDataset, DecodeStepTrace

    if dry_run:
        prompts = prompts[:2]
        max_new_tokens = 20
        print(f"DRY RUN: using {len(prompts)} prompts, {max_new_tokens} tokens each")

    all_steps = []
    total = len(prompts)

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{total}] Prompt: {prompt[:60]}...")

        collector = Collector(
            dataset=dataset,
            prompt_id=i,
            model_target=model.base_model_name_or_path,
            model_draft="EAGLE2",
        )
        collector.attach(model)

        try:
            # Format with chat template for instruction-tuned models
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer([text], return_tensors="pt").input_ids.to(
                model.base_model.device
            )

            with torch.no_grad():
                model.eagenerate(
                    input_ids,
                    is_llama3=False,
                    temperature=0,
                    top_p=0,
                    top_k=0,
                    max_new_tokens=max_new_tokens,
                )
        except Exception as e:
            print(f"    WARNING: inference failed: {e}")
        finally:
            partial = collector.detach()
            for step in partial.steps:
                for node in step.nodes:
                    node.token_str = tokenizer.decode([node.token_id])
            all_steps.extend(partial.steps)
            print(f"    Collected {len(partial.steps)} steps "
                  f"(total so far: {len(all_steps)})")

    td = TraceDataset(
        steps=all_steps,
        model_target=model.base_model_name_or_path,
        model_draft="EAGLE2-Qwen2.5-7B-Instruct",
        metadata={
            "dataset": dataset,
            "n_prompts": len(prompts),
            "max_new_tokens": max_new_tokens,
        },
    )
    td.compute_summary()
    return td


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect CAPIM traces from EAGLE-2 inference"
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help=(
            "Run a quick sanity check using 20 built-in prompts. "
            "No dataset download required. Useful on Colab to validate the "
            "full pipeline before committing to a long collection run."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["alpaca", "gsm8k"],
        default=None,
        help="Dataset to use for prompts (required unless --sanity is set)",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=200,
        help="Number of prompts to run (default: 200; ignored with --sanity)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace repo ID or local path for the target model",
    )
    parser.add_argument(
        "--ea-model",
        type=str,
        default="yuhuili/EAGLE2-Qwen2.5-7B-Instruct",
        help="HuggingFace repo ID or local path for the EAGLE-2 draft model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="traces",
        help="Directory to save trace JSON files",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Max tokens to generate per prompt (default: 200; --sanity uses 100)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load base model in 4-bit (requires bitsandbytes, needs ~10GB VRAM)",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load base model in 8-bit (requires bitsandbytes, needs ~12GB VRAM)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 2 prompts to verify the pipeline works end-to-end",
    )
    args = parser.parse_args()

    if not args.sanity and args.dataset is None:
        parser.error("--dataset is required unless --sanity is set")

    # Resolve prompts, dataset label, output path, and token budget
    os.makedirs(args.output_dir, exist_ok=True)

    if args.sanity:
        dataset_label = "sanity"
        prompts = SANITY_PROMPTS
        max_new_tokens = args.max_new_tokens if args.max_new_tokens != 200 else 100
        output_path = os.path.join(args.output_dir, "qwen25_sanity.json")
        print(f"\n=== Step 1: Using {len(prompts)} built-in sanity prompts ===")
        print("(No dataset download required)")
    else:
        dataset_label = args.dataset
        max_new_tokens = args.max_new_tokens
        output_path = os.path.join(args.output_dir, f"qwen25_{args.dataset}.json")
        print(f"\n=== Step 1: Loading {args.dataset} prompts ===")
        loader = DATASET_LOADERS[args.dataset]
        prompts = loader(args.n_prompts)

    # Load model
    print(f"\n=== Step 2: Loading EAGLE-2 model ===")
    model, tokenizer = load_eagle_model(
        base_model_id=args.base_model,
        ea_model_id=args.ea_model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    # Collect traces
    print(f"\n=== Step 3: Collecting traces ({len(prompts)} prompts) ===")
    trace = run_collection(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        dataset=dataset_label,
        max_new_tokens=max_new_tokens,
        dry_run=args.dry_run,
    )

    # Save
    print(f"\n=== Step 4: Saving trace ===")
    trace.save(output_path)
    print(f"Saved {len(trace.steps)} steps to {output_path}")
    print(f"\nSummary:")
    print(f"  Mean tree size        : {trace.mean_tree_size:.1f} tokens")
    print(f"  Mean accepted/step    : {trace.mean_accepted_length:.2f} tokens")
    print(f"  Mean acceptance rate  : {trace.mean_acceptance_rate*100:.1f}%")

    if args.sanity and not args.dry_run:
        print(f"\nSanity trace saved. Next steps:")
        print(f"  1. Run the simulation:  python sim/scripts/run_simulation.py --trace {output_path}")
        print(f"  2. If results look sensible, re-run with --dataset alpaca/gsm8k for full collection.")
    elif not args.dry_run:
        print(f"\nNext step: run the CAPIM simulation with:")
        print(f"  python sim/scripts/run_simulation.py --trace {output_path}")


if __name__ == "__main__":
    main()
