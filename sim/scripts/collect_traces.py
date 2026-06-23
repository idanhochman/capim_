"""
CAPIM Trace Collection Script

Downloads models and datasets, runs instrumented EAGLE-2 inference, and
saves a TraceDataset JSON for each evaluation scenario.

Requirements:
  - A CUDA GPU with at least 16 GB VRAM (FP16)
    OR 10 GB VRAM (with 4-bit quantization via bitsandbytes)
  - ~20 GB free disk space for models
  - The packages listed in requirements_collection.txt

Usage:
    # LLaMA-2-7B-Chat (default) — Alpaca traces
    python sim/scripts/collect_traces.py --dataset alpaca --n-prompts 200

    # Vicuna-7B EAGLE-2 (CAPIM side) — Alpaca traces
    python sim/scripts/collect_traces.py --model-family vicuna7b --dataset alpaca --n-prompts 200

    # Vicuna-7B MEDUSA (LP-Spec baseline) — SAME backbone, SAME prompts (symmetric)
    python sim/scripts/collect_traces.py --model-family vicuna7b --method medusa --dataset alpaca --n-prompts 200

    # Sanity check: 20 built-in prompts, no dataset download needed
    python sim/scripts/collect_traces.py --model-family vicuna7b --sanity
    python sim/scripts/collect_traces.py --model-family vicuna7b --method medusa --sanity

    # Dry run: check everything loads without running full inference
    python sim/scripts/collect_traces.py --dataset alpaca --n-prompts 3 --dry-run

Model families (set via --model-family):
    llama2    Target : meta-llama/Llama-2-7b-chat-hf          (~15 GB FP16)
              Draft  : yuhuili/EAGLE-llama2-chat-7B (~1 GB)
    vicuna7b  Target : lmsys/vicuna-7b-v1.3               (~13 GB FP16)
              Draft  : yuhuili/EAGLE-Vicuna-7B-v1.3        (~1 GB)
    llama2    Target : meta-llama/Llama-2-7b-chat-hf      (~13 GB FP16)
              Draft  : yuhuili/EAGLE-llama2-chat-7B        (~1 GB)
              NOTE: use this for direct LP-Spec baseline comparison

--base-model and --ea-model override the model-family defaults.

Output is namespaced by family + method + dataset (so EAGLE and MEDUSA traces
never clobber each other), e.g.:
    traces/vicuna7b_eagle_alpaca.json     traces/vicuna7b_medusa_alpaca.json
    traces/vicuna7b_eagle_gsm8k.json      traces/vicuna7b_medusa_gsm8k.json
"""

import argparse
import json
import os
import sys

# Add capim/ to path so sim.* imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
capim_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, capim_dir)

# Add the EAGLE + Medusa repos to path (eagle.* / medusa.* imports)
eagle_dir = os.path.join(capim_dir, "releted-repos", "EAGLE")
sys.path.insert(0, eagle_dir)
medusa_dir = os.path.join(capim_dir, "releted-repos", "Medusa")
sys.path.insert(0, medusa_dir)


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
# Model family configurations
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "llama2": {
        "base_model": "meta-llama/Llama-2-7b-chat-hf",
        "ea_model":   "yuhuili/EAGLE-llama2-chat-7B",
        "output_prefix": "llama2",
        "prompt_format": "chat_template",  # Llama-2-chat tokenizer has chat_template defined
    },
    "vicuna7b": {
        "base_model": "lmsys/vicuna-7b-v1.3",
        "ea_model":   "yuhuili/EAGLE-Vicuna-7B-v1.3",
        "medusa_model": "FasterDecoding/medusa-vicuna-7b-v1.3",  # base auto-loaded from head config
        "output_prefix": "vicuna7b",
        "prompt_format": "vicuna",         # manual format (no chat_template in tokenizer)
    },
}
# NOTE: only vicuna7b has BOTH an official EAGLE head and an official MEDUSA head
# on the SAME backbone -> the symmetric, both-measured comparison lives here.
# llama2-chat has no official MEDUSA head, so --method medusa is unavailable there.


def format_prompt(prompt: str, tokenizer, prompt_format: str) -> str:
    """Format a raw prompt string into model-specific input text."""
    if prompt_format == "chat_template":
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif prompt_format == "vicuna":
        # Standard Vicuna v1.3 conversation format (FastChat convention)
        return (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
            f"USER: {prompt}\nASSISTANT:"
        )
    else:
        raise ValueError(f"Unknown prompt_format: {prompt_format!r}")

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
        # Force safetensors (memory-mapped) over .bin to avoid the ~10GB CPU-RAM
        # spike that OOM-kills the base-model load on a 12.7GB T4.  EaModel
        # forwards **kwargs to the base-model from_pretrained.
        "use_safetensors": True,
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


def precision_label(load_in_4bit: bool = False, load_in_8bit: bool = False) -> str:
    """Canonical precision tag recorded in trace metadata for provenance.

    This labels HOW the model was run during collection, which shapes the
    acceptance/confidence statistics only -- the hardware cost model is
    INT8/W8A8 regardless of this value.  Note these bitsandbytes modes are NOT
    the simulator's W8A8 datapath: 8-bit is LLM.int8() (INT8 weight store +
    FP16 activations with an FP16 outlier path), 4-bit is NF4 dequantized to
    FP16 for compute.  Collect EAGLE and MEDUSA at the SAME precision so the
    comparison stays symmetric.
    """
    if load_in_4bit:
        return "nf4-4bit (bitsandbytes, FP16 compute)"
    if load_in_8bit:
        return "llm.int8()-8bit (bitsandbytes, FP16 compute)"
    return "fp16"


def load_medusa_model(
    medusa_model_id: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    """
    Load a MedusaModel from HuggingFace (LP-Spec baseline).

    Only the Medusa head repo is needed -- MedusaModel.from_pretrained reads the
    base model from the head's config and loads it automatically (e.g.
    FasterDecoding/medusa-vicuna-7b-v1.3 -> lmsys/vicuna-7b-v1.3).

    Returns:
        (medusa_model, tokenizer) tuple.
    """
    import torch
    from medusa.model.medusa_model import MedusaModel

    print(f"Loading Medusa model: {medusa_model_id}")

    kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        # Force safetensors: the base repo ships both .bin and .safetensors, and
        # .bin deserializes the whole 10GB shard into CPU RAM before quantization
        # (OOM-kills a 12.7GB T4). safetensors memory-maps straight to the GPU.
        "use_safetensors": True,
    }
    if load_in_4bit:
        kwargs["load_in_4bit"] = True
        print("Using 4-bit quantization")
    elif load_in_8bit:
        kwargs["load_in_8bit"] = True
        print("Using 8-bit quantization")

    model = MedusaModel.from_pretrained(medusa_model_id, **kwargs)
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
    prompt_format: str = "chat_template",
    model_draft_name: str = "EAGLE2",
    max_new_tokens: int = 200,
    precision: str = "fp16",
    method: str = "eagle",
    dry_run: bool = False,
) -> "TraceDataset":
    """
    Run EAGLE-2 inference with the CAPIM collector attached.

    Returns a TraceDataset with per-token confidence scores logged.
    """
    import torch
    from sim.trace.schema import TraceDataset, DecodeStepTrace
    if method == "medusa":
        from sim.trace.medusa_collector import MedusaCollector
    else:
        from sim.trace.eagle_collector import Collector

    if dry_run:
        prompts = prompts[:2]
        max_new_tokens = 20
        print(f"DRY RUN: using {len(prompts)} prompts, {max_new_tokens} tokens each")

    all_steps = []
    total = len(prompts)

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{total}] Prompt: {prompt[:60]}...")

        if method == "medusa":
            collector = MedusaCollector(
                dataset=dataset,
                prompt_id=i,
                model_target=model.base_model_name_or_path,
                model_draft=model_draft_name,
                precision=precision,
            )
        else:
            collector = Collector(
                dataset=dataset,
                prompt_id=i,
                model_target=model.base_model_name_or_path,
                model_draft=model_draft_name,
            )
        collector.attach(model)

        try:
            text = format_prompt(prompt, tokenizer, prompt_format)
            input_ids = tokenizer([text], return_tensors="pt").input_ids.to(
                model.base_model.device
            )

            with torch.no_grad():
                if method == "medusa":
                    # medusa_generate is a GENERATOR -> iterate it to drive the
                    # decode loop (which triggers the collector's hooks).  max_steps
                    # bounds decode STEPS (each yields >=1 token), the MEDUSA analog
                    # of eagenerate's max_new_tokens token budget.
                    for _ in model.medusa_generate(
                        input_ids,
                        temperature=0.0,
                        max_steps=max_new_tokens,
                    ):
                        pass
                else:
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
                step.sample_token_str = tokenizer.decode([step.sample_token_id])
                for node in step.nodes:
                    node.token_str = tokenizer.decode([node.token_id])
            all_steps.extend(partial.steps)
            print(f"    Collected {len(partial.steps)} steps "
                  f"(total so far: {len(all_steps)})")

    td = TraceDataset(
        steps=all_steps,
        model_target=model.base_model_name_or_path,
        model_draft=model_draft_name,
        metadata={
            "dataset": dataset,
            "n_prompts": len(prompts),
            "max_new_tokens": max_new_tokens,
            "precision": precision,
            "sd_method": method,
        },
    )
    td.compute_summary()
    return td


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect CAPIM traces from EAGLE-2 (CAPIM) or MEDUSA (LP-Spec) inference"
    )
    parser.add_argument(
        "--model-family",
        choices=list(MODEL_CONFIGS.keys()),
        default="llama2",
        help="Model family shorthand (sets base-model, ea-model, output prefix, prompt format). "
             "Overridden by --base-model / --ea-model if provided. (default: llama2)",
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
        default=None,
        help="HuggingFace repo ID or local path for the target model "
             "(overrides --model-family default)",
    )
    parser.add_argument(
        "--ea-model",
        type=str,
        default=None,
        help="HuggingFace repo ID or local path for the EAGLE draft model "
             "(overrides --model-family default; used only with --method eagle)",
    )
    parser.add_argument(
        "--method",
        choices=["eagle", "medusa"],
        default="eagle",
        help="SD method to run/instrument: 'eagle' (CAPIM/EAGLE-2 head) or "
             "'medusa' (LP-Spec baseline; needs a model-family with a medusa "
             "head, e.g. vicuna7b). (default: eagle)",
    )
    parser.add_argument(
        "--medusa-model",
        type=str,
        default=None,
        help="HuggingFace repo ID or local path for the MEDUSA head "
             "(overrides --model-family default; used only with --method medusa)",
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

    # Resolve model config: model-family sets defaults; explicit args override
    cfg = MODEL_CONFIGS[args.model_family]
    base_model_id  = args.base_model  or cfg["base_model"]
    output_prefix  = cfg["output_prefix"]
    prompt_format  = cfg["prompt_format"]

    if args.method == "medusa":
        medusa_model_id = args.medusa_model or cfg.get("medusa_model")
        if not medusa_model_id:
            parser.error(
                f"--method medusa needs a MEDUSA head, but model-family "
                f"'{args.model_family}' has none. Use --model-family vicuna7b "
                f"or pass --medusa-model."
            )
        model_draft_name = medusa_model_id.split("/")[-1]  # e.g. "medusa-vicuna-7b-v1.3"
    else:
        ea_model_id = args.ea_model or cfg["ea_model"]
        model_draft_name = ea_model_id.split("/")[-1]       # e.g. "EAGLE-Vicuna-7B-v1.3"

    # Resolve prompts, dataset label, output path, and token budget
    os.makedirs(args.output_dir, exist_ok=True)

    if args.sanity:
        dataset_label = "sanity"
        prompts = SANITY_PROMPTS
        max_new_tokens = args.max_new_tokens if args.max_new_tokens != 200 else 100
        output_path = os.path.join(args.output_dir, f"{output_prefix}_{args.method}_sanity.json")
        print(f"\n=== Step 1: Using {len(prompts)} built-in sanity prompts ===")
        print("(No dataset download required)")
    else:
        dataset_label = args.dataset
        max_new_tokens = args.max_new_tokens
        output_path = os.path.join(args.output_dir, f"{output_prefix}_{args.method}_{args.dataset}.json")
        print(f"\n=== Step 1: Loading {args.dataset} prompts ===")
        loader = DATASET_LOADERS[args.dataset]
        prompts = loader(args.n_prompts)

    # Load model
    precision = precision_label(args.load_in_4bit, args.load_in_8bit)
    print(f"\n=== Step 2: Loading {args.method.upper()} model (precision: {precision}) ===")
    if args.method == "medusa":
        model, tokenizer = load_medusa_model(
            medusa_model_id=medusa_model_id,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )
    else:
        model, tokenizer = load_eagle_model(
            base_model_id=base_model_id,
            ea_model_id=ea_model_id,
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
        prompt_format=prompt_format,
        model_draft_name=model_draft_name,
        max_new_tokens=max_new_tokens,
        precision=precision,
        method=args.method,
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
