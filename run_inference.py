def run_inference():
    import json
    import os

    # Configuration
    MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
    GPU_ID      = "0"                    # CUDA_VISIBLE_DEVICES
    DATA_PATH   = "private.jsonl"
    OUTPUT_PATH = "submission.csv"
    MAX_TOKENS  = 16384

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

    import re
    import sys
    from pathlib import Path
    from typing import Optional

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from tqdm import tqdm

    # Load data
    data = [json.loads(line) for line in open(DATA_PATH)]

    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options")   for d in data)
    print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    # Preview one MCQ and one free-form item
    mcq_sample  = next(d for d in data if d.get("options"))
    free_sample = next(d for d in data if not d.get("options"))

    print("\n── MCQ sample ──")
    print(json.dumps(mcq_sample, indent=2))
    print("\n── Free-form sample ──")
    print(json.dumps(free_sample, indent=2))

    # Prompt Construction
    SYSTEM_PROMPT_MATH = (
        "You are an expert mathematician participating in a highly rigorous math competition. "
        "Solve the problem step-by-step. You must first write out a detailed reasoning trace. "
        "After completing your reasoning, place your final answer inside \\boxed{}. "
        "CRITICAL: If the problem asks for multiple answers (indicated by multiple [ANS] placeholders), "
        "you MUST calculate all of them and separate them by commas inside a SINGLE \\boxed{}, "
        "in the exact order they were requested. For example: \\boxed{3, 7, 42}."
    )

    SYSTEM_PROMPT_MCQ = (
        "You are an expert mathematician. Read the problem and the provided answer choices carefully. "
        "Think step-by-step to derive the correct solution, then compare your solution to the options. "
        "If your calculated answer does not perfectly match any option, you must choose the option that "
        "is mathematically closest or most logically intended. "
        "CRITICAL: You must conclude your response by outputting ONLY the single letter of your chosen option "
        "inside \\boxed{}, for example: \\boxed{C}. Never leave a response without a boxed letter, regardless of your confidence."
    )

    def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) for a question."""
        if options:
            labels    = [chr(65 + i) for i in range(len(options))]
            opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
            return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
        return SYSTEM_PROMPT_MATH, question


    # Verify with samples
    for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
        sys_p, usr_p = build_prompt(item["question"], item.get("options"))
        print(f"── {label} user prompt (first 200 chars) ──")
        print(usr_p[:200], "...\n")
    
    # Load model with vLLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
        max_model_len=MAX_TOKENS,
        trust_remote_code=True,
    )

    # 1. Update Sampling Params to generate multiple outputs per prompt
    SC_SAMPLES = 5

    sampling_params = SamplingParams(
        n=SC_SAMPLES,              
        max_tokens=MAX_TOKENS,
        temperature=0.5,
        top_p=0.95,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    print("Model loaded.")

    # Generate with vLLM
    import csv

    # 1. Initialize the CSV with headers (Write mode)
    OUTPUT_CSV_PATH = OUTPUT_PATH
    out_path = Path(OUTPUT_CSV_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])

    import re
    from collections import Counter
    from pathlib import Path

    # Build prompts for the dataset
    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        
        # --- Free-Form Few Shot ---
        few_shot_free_user = "If $f(x)=4x^2+x+2$, find the following:\n(a) $f(3)=$ [ANS]\n(b) $f(-3)=$ [ANS]"
        few_shot_free_assistant = (
            "Let's break this down step-by-step.\n"
            "First, for (a):\n$f(3) = 4(3)^2 + 3 + 2 = 36 + 5 = 41$\n\n"
            "Second, for (b):\n$f(-3) = 4(-3)^2 + (-3) + 2 = 36 - 1 = 35$\n\n"
            "The problem requests multiple answers. I will format them in a single boxed array.\n"
            "The final answer is \\boxed{41, 35}."
        )

        # --- MCQ Few Shot ---
        few_shot_mcq_user = (
            "Evaluate the integral $\\int 2x \\, dx$.\n\n"
            "Options:\n"
            "A. $x^2 + C$\n"
            "B. $2x^2 + C$\n"
            "C. $\\frac{1}{2}x^2 + C$\n"
            "D. $2 + C$"
        )
        few_shot_mcq_assistant = (
            "Let's think step-by-step.\n"
            "We need to find the indefinite integral of the function $f(x) = 2x$.\n"
            "Using the power rule for integration, $\\int x^n \\, dx = \\frac{x^{n+1}}{n+1} + C$.\n"
            "Therefore, $\\int 2x \\, dx = 2 \\left( \\frac{x^2}{2} \\right) + C = x^2 + C$.\n\n"
            "Now, I will match this result with the given options.\n"
            "Option A is $x^2 + C$. This matches our calculated result exactly.\n"
            "Option B, C, and D are incorrect.\n"
            "Therefore, the correct option is A.\n"
            "The final answer is \\boxed{A}."
        )

        # Build the message history dynamically based on question type
        messages = [{"role": "system", "content": system}]
        
        if not item.get("options"):
            # Inject Free-Form Example
            messages.append({"role": "user", "content": few_shot_free_user})
            messages.append({"role": "assistant", "content": few_shot_free_assistant})
        else:
            # Inject MCQ Example
            messages.append({"role": "user", "content": few_shot_mcq_user})
            messages.append({"role": "assistant", "content": few_shot_mcq_assistant})
            
        messages.append({"role": "user", "content": user})

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    # 2. Generate
    print(f"Generating {SC_SAMPLES} responses per question for {len(prompts)} questions...")

    # 2. Process in chunks
    CHUNK_SIZE = 50  # Process 50 questions at a time

    print(f"Starting generation for {len(prompts)} prompts in chunks of {CHUNK_SIZE}...")

    # Iterate through the prompts in steps of CHUNK_SIZE
    for i in range(0, len(prompts), CHUNK_SIZE):
        chunk_prompts = prompts[i : i + CHUNK_SIZE]
        chunk_data = data[i : i + CHUNK_SIZE] # Keep the IDs aligned!
        
        # Generate for this specific chunk
        outputs = llm.generate(chunk_prompts, sampling_params=sampling_params)
        
        # PRIVATE DATASET: Open CSV in APPEND mode ("a") to add the new rows without overwriting
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            
            # Process Majority Voting for the chunk
            for item, out in zip(chunk_data, outputs):
                candidate_texts = [o.text.strip() for o in out.outputs]
                
                candidate_answers = []
                for text in candidate_texts:
                    match = re.search(r"\\boxed\{(.*?)\}", text, flags=re.DOTALL)
                    if match:
                        candidate_answers.append(match.group(1).strip())
                
                if candidate_answers:
                    majority_answer = Counter(candidate_answers).most_common(1)[0][0]
                    final_full_text = next((t for t in candidate_texts if f"\\boxed{{{majority_answer}}}" in t), candidate_texts[0])
                else:
                    final_full_text = candidate_texts[0]
                
                # Write the row immediately to the disk
                writer.writerow([item["id"], final_full_text])
                
        print(f"Saved chunk. Processed {min(i + CHUNK_SIZE, len(prompts))}/{len(prompts)} questions.")

if __name__ == "__main__":
    run_inference()