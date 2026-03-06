import os
import gc
import torch
import json
import re
from time import sleep
import tqdm
import csv

ROOT_PATH = "output/created_plans" #NOTE"{directory for generated plans}"

JUDGE_MODELS = [
    #NOTE--> ONLY open source models for evaluation (no trained models here to avoid bias)
    "meta-llama/Llama-3.2-3B-Instruct",
    "ibm-granite/granite-4.0-micro",
    "Qwen/Qwen3-1.7B",
    "google/gemma-2-2b-it"
]

CRITERIA = {
    "response in a well-structured JSON format": (0, 10),
    "mitigation plan can be implemented as that it is feasible": (0, 10),
    "mitigation plan is efficient as that it solves the root cause": (0, 10),
    "mitigation plan is scalable as that it can be applied for large-scale issues": (0, 10),
    "mitigation plan is clear for the operator as that it provides detailed steps and guidance": (0, 10),
}

# ===========================
# Helper: Initialize Model
# ===========================
def init_pipeline(model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "bitnet" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def build_prompt(plan_text, criterion, min_score, max_score):
    return (
        f"Here is a mitigation plan:\n{plan_text}\n\n"
        f"On a scale from {min_score} (low) to {max_score} (high), "
        f"rate how well the {criterion}. Provide only a numeric score."
    )


def evaluate_with_judges_sequential(generator, raw_plan, criteria):
    results = {}
    for criterion, (min_score, max_score) in criteria.items():
        plan_text = json.dumps(raw_plan, indent=2) if isinstance(raw_plan, (dict, list)) else str(raw_plan)
        prompt = build_prompt(plan_text, criterion, min_score, max_score)
        output = generator(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
        output_text = output[0]["generated_text"]
        match = re.findall(r"[-+]?\d*\.\d+|\d+", output_text)
        score = float(match[-1]) if match else 0.0
        score = max(min(score, max_score), min_score)
        results[criterion] = score
    return results


# ===========================
# Main Evaluation
# ===========================
if __name__ == "__main__":
    # folders for the plans that were generated from create_plans.py, each folder corresponds to a generator (open source or trained)
    folders = [
                #NOTE--> open source models for generation
                "google_gemma-2-2b-it",
                "ibm-granite_granite-4.0-micro",
                "meta-llama_Llama-3.2-3B-Instruct",
                "Qwen_Qwen3-1.7B"

                #--> include the trained models here for creating plans 
                #{model_name}  # placeholder for trained model name, e.g., "my-trained-model"

               ]
    
    output_file = "evaluation_results.csv"#NOTE {OUTPUT_CSV}.csv
    if not os.path.exists(output_file):
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            headers = ["Generator", "Judge"] + list(CRITERIA.keys()) + ["Overall Avg"]
            writer.writerow(headers)

    for judge_model in JUDGE_MODELS:
        print(f"\n Evaluating with Judge: {judge_model}")
        generator = init_pipeline(judge_model)

        for folder in tqdm.tqdm(folders, desc="Generators"):
            folder_path = os.path.join(ROOT_PATH, folder)
            json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
            all_scores = {criterion: [] for criterion in CRITERIA}

            for file in json_files:
                file_path = os.path.join(folder_path, file)
                with open(file_path, "r") as f:
                    data = json.load(f)

                # extract the mitigation text
                if "raw_text" in data.get("plan", {}):
                    clean_text = data["plan"]["raw_text"].split("\nMitigation Plan:\n")[-1].strip()
                else:
                    clean_text = str(data.get("plan", ""))

                per_criteria = evaluate_with_judges_sequential(generator, clean_text, CRITERIA)
                for c, score in per_criteria.items():
                    all_scores[c].append(score)

            # Average over all files in this folder
            avg_scores = {c: sum(v) / len(v) for c, v in all_scores.items()}
            overall = sum(avg_scores.values()) / len(avg_scores)

            # Append to CSV
            with open(output_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([folder, judge_model] + [f"{avg_scores[c]:.2f}" for c in CRITERIA] + [f"{overall:.2f}"])

        del generator
        torch.cuda.empty_cache()
        gc.collect()
        sleep(5)

    print(f"\n Done! Results saved to {output_file}")
