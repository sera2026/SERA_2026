#--------------------------------------------------
#------------batch preprocessing-------------------
#--------------------------------------------------
import argparse
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from datasets import Dataset
from sklearn.model_selection import train_test_split
from textstat import flesch_reading_ease
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, GRPOTrainer
import torch
from torch.utils.tensorboard import SummaryWriter  # TensorBoard logging
import os
from datetime import datetime
import json
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig
from peft import prepare_model_for_kbit_training
from collections import defaultdict

# ==================================================
# 1. Reward Function
# ==================================================
REWARD_CONFIGS = {
    "balanced": {
        "json_structure": 0.25,
        "efficiency": 0.25,
        "scalability": 0.25,
        "clarity": 0.25,
    },
    "efficiency_heavy": {
        "json_structure": 0.15,
        "efficiency": 0.50,
        "scalability": 0.20,
        "clarity": 0.15,
    },
    "structure_heavy": {
        "json_structure": 0.40,
        "efficiency": 0.20,
        "scalability": 0.10,
        "clarity": 0.30,
    },
    "efficiency_dominant": {
        "json_structure": 0.10,
        "efficiency": 0.65,
        "scalability": 0.15,
        "clarity": 0.10,
    },
    "structural_dominant": {
        "json_structure": 0.50,
        "efficiency": 0.10,
        "scalability": 0.10,
        "clarity": 0.30,
    },
    "scalability_dominant": {
        "json_structure": 0.10,
        "efficiency": 0.15,
        "scalability": 0.65,
        "clarity": 0.10,
    }
}
CRITERIA = [
    "json_structure",
    "efficiency",
    "scalability",
    "clarity"
]

# --- Keyword groups per criterion ---
KEYWORDS = {
    "feasibility": [
        "deploy", "configure", "restart", "update",
        "monitor", "rollback", "patch", "validate", "execute"
    ],

    "efficiency": [
        "resolve", "eliminate", "optimize", "fix", "address"
    ],

    "scalability": [
        "automate", "distributed", "cluster", "replicate",
        "horizontal", "load balancing", "orchestrate", "container"
    ],

    "clarity_markers": [
        "step", "first", "second", "then", "finally"
    ]
}



CONFIG = {
    "root_cause_file": "data/root_causes.json",
    "max_causes": 100,
    "num_train_epochs": 6,
    "logging_steps": 10,
    "max_new_tokens": 64,
    "temperature": 0.9,
}

def score_feasibility(text):
    text_lower = text.lower()

    # 1 Action verbs
    action_verbs = [
        "restart", "deploy", "configure", "update",
        "patch", "rollback", "adjust", "validate",
        "monitor", "scale"
    ]

    verb_score = min(sum(text_lower.count(v) for v in action_verbs) / 4, 1.0)

    # 2 Technical artifact references
    technical_terms = [
        "configuration", "parameter", "threshold",
        "service", "pod", "container",
        "instance", "node", "cluster"
    ]

    tech_score = min(sum(text_lower.count(t) for t in technical_terms) / 4, 1.0)

    # 3 Penalize abstract language
    abstract_terms = ["consider", "should", "might", "possibly"]
    abstraction_penalty = min(sum(text_lower.count(t) for t in abstract_terms) / 4, 0.5)

    return max(0.0, 0.6 * verb_score + 0.4 * tech_score - abstraction_penalty)

def reward_function(completions, flags=None, weights=None, prompts=None, normalize=True, **kwargs):

    # Default: enable all aligned criteria
    default_flags = {
        "json_structure": True,
        "feasibility": True,
        "efficiency": True,
        "scalability": True,
        "clarity": True,
    }

    default_weights = {
        "json_structure": 0.25,
        "efficiency": 0.25,
        "scalability": 0.25,
        "clarity": 0.25,
    }
    
    flags = default_flags #flags or default_flags --> i am disabling custom flags for now to keep it simple, but we can re-enable this later if needed
    weights = default_weights #weights or default_weights --> i am disabling custom weights for now to keep it simple, but we can re-enable this later if needed
    #total = sum(weights.values())
    #if total > 0:
    #    weights = {k: v / total for k, v in weights.items()}
    rewards = []

    for i, c in enumerate(completions):

        text = c
        text_lower = text.lower()
        root = prompts[i] if prompts else ""
        root_lower = root.lower()

        # -----------------------
        # 1 JSON Structure
        # -----------------------
        if flags.get("json_structure", True):
            try:
                parsed = json.loads(text)
                json_score = 1.0 if isinstance(parsed, dict) else 0.5
            except:
                json_score = 0.0
        else:
            json_score = 0.0

        # -----------------------
        # 2 Efficiency (Causal Root Alignment with Proximity)
        # Must:
        # - mention root token
        # - include causal marker
        # - causal marker appears near root term
        # -----------------------
        if flags.get("efficiency", True) and root:
            sentences = re.split(r'[.!?]', text_lower)

            causal_markers = ["because", "due to", "caused by", "as a result"]
            stopwords = {"the", "is", "a", "an", "of", "to", "and", "in", "for", "on", "with"}
            root_tokens = {
                t for t in re.findall(r'\b\w+\b', root_lower)
                if len(t) > 3 and t not in stopwords
            }
            efficiency = 0.0

            for s in sentences:
                tokens = s.split()

                root_positions = [
                    i for i, tok in enumerate(tokens)
                    if tok in root_tokens
                ]

                causal_positions = [
                    i for i, tok in enumerate(tokens)
                    if tok in causal_markers
                ]

                if not root_positions or not causal_positions:
                    continue

                for rp in root_positions:
                    if any(abs(rp - cp) <= 8 for cp in causal_positions):
                        efficiency = 1.0
                        break

                if efficiency == 1.0:
                    break
        else:
            efficiency = 0.0


        # -----------------------
        # 3 Scalability (System-Level Scope Only)
        # Must include plural/system-wide reference
        # -----------------------
        if flags.get("scalability", True):
            scale_scope_terms = [
                "across",
                "cluster",
                "replica",
                "instances",
                "horizontal",
                "autoscaling",
                "auto scaling",
            ]

            scalability = 1.0 if any(term in text_lower for term in scale_scope_terms) else 0.0
        else:
            scalability = 0.0


        # -----------------------
        # 4 Clarity (Explicit Step Structure Only)
        # Must contain 2+ structured steps
        # -----------------------
        if flags.get("clarity", True):
            numbered = len(re.findall(r"^\s*\d+\.", text, re.MULTILINE))
            bullet = len(re.findall(r"^\s*[-•]", text, re.MULTILINE))

            clarity = 1.0 if (numbered + bullet) >= 2 else 0.0
        else:
            clarity = 0.0


        # -----------------------
        # Final Weighted Sum
        # -----------------------

        reward = (
            weights.get("json_structure", 0) * json_score +
            weights.get("efficiency", 0) * efficiency +
            weights.get("scalability", 0) * scalability +
            weights.get("clarity", 0) * clarity
        )

        rewards.append(reward)

    return rewards

def make_reward_for_experiment(
    experiment="full",
    leave_out_crit=None,
    only_one_crit=None,
    active_criteria=None,
    active_config=None,
    normalize=False
):
    """
    Returns a reward function for GRPOTrainer based on the experiment.
    
    Args:
        experiment: "full" or "leave_one_out"
        leave_out_crit: (str) the criterion to leave out if experiment=="leave_one_out"
        normalize: bool, normalize individual components
    """
    # All criteria flags
    all_flags = {k: True for k in CRITERIA}    
    weights = None 

    if experiment == "leave_one_out":
        if leave_out_crit is None or leave_out_crit not in CRITERIA:
            raise ValueError(f"leave_out_crit must be one of {CRITERIA}")
        # disable the chosen criterion
        all_flags[leave_out_crit] = False

    elif experiment == "only_one":
        if only_one_crit is None or only_one_crit not in CRITERIA:
            raise ValueError(f"only_one_crit must be one of {list(CRITERIA)}")
        # disable all except the selected one
        all_flags = {k: (k == only_one_crit) for k in CRITERIA}

    elif experiment == "custom":
        active_criteria = active_criteria.split(",") if isinstance(active_criteria, str) else active_criteria
        all_flags = {k: (k in active_criteria) for k in CRITERIA}

    elif experiment == "configs":
        if active_config is None or active_config not in REWARD_CONFIGS:
            raise ValueError(f"active_config must be one of {list(REWARD_CONFIGS)}")
        weights = REWARD_CONFIGS[active_config]
        all_flags = {k: True for k in CRITERIA}
    
    def reward_fn(completions, prompts=None, **kwargs):
        return reward_function(
            completions,
            flags=all_flags,
            weights=weights if experiment == "configs" else None,
            prompts=prompts,
            **kwargs
        )
    
    return reward_fn

#--------------------------------------------------
#------------------TensorBoard Training Loop--------
#--------------------------------------------------

def load_root_causes_by_files(path, datasize=1, max_causes_per_file=5):
    """
    datasize: number of files to select per scenario type (e.g., re2-ss, re2-tt)
    max_causes_per_file: number of root causes per file
    """

    # Collect files by scenario type (based on suffix)
    grouped_files = defaultdict(list)

    for f in os.listdir(path):
        if f.endswith(".json"):
            # extract scenario type (e.g., re2-ss, re2-tt)
            scenario = f.split("_")[-1].replace(".json", "")
            grouped_files[scenario].append(f)

    causes = []

    for scenario, files in grouped_files.items():
        selected_files = files[:datasize]  # select N files per scenario

        for file in selected_files:
            file_path = os.path.join(path, file)

            with open(file_path, "r") as f:
                data = json.load(f)

            for key, items in data.items():
                for cause in items[:max_causes_per_file]:
                    causes.append({
                        "prompt": f"Root cause: {cause}. Suggest a clear, feasible, and effective mitigation plan.",
                        "solution": ""
                    })

    return causes

import os
import json
from collections import defaultdict

def load_root_causes(path, datasize=1, max_causes_per_file=5):
    """
    datasize:
        1 -> one microservice
        2 -> two microservices
        3 -> three microservices (e.g., re2-ob, re2-ss, re2-tt)

    max_causes_per_file:
        number of root causes per file
    """

    # Group files by microservice name (prefix before first underscore)
    grouped_by_service = defaultdict(list)

    for f in os.listdir(path):
        if f.endswith(".json"):
            service_name = f.split("_")[0]  # carts, orders, ts-auth-service, etc.
            grouped_by_service[service_name].append(f)

    # Sort for reproducibility
    services = sorted(grouped_by_service.keys())

    # Select first N microservices
    selected_services = services[:datasize]

    causes = []

    for service in selected_services:
        for file in grouped_by_service[service]:

            file_path = os.path.join(path, file)
            print(f"Loading root causes from: {file_path}")
            with open(file_path, "r") as f:
                data = json.load(f)

            for key, items in data.items():
                for cause in items[:max_causes_per_file]:
                    causes.append({
                        "prompt": f"Root cause: {cause}. Suggest a clear, feasible, and effective mitigation plan.",
                        "solution": ""
                    })

    return causes

def generate_plan(generator, cause: str, max_new_tokens: int = 256, temperature: float = 0.7) -> dict:
    """
    Generate a mitigation plan for a given root cause using the provided generator.
    Returns both the prompt and completion.
    """
    #prompt = f"Root cause: {cause}. Suggest a clear, feasible, and effective mitigation plan."
    
    prompt = f"""
        You are a senior system reliability engineer participating in RLHF training.

        Your objective is to generate a high-quality, actionable mitigation plan for a given system failure root cause.
        The response will be evaluated by human reviewers based on the following criteria:

        1. Technical correctness
        2. Practical feasibility
        3. Specificity and actionability
        4. Risk reduction effectiveness
        5. Alignment with industry best practices (e.g., SRE, DevOps, reliability engineering)
        .\n\n

        The data you receive will be in the name of the root cause. 
        Your task is to analyze the root cause and generate a mitigation plan that addresses it effectively.
        Root Causes:{cause}.
    """
    

    outputs = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
    )

    full_text = outputs[0]["generated_text"]
    completion = full_text.replace(prompt, "").strip()

    return {"prompt": prompt, "completion": completion}

def prepare_dataset(causes, generator, reward_function, max_new_tokens=256, temperature=0.7):
    """
    Prepare dataset for RLHF training by generating mitigation plans and assigning rewards.
    """
    dataset = []

    for cause in tqdm(causes, desc="Generating mitigation plans"):
        result = generate_plan(generator, cause, max_new_tokens, temperature)
        prompt = result["prompt"]
        completion = result["completion"]

        # Compute the reward based on the generated mitigation plan
        reward = reward_function(prompt, completion)

        dataset.append({
            "prompt": prompt,
            "solution": completion,  # "solution" is what GRPOTrainer expects
            "reward": reward,
        })

        # Optional cleanup to avoid GPU memory leaks
        gc.collect()
        torch.cuda.empty_cache()

    return dataset

#-----------------------------------------------
#-------------------Train PPO-------------------
#-----------------------------------------------

from peft import prepare_model_for_kbit_training


from peft import prepare_model_for_kbit_training

def run(arguments=None):
    # Load root causes
    if arguments and arguments.experiment == "datasize":
        print(f"Loading dataset with datasize={arguments.active_criteria[0]} files per scenario...")
    #NOTE: set the location for training root cases here.
    training_root_cases_path = "output/training_root_cases"  # Update this path to your actual root causes directory
    dataset = load_root_causes(training_root_cases_path, datasize=int(arguments.active_criteria[0]) if arguments and arguments.experiment == "datasize" else 100)  
    train_data, test_data = train_test_split(
        dataset,
        test_size=0.01,
        random_state=42
    )
    train_data = Dataset.from_list(train_data)
    test_data = Dataset.from_list(test_data)
    
    # Use 4-bit quantization with FP16 compute dtype
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Changed from bfloat16 to float16
    )
    
    # Initialize TensorBoard
    log_dir = f"logs/ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    
    # Load model and tokenizer
    model_id = "ibm-granite/granite-4.0-micro"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,  # Changed from bfloat16 to float16
        attn_implementation="eager",
        device_map="auto",
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    
    # Enable input gradients
    model.enable_input_require_grads()
    
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(arguments.active_criteria[0]) if arguments and arguments.experiment == "rank" else 4,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Configure training arguments using GRPOConfig
    print("current experiment:", arguments.experiment )
    print("current active criteria:", arguments.active_criteria)
    if arguments.experiment == "temperature":
        print("current active config:", float(arguments.active_criteria[0]))
    training_args = GRPOConfig(
        learning_rate=float(arguments.active_criteria[0]) if arguments and arguments.experiment == "lr" else 1e-5,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        gradient_accumulation_steps=4,
        num_train_epochs=int(arguments.active_criteria[0]) if arguments and arguments.experiment == "epochs" else 12,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,   # Changed to True
        bf16=False,  # Changed to False
        max_completion_length=int(arguments.active_criteria[0]) if arguments and arguments.experiment == "completion" else 128,
        num_generations=int(arguments.active_criteria[0]) if arguments and arguments.experiment == "generation" else 8,
        max_prompt_length=128,
        max_grad_norm=1.0,
        beta=float(arguments.active_criteria[0]) if arguments and arguments.experiment == "kl" else 0.0,#Kl divergence coefficient (beta)
        temperature=float(arguments.active_criteria[0]) if arguments and arguments.experiment == "temperature" else 0.7,
        top_p=float(arguments.active_criteria[0]) if arguments and arguments.experiment == "topp" else 0.9,
        report_to=["tensorboard"],
        logging_steps=10,
        push_to_hub=False,
        save_strategy="steps",
        dataloader_drop_last=True,
        optim= arguments.active_criteria[0] if arguments and arguments.experiment == "optimizer" else "paged_adamw_8bit",
    )
    
    if arguments is not None:
        experiment = arguments.experiment
        leave_out_crit = arguments.leave_out_crit
        normalize = arguments.normalize

    # Create save directory
    os.makedirs("./saved_models", exist_ok=True) #NOTE: change this to your desired directory for saving models

    if experiment == "full" :
        reward_full = make_reward_for_experiment(experiment="full")

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_full],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained("./saved_models/full_reward_meta")
        tokenizer.save_pretrained("./saved_models/full_reward_meta")
    elif experiment == "leave_one_out":
        reward_loo = make_reward_for_experiment(experiment="leave_one_out", leave_out_crit=arguments.leave_out_crit)
    
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_loo],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )

        save_dir = f"./saved_models/leave_out_{arguments.leave_out_crit}"
        os.makedirs(save_dir, exist_ok=True)
        
        trainer.train()
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    elif experiment == "only_one":
        reward_one = make_reward_for_experiment(
            experiment="only_one",
            only_one_crit=arguments.only_one_crit,
        )

        save_dir = f"./saved_models/only_one_{arguments.only_one_crit}"
        os.makedirs(save_dir, exist_ok=True)

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_one],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    elif experiment == "custom":
        reward_custom = make_reward_for_experiment(
            experiment="custom",
            active_criteria=arguments.active_criteria,
        )

        save_dir = f"./saved_models/custom_3DB_{'_'.join(arguments.active_criteria)}"
        os.makedirs(save_dir, exist_ok=True)

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_custom],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    elif experiment == "configs":
        reward_config = make_reward_for_experiment(
            experiment="configs",
            active_config=arguments.active_config,
        )

        save_dir = f"./saved_models/config_{arguments.active_config}"
        os.makedirs(save_dir, exist_ok=True)

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_config],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    elif experiment == "epochs":
        reward_full = make_reward_for_experiment(experiment="full")

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_full],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(f"./saved_models/epochs_{arguments.active_criteria[0]}")
        tokenizer.save_pretrained(f"./saved_models/epochs_{arguments.active_criteria[0]}")
    elif experiment == "datasize":
        reward_full = make_reward_for_experiment(experiment="full")

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_full],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(f"./saved_models/datasize_micro_{arguments.active_criteria[0]}")
        tokenizer.save_pretrained(f"./saved_models/datasize_micro_{arguments.active_criteria[0]}")
    elif experiment == "completion":
        reward_full = make_reward_for_experiment(experiment="full")

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_full],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(f"./saved_models/completion_{arguments.active_criteria[0]}")
        tokenizer.save_pretrained(f"./saved_models/completion_{arguments.active_criteria[0]}")
    elif experiment == "generation":
        reward_full = make_reward_for_experiment(experiment="full")

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_full],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(f"./saved_models/generation_{arguments.active_criteria[0]}")
        tokenizer.save_pretrained(f"./saved_models/generation_{arguments.active_criteria[0]}")
    elif experiment == "rank":
        reward_full = make_reward_for_experiment(experiment="full")

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_full],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(f"./saved_models/rank_{arguments.active_criteria[0]}")
        tokenizer.save_pretrained(f"./saved_models/rank_{arguments.active_criteria[0]}")
    elif experiment == "temperature":
        reward_full = make_reward_for_experiment(experiment="full")

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_full],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(f"./saved_models/temperature_{arguments.active_criteria[0]}")
        tokenizer.save_pretrained(f"./saved_models/temperature_{arguments.active_criteria[0]}")
    elif experiment == "lr":
        reward_full = make_reward_for_experiment(experiment="full")

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_full],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(f"./saved_models/lr_{arguments.active_criteria[0]}")
        tokenizer.save_pretrained(f"./saved_models/lr_{arguments.active_criteria[0]}")
    elif experiment == "topp":
        reward_full = make_reward_for_experiment(experiment="full")

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_full],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(f"./saved_models/topp_{arguments.active_criteria[0]}")
        tokenizer.save_pretrained(f"./saved_models/topp_{arguments.active_criteria[0]}")
    elif experiment == "optimizer":
        reward_full = make_reward_for_experiment(experiment="full")

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_full],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(f"./saved_models/optimizer_{arguments.active_criteria[0]}")
        tokenizer.save_pretrained(f"./saved_models/optimizer_{arguments.active_criteria[0]}")
    elif experiment == "kl":
        reward_full = make_reward_for_experiment(experiment="full")

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_full],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(f"./saved_models/kl_{arguments.active_criteria[0]}")
        tokenizer.save_pretrained(f"./saved_models/kl_{arguments.active_criteria[0]}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="full")
    parser.add_argument("--leave_out_crit", type=str, default=None)
    parser.add_argument("--only_one_crit", type=str, default=None)
    parser.add_argument("--active_criteria", type=str, nargs='*', default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--active_config", type=str, default=None)
    args = parser.parse_args()

    run(arguments=args)