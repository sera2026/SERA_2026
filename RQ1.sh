#!/bin/bash

# Set Python environment (optional)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

SCRIPT="Mitigation/train2.py"

# For RQ1, Majoriry Voting
# use Create Plans and Evaluation scripts over the open-source LLMs 
# to select the best performing LLM for RQ2 and RQ3

# First, we need to obtain the root causes before creating plans.

echo "============================="
echo "Obtain rootcauses before creating plans using S.O.T.A BARO..."
echo "============================="

datasets=(
    "re1-ob" "re1-ss" "re1-tt"
)
methods=(baro)  
RESULTS_DIR="output/results"
DEST_DIR="output/eval_rootcauses"

mkdir -p "$DEST_DIR"

for dataset in "${datasets[@]}"; do
    echo "Dataset: $dataset"
    
    cmd="python main.py --dataset $dataset --method ${methods[0]} --research_question RQ_mitigation"

    echo "Running: $cmd"
    eval $cmd


    echo "Selecting 3 containers and 1 file from each..."

    # Get unique container names
    containers=$(ls "$RESULTS_DIR" | cut -d'_' -f1 | sort -u | head -n 3)

    for container in $containers; do
        echo "Processing container: $container"

        # Pick one file from that container
        file=$(ls "$RESULTS_DIR"/${container}_*.json | sort | head -n 1)

        base=$(basename "$file" .json)
        new_name="${base}_${dataset}.json"

        cp "$file" "$DEST_DIR/$new_name"
        echo "Saved: $new_name"
    done

    echo "Finished dataset: $dataset"
done

echo "============================="
echo "Creating Plans..."
echo "============================="
python Mitigation/create_plans.py


echo "============================="
echo "Evaluating Plans..."
echo "============================="
python Mitigation/evaluation.py