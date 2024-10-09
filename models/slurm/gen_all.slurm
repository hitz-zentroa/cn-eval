#!/bin/bash
#SBATCH --job-name=
#SBATCH --cpus-per-task=
#SBATCH --gres=
#SBATCH --mem=
#SBATCH --time=


#SBATCH --output=./logs/out/%x_%j.out
#SBATCH --error=.logs/errors/%x_%j.err

source ./venv/bin/activate

export HF_HOME=./cache/
export CORPUS=CONAN # other option MT-CONAN

base_models=(HuggingFaceH4/zephyr-7b-beta
mistralai/Mistral-7B-v0.1
mistralai/Mistral-7B-Instruct-v0.2
meta-llama/Llama-2-7b-chat-hf) # List of huggingface checkpoint names

filenames=() # list of output file names. One per model

architectures=(zephyr
mistral
mistral-instruct
llama-chat) # In generation this parameter only affects prompting format


for (( num=0; num<${#base_models[@]}; num++ ));
do
    export CH_MODEL=${base_models[$num]}
    export MODEL_PATH=./models/
    export TR_MODEL= # if empty, zero-shot. Otherwise, expected to contain the path to the finetuned model
    export SAVE_FOLDER=./generation/
    export SAVE_NAME=${filenames[$num]}
    export DATA_FOLD=./data/
    export MAX_GEN=1024
    export ARCH=${architectures[$num]}

    srun python ./models/scripts/generate_decoder.py
done