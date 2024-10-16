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
meta-llama/Llama-2-7b-chat-hf)  # list of huggingface checkpoint names

learning_rates=() # list of learning rates for each model

architectures=(zephyr
mistral
mistral-instruct
llama-chat)

for (( num=0; num<${#base_models[@]}; num++ ));
do
    export CH_MODEL=${base_models[$num]}
    export SAVE_FOLDER=./models/
    export SAVE_NAME=${architectures[$num]}-$lr
    export DATA_FOLD=./data/
    export ARCH=${architectures[$num]}

    #HIPERPARAMETERS
    export BS=32
    export LR=${learning_rates[$num]}
    export EPOCHS=5
    export MAX_GEN=1024

    srun python ./models/scripts/finetune_decoder.py
done
