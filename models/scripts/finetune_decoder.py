# INITIAL IMPORTS
import pandas as pd
import numpy as np
import math
import os
import wandb

from datasets import Dataset
import torch

from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, EarlyStoppingCallback,BitsAndBytesConfig,TrainingArguments
from trl import SFTTrainer

# GENERAL FUNCTIONS
def train_formatting_function(data):
    """
    Template for training
    """
    full_text = []
    
    for i in range(len(data['HS'])):
        # brief now concise
        formated_sen_instruct = [
            {"role": "user", "content":f"Provide a brief counter-narrative in response to the user's hate speech. Ensure the output does not contain line breaks.{data['HS'][i]}"},
            {"role": "assistant", "content":data['CN'][i]}
            ]
        formated_sen_chat = [
        {"role": "system", "content": "Provide a brief counter-narrative in response to the user's hate speech. Ensure the output does not contain line breaks."},
        {"role": "user", "content":data['HS'][i]},
        {"role": "assistant", "content":data['CN'][i]},
        ]
        
        if model_type in ["mistral-instruct"]:
            formated_sen = tokenizer.apply_chat_template(formated_sen_instruct, add_generation_prompt=True, tokenize=False)
        elif model_type in ["zephyr","llama-chat"]:
            formated_sen = tokenizer.apply_chat_template(formated_sen_chat, add_generation_prompt=True, tokenize=False)
        elif model_type in ["mistral"]:
            formated_sen = f"Provide a brief counter-narrative in response to the user's hate speech. Ensure the output does not contain line breaks.\n###Input:\n{data['HS'][i]}\n###Output:\n{data['CN'][i]}"
        else:
            print("No such template")
        
        full_text.append(formated_sen)

    return {"text":full_text}


def compute_metrics(eval_results):
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

if __name__ == "__main__":

    #SET SEEDS
    torch.manual_seed(42)
    np.random.seed(42)

    #GET ENVIRONMENT VARIABLES
    model_chk = os.environ.get('CH_MODEL')
    save_folder = os.environ.get('SAVE_FOLDER')
    save_name = os.environ.get('SAVE_NAME')
    data_folder = os.environ.get('DATA_FOLD')
    model_type = os.environ.get('ARCH')
    corpus = os.environ.get('CORPUS')

    #HIPERPARAMETERS
    bs = int(os.environ.get('BS'))
    lr = float(os.environ.get('LR'))
    epochs = int(os.environ.get('EPOCHS'))
    max_seq_length = int(os.environ.get('MAX_GEN'))

    #WANDB
    wandb.init(
    # set the wandb project where this run will be logged
    project=f"{corpus}-QLORA-ft",

    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": model_type,
    "dataset": corpus,
    "epochs": epochs,
    "batch_size":bs,
    }
)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.bfloat16,
        bnb_4bit_use_double_quant= False,
    )
    model = AutoModelForCausalLM.from_pretrained(model_chk,quantization_config=bnb_config, device_map="auto")
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(model_chk, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = True
    tokenizer.add_bos_token, tokenizer.add_eos_token

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
        )
    model = get_peft_model(model, peft_config)

    # LOAD DATA
    try:
        train_data = pd.read_csv(f"{data_folder}{corpus}/{corpus}_train.csv")
        val_data = pd.read_csv(f"{data_folder}{corpus}/{corpus}_val.csv")

    except FileNotFoundError: 
        print("Data not found")
    
    #DF2DATASET
    train = Dataset.from_pandas(train_data, preserve_index=False)
    val = Dataset.from_pandas(val_data, preserve_index=False)

    formated_train = train.map(train_formatting_function, batched=True,remove_columns=train.column_names)
    formated_val = val.map(train_formatting_function, batched=True,remove_columns=val.column_names)

    # TRAINING
    training_args = TrainingArguments(
        output_dir=save_folder + save_name,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        save_total_limit=3,
        load_best_model_at_end=True,
        num_train_epochs=epochs,
        lr_scheduler_type="inverse_sqrt",
        warmup_ratio=0.1,
        fp16=False,
        optim = "paged_adamw_8bit",
        report_to="wandb",
        run_name=f"{model_type}-{lr}",
        logging_steps=1,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        train_dataset=formated_train,
        eval_dataset=formated_val,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],
        packing=False
    )

    trainer.train()
    model.config.use_cache = True
    eval_results = trainer.evaluate()

    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    print(trainer.state.best_model_checkpoint)

