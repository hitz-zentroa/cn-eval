import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from datasets import Dataset
    
def preprocess_function(hs):

    if mod_type in ["mistral-instruct"]:
        formated_sen = [
            {"role": "user", "content":f"Provide a brief counter-narrative in response to the user's hate speech. Ensure the output does not contain line breaks.{hs}"},
            ]
        formated_sen = tokenizer.apply_chat_template(formated_sen, add_generation_prompt=True, tokenize=False)
    elif mod_type in ["zephyr","llama-chat"]:
        formated_sen = [
            {"role": "system", "content": f"Provide a brief counter-narrative in response to the user's hate speech. Ensure the output does not contain line breaks."},
            {"role": "user", "content":hs},
            ]
        formated_sen = tokenizer.apply_chat_template(formated_sen, add_generation_prompt=True, tokenize=False)
    elif mod_type in ["mistral"]:
            formated_sen = f"Provide a brief counter-narrative in response to the user's hate speech. Ensure the output does not contain line breaks.\n###Input:\n{hs}\n###Output:\n"
        
    else:
        print("No model type selected")
        return None
    return formated_sen

def generate_cn (sentence,label,max_new_tokens):
    outputs=dict()

    outputs["HS"]=[sentence]
    outputs["Label"]=[label]

    input_text = preprocess_function(sentence)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    gen_tokens = model.generate(**inputs, max_new_tokens=max_new_tokens,do_sample=True)
    response_tokens = gen_tokens[0][inputs.input_ids.shape[-1]:]
    outputs["generated"] = [tokenizer.decode(response_tokens, skip_special_tokens=True).strip().split("\n")[0].strip()]
    
    return(pd.DataFrame.from_dict(outputs))

if __name__ == "__main__":

    #SET SEEDS
    torch.manual_seed(42)
    np.random.seed(42)

    #GET ENVIRONMENT VARIABLES
    model_chk = os.environ.get('CH_MODEL')
    model_path = os.environ.get('MODEL_PATH')
    trained_model = os.environ.get('TR_MODEL')
    save_folder = os.environ.get('SAVE_FOLDER')
    save_name = os.environ.get('SAVE_NAME')
    data_folder = os.environ.get('DATA_FOLD')
    max_new_tokens = int(os.environ.get('MAX_GEN'))
    mod_type = os.environ.get('ARCH')
    corpus=os.environ.get('CORPUS')
    
    #LOAD DATA
    try:
        test_data = pd.read_csv(data_folder+f"{corpus}/{corpus}_test.csv")
    except FileNotFoundError : 
        print("File not found")
    
    test = Dataset.from_pandas(test_data,preserve_index=False)

    # LOAD MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_chk)
    model = AutoModelForCausalLM.from_pretrained(model_chk, device_map="auto")

    if trained_model:
        peft_model_id = model_path+trained_model
        config = PeftConfig.from_pretrained(peft_model_id)
        model = PeftModel.from_pretrained(model, peft_model_id)

    # GENERATE  
    generated_df=pd.DataFrame()
    for i in range(len(test)):
        generated_df = pd.concat([generated_df,generate_cn(test[i]["HS"], test[i]["CN"],max_new_tokens)])
        generated_df.to_csv(save_folder+save_name, index=False) # Save while gen
