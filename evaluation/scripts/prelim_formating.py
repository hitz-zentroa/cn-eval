import json
import os
import pandas as pd

def formatting_function(data):
    for i,hs in enumerate(data['HS']):
        question=f"Provide a brief counter-narrative in response to the following hate speech: {hs} It is very important for the counter-narrative to be relevant to the hate speech."
        with open(f'{save_folder}/{file[:-4]}.txt', 'a') as f:
            f.write(json.dumps({"question_id":i,"question_body":question,"model":file[:-7],"text":data["generated"].loc[i]})+"\n")

def consider_gt(data):
    for i,hs in enumerate(data['HS']):
        question=f"Provide a brief counter-narrative in response to the following hate speech: {hs} It is very important for the counter-narrative to be relevant to the hate speech."
        with open(f'{save_folder}/golden_truth.txt', 'a') as f:
            f.write(json.dumps({"question_id":i,"question_body":question,"model":"golden_truth","text":data["Label"].loc[i]})+"\n")

if __name__ == "__main__":

    save_folder = os.environ.get('SAVE_FOLDER')
    path = os.environ.get('DIC_PH')
    
    files= os.popen(f'ls {path} | grep "csv"').read().split()
    for file in files:
        data = pd.read_csv(path+"/"+file)
        formatting_function(data)
    consider_gt(data)