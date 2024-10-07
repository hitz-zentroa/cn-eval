import math
import numpy as np
import pandas as pd
import os
import random
random.seed(42)

def calculate_instancelv_results(path,file_name):
    """
    Reorganize results to rank
    """
    data = pd.read_json(path+file_name,lines=True)

    if len(data[data["pred_text"].apply(lambda x: len(x.split("\n")[0].split()) != 2)]):
        print("no two scores in first row!!!")
        return(None)
    
    data["score1"],data["score2"]=data["pred_text"].apply(lambda x: float(x.split("\n")[0].split()[0])), data["pred_text"].apply(lambda x: float(x.split("\n")[0].split()[1]))
    data["win1"], data["win2"]=data.apply(lambda row: 1 if row['score1'] > row['score2'] else 0, axis=1),data.apply(lambda row: 1 if row['score1'] < row['score2'] else 0, axis=1)

    wins=[]
    for a,b in zip(data["win1"], data["win2"]):
    
    # by winEvaluation code
        if a==b :
            by_win = "tie"
        
        elif a==1 and b ==0:
            by_win = data.loc[0]["answer1_model_id"]
            
        elif a==0 and b==1:
            by_win = data.loc[0]["answer2_model_id"]
        else:
            (print(a,b))
        wins.append(by_win)

    result={"question_id":data["question_id"],"judge_model":[data.loc[0]["pred_model_id"]]*len(data),"model_1":[data.loc[0]["answer1_model_id"]]*len(data),"model_2":[data.loc[0]["answer2_model_id"]]*len(data),\
            "wins":wins}
    return(pd.DataFrame(result))

if __name__ == "__main__":

    judgement_path= os.environ.get('SAVE_FOLDER')
    params= os.environ.get('PARAMS')

    battles=pd.DataFrame()
    for fn in os.popen(f'ls {judgement_path}| grep "json"').read().split():
        battles=pd.concat([battles,calculate_instancelv_results(judgement_path,fn)])

    battles.reset_index(drop=True).to_csv(f"{judgement_path}/{params}_ilv_results.csv",index=False)


