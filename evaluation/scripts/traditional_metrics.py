# Imports

import numpy as np
import pandas as pd
import string
import os
import evaluate

# GENERIC FUNCTIONS

def jaccard_similarity(text1, text2):
    a = text1.split()
    b = text2.split()

    intersection = len(list(set(a).intersection(set(b))))
    union = len(set(a).union(set(b)))
    if union == 0:
        print(a, b)
    return float(intersection) / float(union)

def novelty(train, generation):

    novelty_score = []
    for t in generation:
        t_score = []
        for text in train:
            t_score.append(jaccard_similarity(text, t))
        s = 1 - max(t_score)
        novelty_score.append(s)
    return np.mean(np.array(novelty_score))

def read_txt(f):
    sentences = []
    for line in f:
        h = line.translate(str.maketrans('', '', string.punctuation))
        h = " ".join(h.lower().split())
        sentences.append(h)
    return sentences

def read_ww_cn_txt(filename):
    sentences = []
    with open(filename) as f:
        for line in f:
            cn = line.split(" 	 ")
            #
            if len(cn) == 2:
                new_s = formalized_train(cn[1][:-1])
                h = " ".join(new_s.lower().split())
                h = h.translate(str.maketrans('', '', string.punctuation))
                h = " ".join(h.lower().split())
                sentences.append(h)
            else:
                print(cn)
    return sentences

def formalized_train(text):
    if "'" in text:
        a = text.replace("'", " ")
        return a
    else:
        return text

def compute_metrics(labels, predictions):
        result = {}
        result["model"]= model_generation_type
        result['rougeL'] = round(rouge.compute(predictions=predictions, references=labels, use_stemmer=True)['rougeL'],4)
        result["bleu"] = round(bleu.compute(predictions=predictions, references=labels)["bleu"],4)
        result["bertscore"] = round(np.mean(bertscore.compute(predictions=predictions, references=labels, model_type = "bert-base-multilingual-cased", device = "cuda:0")["f1"]),4)
        prediction_lens = [len(pred.split()) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return result

def permutations(elements):
    similarities = []
    for i,el1 in enumerate(elements):
        if i == (len(elements)-1):
            return(np.mean(similarities)) 
        else:
            rest = elements[i+1:]
            result = bertscore.compute(predictions=[el1]*len(rest), references=rest, model_type = "bert-base-multilingual-cased", device = "cuda:0")["f1"]
            similarities+=result 

if __name__ == "__main__":

    #GET ENVIRONMENT VARIABLES
    data_folder = os.environ.get('DATA_FOLD')
    generation_folder = os.environ.get('GEN_FOLD')
    save_folder = os.environ.get('SAVE_FOLD')
    corpus = os.environ.get('CORPUS')
    generated_texts = os.popen(f'ls {generation_folder} | grep "csv"').read().split()

    m1_list=[]
    m2_list=[]

    for gen_text in generated_texts:
        
        train = pd.read_csv(data_folder+corpus+f"/{corpus}_train.csv")

        all_hs = pd.read_csv(generation_folder+gen_text)

        all_generated=all_generated.fillna("")

        model_generation_type = gen_text[:-4]

        # Load metrics
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")
        bertscore = evaluate.load("bertscore")   

        # NOVELTY
        train = read_txt(train["CN"])
        
        generation = read_txt(all_generated["generated"])
        score = novelty(list(set(train)), generation)
        nov=score
        # RR
        os.chdir("./Notebooks")
        
        generation = all_generated["generated"]
        generation.to_csv(f"test_generated.csv",index=False, header=False)

        os.chdir("./evaluation/repetition_rate/")
        
        res = os.popen(f"sh run_repetition_rate.sh ./Notebooks/test_generated.csv").read()
        rr=float(res.split()[-1])

        os.chdir("./Notebooks")

        grouped_gen=all_hs[all_hs["HS"].isin(all_generated["HS"].to_list())]
        grouped_gen = all_generated.groupby('HS',sort=False)["Label"].agg(list).reset_index()
        grouped_gen.columns = ['HS', 'label_list']

        generation = all_generated["generated"]
        metrics = compute_metrics(all_generated["Label"], generation)
        metrics["novelty"]=round(nov,4)
        metrics["RR"]=round(rr,4)

        m1_list.append(metrics)

        group_eval = all_generated.merge(grouped_gen, on="HS")
        metrics2 = compute_metrics(group_eval["label_list"], group_eval["generated"])

        #The same in both cases
        metrics2["novelty"]=round(nov,4)
        metrics2["RR"]=round(rr,4)
        m2_list.append(metrics2)

    pd.DataFrame(m1_list).to_csv(save_folder+f"{corpus}_Regular_metrics_human.csv", index=False)
    pd.DataFrame(m2_list).to_csv(save_folder+f"{corpus}_Multireference_metrics_human.csv", index=False)


