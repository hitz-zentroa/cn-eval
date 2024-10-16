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

    train = pd.read_csv(data_folder+corpus+f"/{corpus}_train.csv")
    formatted_train = read_txt(train["KN_CN"])

    m_list=[]

    for gen_text in generated_texts:

        generated_test = pd.read_csv(generation_folder+gen_text).fillna("")
        generation = generated_test["generated"]

        model_generation_type = gen_text[:-4]

        # Load metrics
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")
        bertscore = evaluate.load("bertscore")   

        # NOVELTY
        formatted_generation = read_txt(generation)
        score = novelty(list(set(formatted_train)), formatted_generation)
        nov=score

        # METRICS
        metrics = compute_metrics(generated_test["Label"], generation)
        metrics["novelty"]=round(nov,4)

        m_list.append(metrics)

    pd.DataFrame(m_list).to_csv(save_folder+f"{corpus}_Regular_metrics_human.csv", index=False)


