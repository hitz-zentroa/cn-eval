#!/bin/bash
#SBATCH --job-name=
#SBATCH --cpus-per-task=
#SBATCH --gres=
#SBATCH --mem=
#SBATCH --time=

#SBATCH --output=./evaluation/logs/%x_%j.out
#SBATCH --error=./evaluation/logs/%x_%j.err

source ./venv/bin/activate

params=33
corpus=CONAN

directory_path="./models/${corpus}/generation/"
save_path="./evaluation/judgements/${corpus}/${params}B/"
mkdir $save_path/formated_data

export SAVE_FOLDER=${save_path}
export DIC_PH=$directory_path
export PARAMS=$params

# Create txt files
srun python ./evaluation/scripts/prelim_formating.py

echo "formatting into txt done"

# Format into json files
file_list=($(ls "$save_path" | grep ".txt"))

for ((i=0; i<${#file_list[@]}; i++)); do
    for ((j=i+1; j<${#file_list[@]}; j++)); do
        
        name=${file_list[i]::-4}-${file_list[j]::-4}.json

        srun python ./evaluation/JudgeLM-main/judgelm/data/JudgeLM/judgelm_preprocess.py \
        --ans1_file_path $save_path${file_list[i]} \
        --ans2_file_path $save_path${file_list[j]} \
        --save_path ${save_path}formated_data/${name}

    done
done

echo "formatting into json done"

#Delete txt files 
rm -r ${save_path}/*.txt

echo "txt files removed"

#Judge
file_list_2=($(ls "${save_path}formated_data/"))

for file in "${file_list_2[@]}"; do
    python ./evaluation/JudgeLM-main/judgelm/llm_judge/gen_model_judgement.py \
    --model-path BAAI/JudgeLM-${params}B-v1.0 \
    --model-id ${params}b-JudgeLM \
    --question-file ${save_path}formated_data/$file \
    --answer-file $save_path$file \
    --num-gpus-per-model 1 \
    --num-gpus-total 1 \
    --temperature 0.2 \
    --if-fast-eval 1 
done

echo "Judgement files created"

#Delete json files
rm -r ${save_path}formated_data
echo "json files removed"

#Format into instance level
srun python ./evaluation/scripts/format_ilv.py
echo "Judgement files formatted for ./Notebooks/Judge_ranking.ipynb"
