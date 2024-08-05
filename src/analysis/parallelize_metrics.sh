#!/bin/bash

# Here is a list of metric options:
# end - simple string manip
# lexical - string manip
# capitalization - string manip
# grammar -
# punctuation - string manip
# pos - neural classifier
# sbert - neural classifier
# semantic - a variety of slower things
# liwc -
# topic - neural classifier
# sentiment - quick neural classifier
# formality - slightly slower neural classifier
# politeness - even slower neural classifier
# toxicity - neural classifier
# subjectivity - nerual classifier
# factuality - neural classifier
# constituency -
# readability - neural classifier

# Metrics that have to be run on burger
# factuality -- needs spacy
# subjectivity -- needs textblob
# constituency -- needs benepar, has to be run under jam env
# grammar -- needs language tools only on burger

export CUDA_VISIBLE_DEVICES=1,2
. /opt/anaconda/etc/profile.d/conda.sh
#metric_groups=("end,lexical,capitalization,grammar,punctuation" "pos,sbert,topic,sentiment,formality" "politeness,toxicity,subjectivity" "factuality,constituency,readability", "semantic")
# without constituency or grammar
metric_groups=("end,lexical,capitalization,punctuation,perplexity" "sentiment,pos,sbert,formality,topic" "politeness,toxicity,readability,semantic,liwc")
#metric_groups=("lexical,capitalization")
# These have to be run in the "jam" environment on taco
jam_metrics="subjectivity,factuality"

### 2k FILES

# Llama3.1-70B
#input_path=/shared/0/projects/research-jam-summer-2024/data/english_only/prompting_results_clean/wildchat_subset_en_2k_prompting_Meta-Llama-3.1-70B-Instruct.jsonl
#output_directory=../data/llama3.1_70B_en_2k_metrics/
#tmp_path=${output_directory}/tmp
#output_file=${output_directory}/

# Llama3.1-8B
#input_path=/shared/0/projects/research-jam-summer-2024/data/english_only/prompting_results_clean/wildchat_subset_en_2k_prompting_Meta-Llama-3.1-8B-Instruct.jsonl
#output_directory=../data/llama3.1_8B_en_2k_metrics/
#tmp_path=${output_directory}/tmp2
##output_file=${output_directory}/wildchat_subset_en_2k_prompting_Meta-Llama-3.1-8B-Instruct.jsonl
#output_file=${output_directory}/fixed_cap_lex.jsonl

# Mistral-7B
#input_path=/shared/0/projects/research-jam-summer-2024/data/english_only/prompting_results_clean/wildchat_subset_en_2k_prompting_Mistral-7B-Instruct-v0.3.jsonl
#output_directory=../data/mistral_en_2k_metrics/
#tmp_path=${output_directory}/tmp2
##output_file=${output_directory}/wildchat_subset_en_2k_prompting_Mistral-7B-Instruct-v0.3.jsonl
#output_file=${output_directory}/fixed_cap_lex.jsonl

# Mixtral-8x7B
#input_path=/shared/0/projects/research-jam-summer-2024/data/english_only/prompting_results_clean/wildchat_subset_en_2k_prompting_Mixtral-8x7B-Instruct-v0.1.jsonl
#output_directory=../data/mixtral_en_2k_metrics/
#tmp_path=${output_directory}/tmp2
##output_file=${output_directory}/wildchat_subset_en_2k_prompting_Mixtral-8x7B-Instruct-v0.1.jsonl
#output_file=${output_directory}/fixed_cap_lex.jsonl

# Phi-3
#input_path=/shared/0/projects/research-jam-summer-2024/data/english_only/prompting_results_clean/wildchat_subset_en_2k_prompting_Phi-3-medium-4k-instruct.jsonl
#output_directory=../data/phi3_2k_metrics/
#tmp_path=${output_directory}/tmp
#output_file=${output_directory}/wildchat_subset_en_2k_prompting_Phi-3-medium-4k-instruct.jsonl

# Qwen2
#input_path=/shared/0/projects/research-jam-summer-2024/data/english_only/prompting_results_clean/wildchat_subset_en_2k_prompting_Qwen2-72B-Instruct.jsonl
#output_directory=../data/qwen2_2k_metrics/
#tmp_path=${output_directory}/tmp
#output_file=${output_directory}/wildchat_subset_en_2k_prompting_Qwen2-72B-Instruct.jsonl


### 100K FILES

# Mixtral-8x7B
input_path=/shared/0/projects/research-jam-summer-2024/data/english_only/100k_results/wildchat_subset_en_100k_Mixtral-8x7B.jsonl
output_directory=../data/mixtral_en_100k_metrics/
tmp_path=${output_directory}/tmp
output_file=${output_directory}/wildchat_subset_en_100k_Mixtral-8x7B.jsonl

# Mistral large
#input_path=/shared/0/projects/research-jam-summer-2024/data/english_only/100k_results/wildchat_subset_en_100k_Mistral-Large-Instruct.jsonl
#output_directory=../data/mistral_en_100k_metrics/
#tmp_path=${output_directory}/tmp
#output_file=${output_directory}/wildchat_subset_en_100k_Mistral-Large-Instruct.jsonl

# Llama3 .1 70B
#input_path=/shared/0/projects/research-jam-summer-2024/data/english_only/100k_results/wildchat_subset_en_100k_Llama-3.1-70B.jsonl
#output_directory=../data/llama3.1_70B_en_100k_metrics/
#tmp_path=${output_directory}/tmp
#output_file=${output_directory}/wildchat_subset_en_100k_Llama-3.1-70B.jsonl


#### SMALL TEST FILES

#input_path=../data/mixtral_100_sample.jsonl
#output_directory=../data/mixtral_100_sample_metrics
#tmp_path=${output_directory}/tmp
#output_file=${output_directory}/mixtral_100_sample.jsonl

#input_path=../data/llama_3.1-70B_100_sample.jsonl
#output_directory=../data/llama_3.1-70B_100_sample_metrics
#tmp_path=${output_directory}/tmp
#output_file=${output_directory}/llama_3.1-70B_100_sample.jsonl

clean=false
# Set this if we're on taco and we just want to run the jam metrics and merge
jam_only=false

mkdir -p ${output_directory}
mkdir -p ${tmp_path}

if [ ${HOSTNAME} = taco ] ; then
  # Run the jam environment metrics
  conda activate jam
  python analysis/metrics.py \
        --input_path ${input_path} \
        --output_path ${tmp_path}/${jam_metrics}.jsonl \
        --metrics ${jam_metrics} &
  conda deactivate
fi

if [ ${jam_only} = false ] ; then

  for i in "${!metric_groups[@]}"; do

    if [ ! -f "${output_directory}/${metric_groups[$i]}.jsonl" ]; then
      python analysis/metrics.py \
        --input_path ${input_path} \
        --output_path ${tmp_path}/${metric_groups[$i]}.jsonl \
        --metrics ${metric_groups[$i]} &
    fi
  done

fi

wait

if [ ${HOSTNAME} = taco ] ; then

  python analysis/merge_metrics_files.py \
    --input_dir ${tmp_path} \
    --output_path ${output_file}

  if [ ${clean} = true ] ; then

    # Clean up intermediate files
    for i in "${!metric_groups[@]}"; do
      rm ${tmp_path}/${metric_groups[$i]}.jsonl
    done

    rm ${tmp_path}/${jam_metrics}.jsonl
  fi
else
  echo "Merged metrics file not created; you need to run the jam metrics on taco in the jam environment first"
fi
