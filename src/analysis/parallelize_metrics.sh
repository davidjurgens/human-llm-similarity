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

export CUDA_VISIBLE_DEVICES=0

#metric_groups=("end,lexical,capitalization,grammar,punctuation" "pos,sbert,topic,sentiment,formality" "politeness,toxicity,subjectivity" "factuality,constituency,readability", "semantic")
# Need to install: textblob, spacy, benepar
# luar throws a bug
# without subjectivity or factuality or constituency or luar
metric_groups=("end,lexical,capitalization,grammar,punctuation,pos,sbert,topic" "sentiment,formality,politeness,toxicity,readability,semantic,liwc")

#input_path=/shared/0/projects/research-jam-summer-2024/data/english_only/prompting_results_clean/wildchat_subset_en_2k_prompting_Meta-Llama-3.1-70B-Instruct.jsonl
##input_path=../data/llama_3.1-70B_100_sample.jsonl
#output_directory=../data/llama3.1_70B_en_2k_metrics/
#output_file=${output_directory}/wildchat_subset_en_2k_prompting_Meta-Llama-3.1-70B-Instruct.jsonl

#input_path=/shared/0/projects/research-jam-summer-2024/data/english_only/prompting_results_clean/wildchat_subset_en_2k_prompting_Meta-Llama-3.1-70B-Instruct.jsonl
##input_path=../data/llama_3.1-70B_100_sample.jsonl
#output_directory=../data/llama3.1_70B_en_2k_metrics/
#output_file=${output_directory}/wildchat_subset_en_2k_prompting_Meta-Llama-3.1-70B-Instruct.jsonl

#input_path=../data/mixtral_100_sample.jsonl
#output_directory=../data/mixtral_100_sample_metrics
#tmp_path=${output_directory}/tmp
#output_file=${output_directory}/mixtral_100_sample.jsonl

input_path=../data/llama_3.1-70B_100_sample.jsonl
output_directory=../data/llama_3.1-70B_100_sample_metrics
tmp_path=${output_directory}/tmp
output_file=${output_directory}/llama_3.1-70B_100_sample.jsonl

clean=false

mkdir -p ${output_directory}
mkdir -p ${tmp_path}

for i in "${!metric_groups[@]}"; do

  if [ ! -f "${output_directory}/${metric_groups[$i]}.jsonl" ]; then
    python analysis/metrics.py \
      --input_path ${input_path} \
      --output_path ${tmp_path}/${metric_groups[$i]}.jsonl \
      --metrics ${metric_groups[$i]} &
  fi
done

wait

python analysis/merge_metrics_files.py \
  --input_dir ${tmp_path} \
  --output_path ${output_file}

if [ ${clean} = true ] ; then

  # Clean up intermediate files
  for i in "${!metric_groups[@]}"; do
    rm ${output_directory}/${metric_groups[$i]}.jsonl
  done

fi
