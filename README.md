# human-llm-similarity

## Running trial prompt generation

First, download the Google sheet with all of the trial prompts as a CSV. Then run the following:

```
python src/models/run_trial_prompts.py --input_path [WildChat JSONL location] --prompt_path [Prompt CSV location] --output_path [Save location] --model_path [Model ID]
```

This will run all of the WildChat data specified by `--input_path` through the model specified by `--model_path` using all of the prompt variations in the prompt CSV. 

It creates a json file with the following fields:

```
human_turn_1: Text of the first human turn
ai_turn_2: Text of the LLM response
human_turn_3: Text of the 3rd human turn
hashed_ip: Hash identifying the user
model: The model used for the conversation
country: The country associated with the IP address
language: The language of the conversation
conversation_hash: Hash identifying the conversation
toxic: Boolean indicator for whether or not the conversation was labeled as false
Prompt_N: All of the LLM responses for the prompt with ID N in the original CSV (there will be one key for each prompt)
```
