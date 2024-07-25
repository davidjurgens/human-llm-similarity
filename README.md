# human-llm-similarity

## Running trial prompt generation

First, download the Google sheet with all of the trial prompts as a CSV. Then run the following:

```
python src/models/run_trial_prompts.py --input_path [WildChat JSONL location] --prompt_path [Prompt CSV location] --output_path [Save location] --model_path [Model ID]
```

This will run all of the WildChat data specified by `--input_path` through the model specified by `--model_path` using all of the prompt variations in the prompt CSV. 

It creates a json file where the keys are the Prompt ID in the prompt CSV and each value is a list of responses, one for each of the conversations in the WildChat data.
