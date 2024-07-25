# human-llm-similarity

## Running trial prompt generation

First, download the Google sheet with all of the trial prompts as a CSV. Then run the following:

```
python src/models/run_trial_prompts.py --input_path [WildChat JSONL location] --prompt_path [Prompt CSV location] --output_path [Save location] --model_path [Model ID]
```

This will run all of the WildChat data specified by `--input_path` through the model specified by `--model_path` using all of the prompt variations in the prompt CSV. It saves a json file where the keys are indices of the rows in the prompt CSV (i.e. corresponding to the prompt template that was used) and each value is a list of responses to each of the conversations in the WildChat data.
