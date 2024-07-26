from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

# Load the model and tokenizer
model_name = "valpy/prompt-classification"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to truncate text to a maximum character length
def truncate_text(text, max_char_len=2000):
    return text[:max_char_len]

# Function to get logits for each entry in the data
def get_logits_for_data(data, model, tokenizer, max_char_len=2000):
    logits_dict = {}
    for key, text in data.items():
        if isinstance(text, str) and (key.startswith("human_turn_") or key.startswith("ai_turn_") or key.startswith("Prompt_")):
            truncated_text = truncate_text(text, max_char_len)
            inputs = tokenizer(truncated_text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                logits_dict[key] = logits.tolist()  # Convert to list for easier readability
    return logits_dict

# Function to get probabilities for each entry in the data
def get_probabilities_for_data(data, model, tokenizer, max_char_len=2000):
    probabilities_dict = {}
    for key, text in data.items():
        if isinstance(text, str) and (key.startswith("human_turn_") or key.startswith("ai_turn_") or key.startswith("Prompt_")):
            truncated_text = truncate_text(text, max_char_len)
            inputs = tokenizer(truncated_text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                probabilities_dict[key] = probabilities.tolist()  # Convert to list for easier readability
    return probabilities_dict

# Function to get likelihoods for each entry in the data
def get_likelihoods_for_data(data, model, tokenizer, max_char_len=2000):
    likelihoods_dict = {}
    for key, text in data.items():
        if isinstance(text, str) and (key.startswith("human_turn_") or key.startswith("ai_turn_") or key.startswith("Prompt_")):
            truncated_text = truncate_text(text, max_char_len)
            inputs = tokenizer(truncated_text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                likelihoods = torch.log(probabilities)
                likelihoods_dict[key] = likelihoods.tolist()  # Convert to list for easier readability
    return likelihoods_dict

# Read input data from JSONL file
input_file = 'wildchat_subset_en_2k_prompting_Mistral-7B-Instruct-v0.3.jsonl'
output_data = []
n_lines = 2  # Set the number of lines to process

with open(input_file, 'r') as file:
    for i, line in enumerate(file):
        if i >= n_lines:
            break
        data = json.loads(line.strip())

        print('Read in lines')
        
        # Get logits, probabilities, and likelihoods for the data
        logits_dict = get_logits_for_data(data, model, tokenizer)
        probabilities_dict = get_probabilities_for_data(data, model, tokenizer)
        likelihoods_dict = get_likelihoods_for_data(data, model, tokenizer)
        
        print('Got values in lines')

        # Append results to the original data
        for key in data.keys():
            if key in logits_dict:
                output_entry = {
                    "key": key,
                    "text": data[key],
                    "logits": logits_dict[key],
                    "probabilities": probabilities_dict[key],
                    "likelihoods": likelihoods_dict[key]
                }
                output_data.append(output_entry)

# # Write results to a JSONL file
# output_file = 'output_data.jsonl'
# with open(output_file, 'w') as jsonl_file:
#     for entry in output_data:
#         jsonl_file.write(json.dumps(entry) + '\n')

# Print the results (optional)
print(json.dumps(output_data, indent=4))