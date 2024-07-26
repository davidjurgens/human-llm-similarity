import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import math
from tqdm import tqdm

class PerplexityAnalyzer:
    def __init__(self, model_name: str):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set the pad token to be the EOS token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id


    def get_perplexity(self, texts: List[str]) -> List[float]:
        """
        Get the perplexity of a list of texts.
        """
        # Tokenize texts
        encodings = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(self.device)
        
        perplexities = []
        # Iterate over each text separately
        for i in tqdm(range(len(texts)), desc="Calculating perplexity"):
            nll_sum = 0.0
            token_count = 0
            # Iterate over each token in the sequence
            for idx in range(1, (encodings.attention_mask[i] == 1).sum().item()):
                # Get the input tokens (up to the current token)
                inputs = encodings.input_ids[i, :idx].unsqueeze(0)
                # Get the target token
                target = encodings.input_ids[i, idx].unsqueeze(0).unsqueeze(0)
                
                # Inference
                with torch.no_grad():
                    outputs = self.model(inputs)
                    logits = outputs.logits[:, -1, :]
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    nll = -log_probs.gather(1, target).squeeze(1)
                
                nll_sum += nll.item()
                token_count += 1
            
            # Calculate the perplexity
            ppl = math.exp(nll_sum / token_count)
            perplexities.append(ppl)
        
        return perplexities
    

if __name__ == "__main__":

    analyzer = PerplexityAnalyzer('lucadiliello/bart-small')
    result = analyzer.get_perplexity(["This is a test phrase.", "This is another test phrase."])
    print(result)