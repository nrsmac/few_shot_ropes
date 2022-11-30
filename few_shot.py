from RopesFewShot import RopesFewShot
from transformers import AutoTokenizer, GPTJForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

R = RopesFewShot()
prompts, targets = R.get_prompts_and_targets(num_prompts=10)

for prompt, target in zip(prompts, targets):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = model.generate(input_ids, max_length=100, do_sample=True, temperature=0.3)
    generated_text = tokenizer.decode(generated_ids[0])
    print(generated_text)
