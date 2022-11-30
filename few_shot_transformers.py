from RopesFewShot import RopesFewShot
from transformers import GPTJForCausalLM , AutoTokenizer
import torch

R = RopesFewShot()
prompts, targets = R.get_prompts_and_targets(num_prompts=1)

if torch.cuda.is_available():
	model = torch.nn.DataParallel(GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)).cuda()
else:
	model =  GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model.eval()

#for prompt, target in zip(prompts, targets):
#	print(f'generated: {generated_text}\ntarget:{target}')
