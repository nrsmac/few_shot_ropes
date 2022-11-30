from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json

def get_prompt(background, situation, questions, answers):
    # TODO - adapt to allow for variable number of questions

    return f"""Given the following information, answer the questions below:
{background} {situation}
Question: {questions[0]}
Answer: {answers[0]}
Question: {questions[1]} 
Answer: {answers[1]}
Answer: {answers[2]}
Question: {questions[3]}
Answer:"""

ropes = json.loads(open("validation_condensed.json", 'r').read()) 

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom", device_map='auto', torch_dtype='auto')
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

for i, r in enumerate(tqdm(ropes)):
	if i == 1:
		break
	background, situation, questions, answers = r.values()
	prompt = get_prompt(background, situation, questions, answers)
	
	print(f"Prompt: {prompt}")

	print("Tokenizing prompt: ")
	input_ids = tokenizer(prompt, return_tensors="pt").input_ids

	print("Feeding to model...")
	gen_tokens = model.generate(
		input_ids,
		pad_token_id=tokenizer.eos_token_id,
		do_sample=True,
		temperature=0.33,
		max_length=input_ids.shape[1]+5)

	y_hat = tokenizer.batch_decode(gen_tokens)[0]
	print(f"{y_hat=}")
	pred_answers.append(f"{prompt}:{y_hat} CORRECT:{answers}\n")

out = f"few_shot_gpt-neox-20b_answers.txt"
try:
	with open(out, 'w') as f:
		f.writelines(answers)
except IOError:
	print("I/O error")
