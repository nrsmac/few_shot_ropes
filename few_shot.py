from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json

def get_prompt(background, situation, questions, answers):
    # TODO - adapt to allow for variable number of questions

    return f"""Given the following information, answer the questions below:
{background} {situation}
Q: {questions[0]}
A: {answers[0]}
Q: {questions[1]} 
A: {answers[1]}
A: {answers[2]}
Q: {questions[3]}
A:"""

ropes = json.loads(open("validation_condensed.json", 'r').read()) 
print(ropes[20])

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

for i, r in enumerate(tqdm(ropes)):
    if i == 10:
        break
    background, situation, questions, answers = r.values()
    prompt = get_prompt(background, situation, questions, answers)
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

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