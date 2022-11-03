from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def get_prompt(background, situation, question):
    return f"""Given the following information, answer the questions below:
Information: {background} {situation}
Q:{question}
A:"""

ropes = load_dataset("ropes", split="validation")

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

pred_answers = [] 

for i, r in enumerate(tqdm(ropes)):
    if i == 1:
        break
    _, background, situation, question, answers = r.values()
    prompt = get_prompt(background, situation, question)
    if len(answers['text']) > 1: 
        print(f"{i}:{answers}")
    
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

out = f"zero_shot_gpt-j_answers.txt"
try:
    with open(out, 'w') as f:
        f.writelines(pred_answers)
except IOError:
    print("I/O error")
