from RopesFewShot import RopesFewShot
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from datasets import load_dataset

R = RopesFewShot()
prompts, targets = R.get_prompts_and_targets(num_prompts=10)

dataset = load_dataset['ropes']

# Load SetFit model from Hub
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# Create trainer
trainer = SetFitTrainer(
	model=model,
	train_dataset=dataset['train'],
	eval_dataset=targets[],
	loss_class=CosineSimilarityLoss,
	batch_size=16,
	num_iterations=20, # Number of text pairs to generate for contrastive learning
	num_epochs=1 # Number of epochs to use for contrastive learning
)

#for prompt, target in zip(prompts, targets):
#	print(f"Prompt: {prompt[:100]}...\ntarget:{target}")
#	input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#	generated_ids = model.generate(input_ids, max_length=100, do_sample=True, temperature=0.3)
#	generated_text = tokenizer.decode(generated_ids[0])
#	print(generated_text)
