from datasets import load_dataset
import numpy as np

class RopesFewShot:
    def __init__(self, max_questions=None, split='train'):
        self.data = load_dataset('ropes')[split]
        self.max_questions = max_questions
        self.split_len = len(self.data)

    def _get_qa(self, background):
        """Get questions and answers from a background"""
        df = self.data.filter(lambda x: x['background'] == background)
        questions = df['question']
        answers = [a['text'][0] for a in df['answers']]
        return questions, answers
    
    def _get_prompt_and_target(self, background, situation, questions, answers):
        """Get prompt and target from a background, situation, questions and answers"""
        prompt = f'Background: {background}\nSituation: {situation}\n'
        for q, a in list(zip(questions, answers))[:-1]:
            prompt += f'###\nQuestion: {q}\nAnswer: {a}\n'

        prompt += f'###\nQuestion: {questions[-1]}\nAnswer:'
        return prompt, answers[-1]

    def get_prompts_and_targets(self, num_prompts=None):
        """Get prompts and targets"""
        num_prompts = num_prompts if num_prompts else self.split_len

        prompts = []
        targets = []
        
        backgrounds = np.unique(self.data['background'][:num_prompts])
        situations = np.unique(self.data['situation'][:num_prompts])
        for background, situation in zip(backgrounds, situations):
            questions, answers = self._get_qa(background)
            prompt, target = self._get_prompt_and_target(background, situation, questions, answers)
            prompts.append(prompt)
            targets.append(target)

        return prompts, targets


def main():
    ropes = RopesFewShot()
#    sample = ropes.data['train'][0]
#    background = sample['background']
#    situation = sample['situation']
#    questions, answers = ropes._get_qa(background)
#    prompt, target = ropes._get_prompt_and_target(background, situation, questions, answers)

    prompts, targets = ropes.get_prompts_and_targets(num_prompts=100)
    print(prompts[0])

if __name__ == "__main__":
    main()
