""" This script condenses the questions ROPES dataset, grouped by background and situation, as a JSON file """
import pandas as pd
from datasets import load_dataset
import json

SPLIT = "test"

def main():
    ropes = load_dataset("ropes", split=SPLIT)
    df = pd.concat([ropes.to_pandas()])

    bgs = df['background'].unique().tolist()
    sits = df['situation'].unique().tolist()

    condensed = []
    for _, (bg, sit) in enumerate(zip(bgs, sits)):
        questions = df[df.background == bg].question.tolist()
        answers = [i['text'].tolist()[0] for i in df[df.background == bg].answers.tolist()]
        condensed.append({"background":bg, "situation":sit, "questions":questions, "answers":answers})

    with open(f"{SPLIT}_condensed.json", 'w') as f:
        json.dump(condensed, f)

if __name__ == "__main__":
    main()
