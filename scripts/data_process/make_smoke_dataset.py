import argparse
import os

import pandas as pd


PROMPT_TEMPLATE = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""


def make_row(question: str, answers: list[str], split: str, idx: int) -> dict:
    question = question.strip()
    if question and question[-1] != "?":
        question += "?"
    return {
        "data_source": "smoke",
        "prompt": [{
            "role": "user",
            "content": PROMPT_TEMPLATE.format(question=question),
        }],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target": answers,
            },
        },
        "extra_info": {
            "split": split,
            "index": idx,
        },
    }


def build_examples() -> tuple[list[dict], list[dict]]:
    train = [
        make_row("What is the capital of France", ["Paris"], "train", 0),
        make_row("Who wrote Pride and Prejudice", ["Jane Austen"], "train", 1),
        make_row("What planet is known as the Red Planet", ["Mars"], "train", 2),
        make_row("What is the largest ocean on Earth", ["Pacific Ocean", "the Pacific Ocean"], "train", 3),
        make_row("Who painted the Mona Lisa", ["Leonardo da Vinci", "da Vinci"], "train", 4),
        make_row("What is the chemical symbol for gold", ["Au"], "train", 5),
        make_row("Which country is home to Mount Fuji", ["Japan"], "train", 6),
        make_row("Who developed the theory of relativity", ["Albert Einstein", "Einstein"], "train", 7),
    ]
    test = [
        make_row("What is the capital of Italy", ["Rome"], "test", 0),
        make_row("Who discovered penicillin", ["Alexander Fleming", "Fleming"], "test", 1),
        make_row("What gas do plants absorb from the atmosphere", ["Carbon dioxide", "CO2"], "test", 2),
        make_row("What is the tallest mountain in the world", ["Mount Everest", "Everest"], "test", 3),
    ]
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./data/nq_search_smoke")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train, test = build_examples()

    pd.DataFrame(train).to_parquet(os.path.join(args.output_dir, "train.parquet"))
    pd.DataFrame(test).to_parquet(os.path.join(args.output_dir, "test.parquet"))

    print(f"Wrote {len(train)} train rows to {os.path.join(args.output_dir, 'train.parquet')}")
    print(f"Wrote {len(test)} test rows to {os.path.join(args.output_dir, 'test.parquet')}")


if __name__ == "__main__":
    main()
