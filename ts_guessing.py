import random
from typing import Any, Dict, List, cast

import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from data import load_data
from model import ModelAndTokenizer


def mask_wrong_answer(dataset: Dataset, mask_amount: int) -> List[Dict[str, str]]:
    """
    Given the test split of a dataset, mask one of the wrong answers and return a list of dicts with
    the question, the correct answer, the masked answer and the prompt to give to the model.
    """
    masked_data = []
    for item in dataset:
        question = item["question"]
        answers = item["answers"]
        a_corr = item["correct_answer"]

        wrong_answers = [a for a in answers if a != a_corr]

        # Check for sufficient amount of answers (ignore Yes/No and True/False prompts)
        if (
            len(wrong_answers) < 2
        ):  # changed to 2 from 3 because our data has 3 wrong answers
            continue

        a_masked = random.sample(wrong_answers, mask_amount)

        masked_options = []
        for answer in answers:
            if answer == a_corr:
                masked_options.append(a_corr)
            elif answer in a_masked:
                masked_options.append("[MASK]")
            else:
                masked_options.append(answer)

        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        prompt = f""" Please try to fill in a wrong option into each [MASK]. Provide only those options. The answers should be different from the other options.

        Example:
        Question: Which of the following is a bird?
        Options:
        A. Dove
        B. [MASK]
        C. Fox
        D. Dolphin

        Answer: Dog

        Question: {question}
        Options:
        """
        for i, option in enumerate(masked_options):
            prompt += f"{alphabet[i]}. {option}\n"
        # template = f"{question} {a_corr} {masked_options[0]} {masked_options[1]} {masked_options[2]}"

        masked_data.append(
            {
                "question": question,
                "correct_answer": a_corr,
                "masked_answers": a_masked,
                "prompt": prompt,
            }
        )
    return masked_data


def run_ts_guessing_evaluation(
    model: ModelAndTokenizer, dataset: Dataset, output_file: str, mask_amount: int
) -> float:
    """
    Given a model and a dataset, this function runs the loop over the dataset, generates a response for each item
    and returns the accuracy of the model."""
    masked_data = mask_wrong_answer(dataset, mask_amount)
    masked_answers = []
    responses = []
    for item in tqdm(masked_data):
        response = model.generate_response(item["prompt"])
        responses.append(response)
        masked_answers.append(item["masked_answers"])

    dataframe = pd.DataFrame()
    dataframe["masked_answers"] = masked_answers
    dataframe["response"] = responses
    dataframe.to_csv(output_file, index=False)

    accuracy = evaluate_responses_accuracy(masked_data, responses, mask_amount)
    return accuracy


def evaluate_responses_accuracy(
    masked_data: List[Dict[str, str]], responses: List[str], mask_amount: int
) -> float:
    """
    This function evaluates the accuracy of the predictions by checking if the masked answer is anywhere in the response by the model.
    """
    correct = 0
    total = len(masked_data) * mask_amount
    for item, response in zip(masked_data, responses):
        generated_response_part = response[len(item["prompt"]) :]
        for masked_answer in item["masked_answers"]:
            if masked_answer in generated_response_part:
                correct += 1
    return correct / total


if __name__ == "__main__":
    dataset = load_data("ibragim-bad/arc_challenge")
    mask_amount = 1
    masked_output = mask_wrong_answer(dataset, mask_amount)
    for item in masked_output:
        print(item)
