import random

from data import load_data


def mask_wrong_answer(dataset):
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

        a_masked = random.choice(wrong_answers)

        masked_options = []
        for answer in wrong_answers:
            if answer == a_corr:
                masked_options.append(a_corr)
            elif answer == a_masked:
                masked_options.append("[MASK]")
            else:
                masked_options.append(answer)

        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        prompt = f""" Please try to fill in a wrong option into [MASK]. Provide only that option. The answer should be different from the other options.

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
                "masked_answer": a_masked,
                "prompt": prompt,
            }
        )

    return masked_data


if __name__ == "__main__":
    dataset = load_data("tau/commonsense_qa")
    masked_output = mask_wrong_answer(dataset)
    for item in masked_output:
        print(item)
