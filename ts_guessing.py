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
        if len(wrong_answers) < 3:
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

        template = f"{question} {a_corr} {masked_options[0]} {masked_options[1]} {masked_options[2]}"

        masked_data.append(
            {
                "question": question,
                "correct_answer": a_corr,
                "masked_answer": a_masked,
                "masked_template": template,
            }
        )

    return masked_data


if __name__ == "__main__":
    dataset = load_data("tau/commonsense_qa")
    masked_output = mask_wrong_answer(dataset)
    for item in masked_output:
        print(item)
