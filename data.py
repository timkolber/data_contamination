from datasets import load_dataset

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_data(dataset_identifier: str):
    if dataset_identifier == "cais/mmlu":
        dataset = load_dataset(dataset_identifier, "all", split="test")
        dataset = dataset.rename_column("choices", "answers")
        dataset = dataset.rename_column("answer", "correct_answer")
    elif (
        dataset_identifier == "ibragim-bad/arc_challenge"
        or dataset_identifier == "ibragim-bad/arc_easy"
    ):
        dataset = load_dataset(dataset_identifier, split="test")
        dataset = dataset.map(
            lambda example: {
                **example,
                "answers": example["choices"]["text"],
            }
        )
        dataset = dataset.map(
            lambda example: {
                **example,
                "correct_answer": example["answers"][
                    (
                        alphabet.index(example["answerKey"])
                        if example["answerKey"] in alphabet
                        else int(example["answerKey"]) - 1
                    )
                ],
            }
        )

    elif dataset_identifier == "allenai/openbookqa":
        dataset = load_dataset(dataset_identifier, split="test")
        dataset = dataset.map(
            lambda example: {
                **example,
                "answers": example["choices"]["text"],
            }
        )
        dataset = dataset.map(
            lambda example: {
                **example,
                "correct_answer": example["answers"][
                    (
                        alphabet.index(example["answerKey"])
                        if example["answerKey"] in alphabet
                        else int(example["answerKey"]) - 1
                    )
                ],
            }
        )
        dataset = dataset.rename_column("question_stem", "question")
    elif dataset_identifier == "openlifescienceai/medmcqa":
        dataset = load_dataset(dataset_identifier, split="test")
        dataset = dataset.map(
            lambda example: {
                **example,
                "answers": [
                    example["opa"],
                    example["opb"],
                    example["opc"],
                    example["opd"],
                ],
            }
        )
        dataset = dataset.map(
            lambda example: {
                **example,
                "correct_answer": example["answers"][example["cop"]],
            }
        )
    return dataset


if __name__ == "__main__":
    dataset_identifiers = [
        "ibragim-bad/arc_easy",
        "cais/mmlu",
        "ibragim-bad/arc_challenge",
    ]
    for dataset_identifier in dataset_identifiers:
        dataset = load_data(dataset_identifier)
        print(len(dataset))
        print(dataset[0])
