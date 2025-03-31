import argparse

from data import load_data
from model import ModelAndTokenizer
from ts_guessing import run_ts_guessing_evaluation


def main(model_name_or_path: str, dataset_name_or_path: str, output_dir: str):
    model = ModelAndTokenizer(
        model_name_or_path=model_name_or_path,
        max_input_length=1024,
        max_output_length=512,
        temperature=0.5,
    )
    data = load_data(dataset_name_or_path)
    model_name = model_name_or_path.split("/")[-1]
    dataset_name = dataset_name_or_path.split("/")[-1]
    accuracy = run_ts_guessing_evaluation(
        model, data, output_dir + f"/{model_name}-{dataset_name}.txt"
    )
    print(f"Accuracy {model_name_or_path}-{dataset_name}: {accuracy:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ts-guessing script on specified model and dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model name or path we want to check for contamination.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ibragim-bad/arc_easy",
        help="Dataset name or path we want to check for contamination.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save the output in.",
    )
    args = parser.parse_args()
    main(args.model, args.dataset, args.output_dir)
