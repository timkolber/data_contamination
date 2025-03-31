# Data Contamination Project

## Usage

- Install Python 3.11
- Install Python Dependency Manager (https://pdm-project.org/en/latest/)
- Type ```pdm install``` in repository root directory

  ## Code explanation
  
- data.py contains all code related to loading the different datasets and making sure they have the correct columns and column names to run ts_guessing.py.
    - i.e. a "choices" column is renamed to "answers"
    - if a dataset contains only an "answerKey" column, which usually contains the letter or index of the correct answer, we create a "correct_answer" column that contains the actual answer

- model.py contains the code to load the models and their respective tokenizers
  
- ts_guessing.py 
    - the actual code to mask wrong answers
    - running the evaluation loop
    - return the accuracy of a model over the whole dataset

To run the evaluation, you run the main.py script. With the argument `--model` you can specify the model name or path from huggingface. `--dataset` specifies the dataset to evaluate contamination for. With `--output_dir` you can select an output directory for the model responses. 
