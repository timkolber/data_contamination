from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelAndTokenizer:
    def __init__(
        self,
        model_name_or_path: str,
        max_input_length: int = 1024,
        max_output_length: int = 512,
        temperature: float = 0.5,
    ):
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.temperature = temperature

    def generate_response(self, prompt_str: str):
        prompt = [{"role": "user", "content": prompt_str}]
        prompt = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,  # type: ignore
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
        )
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_output_length,
            temperature=self.temperature,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    model_paths = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/phi-4",
    ]
    for model_path in model_paths:
        model = ModelAndTokenizer(model_path)
        # Example prompt for the models. Few-shot (1-shot) learning seems necessary to get the models to follow instructions.
        prompt = """
        Please try to fill in a wrong option into [MASK]. Provide only that option. The answer should be different from the other options.
        
        Example:
        Question: Which of the following is a bird?
        Options:
        A. Dove
        B. [MASK]
        C. Fox
        D. Dolphin

        Answer: Dog

        Question: Which common public relations tactic involves sending journalists on visits to appropriate locations?
        Options:
        A. Media release
        B. Media tour
        C. Press room
        D. [MASK]
        Answer: 
        """
        print(model.generate_response(prompt))
