from typing import Dict, List, Optional, Callable

class Filter:
    def __init__(self, prompt: str, input_fields: Optional[List]=None):
        self.prompt = prompt
        self.input_fields = input_fields
    
    def get_full_prompt(self, input: Dict) -> str:
        return f"""Given the following input, please provide a response based on the prompt:
        ====INPUT====
        {input}
        ====PROMPT====
        {self.prompt}
        ====INSTRUCTIONS====
        OUTPUT SHOULD BE EXACTLY EITHER "TRUE" OR "FALSE".
       """

    def execute(self, llm: Callable, input: Dict) -> bool:
        if self.input_fields:
            input = {field: input[field] for field in self.input_fields}
        full_prompt = self.get_full_prompt(input)
        response = llm(full_prompt).strip().upper()
        if response == "TRUE":
            return True
        elif response == "FALSE":
            return False
        else:
            raise ValueError(f"Invalid response from LLM: {response}. Expected 'TRUE' or 'FALSE'.")
