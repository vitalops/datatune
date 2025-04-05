from typing import Dict, List, Optional, Callable

class Filter:
    def __init__(self, prompt: str, input_fields: Optional[List]=None):
        self.prompt = prompt
        self.input_fields = input_fields
    
    def get_full_prompt(self, input: Dict) -> str:
        return f"""SAY TRUE OR FALSE TO THE FOLLOWING PROMPT:
{self.prompt}
INPUT: {input}
INSTRUCTIONS: OUTPUT JUST TRUE OR FALSE WITHOUT ANY OTHER TEXT"""


    def process_output(self, output: str) -> bool:
        output = output.strip().upper()
        if output == "TRUE":
            return True
        elif output == "FALSE":
            return False
        else:
            raise ValueError(f"Invalid response from LLM: {output}. Expected 'TRUE' or 'FALSE'.")


    def execute(self, llm: Callable, input: Dict) -> bool:
        if self.input_fields:
            input = {field: input[field] for field in self.input_fields}
        full_prompt = self.get_full_prompt(input)
        print(f"Full Prompt: {full_prompt}")
        response = llm(full_prompt)
        return self.process_output(response)

