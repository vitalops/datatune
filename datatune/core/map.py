from typing import Dict, List, Optional, Callable
import ast


class Map:
    def __init__(
        self,
        prompt: str,
        input_fields: Optional[List] = None,
        output_fields: Optional[List] = None,
    ):
        self.prompt = prompt
        self.input_fields = input_fields
        self.output_fields = output_fields

    def get_full_prompt(self, input: Dict) -> str:
        return f"""Given the following input, please provide a response based on the prompt:
        ====INPUT====
        {input}
        ====PROMPT====
        {self.prompt}
        ====INSTRUCTIONS====
        RESPOND THE RESULT AS A DICT, WITHOUT ANY OTHER TEXT
       """

    def execute(self, llm: Callable, input: Dict) -> Dict:
        if self.input_fields:
            input = {field: input[field] for field in self.input_fields}
        full_prompt = self.get_full_prompt(input)
        response = llm(full_prompt)
        response = ast.literal_eval(response)
        if self.output_fields:
            response = {field: response[field] for field in self.output_fields}
        return response
