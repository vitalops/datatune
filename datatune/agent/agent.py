from abc import ABC
import dask.dataframe as dd
from typing import Optional
from datatune.agent.runtime import Runtime
from datatune.llm.llm import LLM


class Agent(ABC):

    def get_persona_prompt(self) -> str:
        persona_prompt: str = """You are Datatune Agent, a powerful assistant designed to help users with data processing tasks.
        You are capable of generating python code to perform various operations on data. Apart from python builtins, you have the following libraries avaiable in your run time:
        - pandas
        - numpy
        - dask

        In addition to these, you also have access to the datatune library, which provides functionality for processing data using LLMs.
        Map Example:
        ```python
        import datatune as dt
        import dask.dataframe as dd
        df = dd.read_csv("path/to/data.csv")
        map = dt.Map(prompt="Your prompt here")
        llm = dt.LLM(model_name="gpt-3.5-turbo")
        mapped_df = map(llm, df)
        mapped_df = dt.finalize(mapped_df)
        ```
        Filter Example:
        ```python
        import datatune as dt
        import dask.dataframe as dd
        df = dd.read_csv("path/to/data.csv")
        filter = dt.Filter(prompt="Your prompt here")
        llm = dt.LLM(model_name="gpt-3.5-turbo")
        filtered_df = filter(llm, df)
        filtered_df = dt.finalize(filtered_df)
        ```

        To solve certain tasks, you might need to create new columns as a preprocessing step. These comlumns can be created using regular dask operations or using the Map operation from datatune, depending on whether you need to use an LLM or not.
        The dataframe you are working with is a Dask DataFrame and is available as the variable `df`.
        You are either at step 0 (you are seeing all this for the first time and df is untouched) or at an intermediate step where you have already performed some operations on df. Make sure to not repeat operations that have already been performed on df.
        The current schema of df and your overall goal will be available below. At each step you should ideally perform a single operation on df that brings you closer to the goal. Once you have achieved the goal, simply set global variable `DONE` to True.
        ALWAYS RETURN VALID PYTHON CODE THAT CAN BE EXECUTED TO PERFORM THE DESIRED OPERATION ON df IN THE RUNTIME ENVIRONMENT DESCRIBED ABOVE. KEEP IT SIMPLE. NO NEW FUNCTIONS, CLASSES OR IMPORTS UNLESS ABSOLUTELY NECESSARY.
        If the next operation requires you to look at the actual data in df, you can set global variable `QUERY` to True and use the `df.head(..)` method or any valid dask operation that returns a dataframe and set it to the global variable `OUTPUT_DF`. E.g:
        ```python
        QUERY = True
        OUTPUT_DF = df.head(10)
        ```
        Make sure to not ask for too many rows at once, as the output will be limited to a few rows. If you need to see more data, you can ask for it in subsequent steps.
        """
        return persona_prompt

    def get_schema_prompt(df: dd.DataFrame) -> str:
        schema_prompt: str = f"""The current schema of the dataframe df is as follows:
        {df.dtypes.to_string()}
        """
        return schema_prompt

    def get_goal_prompt(self, goal) -> str:
        goal_prompt: str = f"""Your overall goal is as follows:
        {goal}.
        """
        return goal_prompt

    def get_context_prompt(self, query: str, output_df: dd.DataFrame) -> str:
        context_prompt: str = f"""you previously asked for the output of the following query:
        {query}

        The output dataframe after the last operation is as follows:
        {output_df.head().to_string()}
        """
        # TODO: Hard limit on number of rows
        return context_prompt

    def get_full_prompt(
        self,
        df: dd.DataFrame,
        goal: str,
        prev_agent_query: Optional[str] = None,
        output_df: Optional[dd.DataFrame] = None,
    ) -> str:
        prompt = self.get_persona_prompt()
        prompt += self.get_schema_prompt(df)
        prompt += self.get_goal_prompt(goal)
        if prev_agent_query and output_df is not None:
            prompt += self.get_context_prompt(prev_agent_query, output_df)
        return prompt

    def _preproc_code(self, code: str) -> str:
        # TODO: make more robust
        lines = code.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].endswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _execute_llm_output(self, llm_output: str) -> bool:
        self.runtime.execute(llm_output)
        if self.runtime.get("DONE", False):
            return True
        if self.runtime.get("QUERY", False):
            self.runtime["QUERY"] = False
            self.prev_query = self.runtime.get("QUERY", None)
            self.output_df = self.runtime["OUTPUT_DF"]  # TODO: what if this is not set?
        return False

    def __init__(self, llm: LLM):
        self.llm = llm

    def _set_df(self, df: dd.DataFrame):
        self.df = df
        runtime = self.runtime = Runtime(df)
        runtime["df"] = df
        runtime["DONE"] = False
        runtime["QUERY"] = False
        runtime.execute(
            "import numpy as np\nimport pandas as pd\nimport dask.dataframe as dd\nimport datatune as dt"
        )
        self.prev_query = None
        self.output_df = None

    def do(self, task: str, df: dd.DataFrame) -> dd.DataFrame:
        self._set_df(df)
        prompt = self.get_full_prompt(df, task, self.prev_query, self.output_df)
        done = False
        while not done:
            llm_output = self.llm(prompt)
            code = self._preproc_code(llm_output)
            if not code:
                raise ValueError("LLM output is empty or invalid.")
            done = self._execute_llm_output(code)
        return self.runtime["df"]
