from abc import ABC
import dask.dataframe as dd
from typing import Optional, List
import traceback
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

        ### Dask Rules:
        - Only use `.map(...)` on a Dask **Series**, not the full DataFrame. For example: `df['new_col'] = df['existing_col'].map(lambda x: ..., meta='object')`
        - For row-wise operations across multiple columns, use `.apply(..., axis=1)` on the DataFrame. Always include the `meta` argument. Example: `df['new_col'] = df.apply(lambda row: ..., axis=1, meta='object')`
        - Always specify `meta='object'` when using `.map` or `.apply` with Dask

        To solve certain tasks, you might need to create new columns as a preprocessing step. These comlumns can be created using regular dask operations or using the Map operation from datatune, depending on whether you need to use an LLM or not.
        The dataframe you are working with is a Dask DataFrame and is available as the variable `df`.
        
        The current schema of df and your overall goal will be available below. 
        You will be provided a goal. Once your generated code directly and completely fulfills that goal (i.e., the described transformation has been performed on df), you must conclude your response with:
        ```python
        DONE = True
        ```
        ALWAYS RETURN VALID PYTHON CODE THAT CAN BE EXECUTED TO PERFORM THE DESIRED OPERATION ON df IN THE RUNTIME ENVIRONMENT DESCRIBED ABOVE. KEEP IT SIMPLE. NO NEW FUNCTIONS, CLASSES OR IMPORTS UNLESS ABSOLUTELY NECESSARY.
        You must return only executable Python code. Do not include any comments, explanations, or Markdown. Your response must be a clean Python code snippet that can be directly passed to exec() without modification.
        Only use functions and attributes of dask on df
        If the next operation requires you to look at the actual data in df, you can set global variable `QUERY` to True and use the `df.head(..)` method or any valid dask operation that returns a dataframe and set it to the global variable `OUTPUT_DF`. E.g:
        ```python
        QUERY = True
        OUTPUT_DF = df.head(10)
        ```
        Make sure to not ask for too many rows at once, as the output will be limited to a few rows. If you need to see more data, you can ask for it in subsequent steps.

        """
        return persona_prompt

    def get_schema_prompt(self, df: dd.DataFrame) -> str:
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

    def get_error_prompt(self, error_msg: str, failed_code: str) -> str:
        error_prompt: str = f"""
        The previous code execution failed with the following error:
        Error: {error_msg}
        
        Failed code:
        {failed_code}
        
        Please fix the error and provide corrected code. Make sure to:
        1. Address the specific error mentioned above
        2. Follow all the Dask rules mentioned in the persona
        3. Ensure the code is syntactically correct
        4. Test your logic before providing the response
        """
        return error_prompt

    def get_full_prompt(
        self,
        df: dd.DataFrame,
        goal: str,
        prev_agent_query: Optional[str] = None,
        output_df: Optional[dd.DataFrame] = None,
        error_msg: Optional[str] = None,
        failed_code: Optional[str] = None,
    ) -> str:
        prompt = self.get_persona_prompt()
        prompt += self.get_schema_prompt(df)
        prompt += self.get_goal_prompt(goal)
        
        if prev_agent_query and output_df is not None:
            prompt += self.get_context_prompt(prev_agent_query, output_df)
        
        if error_msg and failed_code:
            prompt += self.get_error_prompt(error_msg, failed_code)
            
        return prompt

    def _preproc_code(self, code: List[str]) -> str:
        # TODO: make more robust
        code = code[0]
        lines = code.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].endswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _execute_llm_output(self, llm_output: str) -> tuple[bool, Optional[str]]:
        """
        Execute LLM output and return (success, error_message)
        """
        try:
            self.runtime.execute(llm_output)
            
            # Check if task is done first
            if self.runtime.get("DONE", False):
                _ = self.runtime["df"].head()
                return True, None
                
            # Handle query case
            if self.runtime.get("QUERY", False):
                self.runtime["QUERY"] = False
                self.prev_query = llm_output  # Store the query code
                self.output_df = self.runtime["OUTPUT_DF"]
                
                # Check again if DONE was set to True in the same execution
                if self.runtime.get("DONE", False):
                    _ = self.runtime["df"].head() 
                    return True, None
                    
            return False, None
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return False, error_msg

    def __init__(self, llm: LLM):
        self.llm = llm

    def _set_df(self, df: dd.DataFrame):
        self.df = df
        runtime = self.runtime = Runtime()
        runtime["df"] = df
        runtime["DONE"] = False
        runtime["QUERY"] = False
        runtime.execute(
            "import numpy as np\nimport pandas as pd\nimport dask.dataframe as dd\nimport datatune as dt"
        )
        self.prev_query = None
        self.output_df = None

    def do(self, task: str, df: dd.DataFrame, max_iterations: int = 5) -> dd.DataFrame:
        """
        Execute task with evaluation loop and error handling
        
        Args:
            task: The task description
            df: The input dataframe
            max_iterations: Maximum number of correction attempts (default: 5)
        """
        self._set_df(df)
        
        iteration = 0
        done = False
        last_error = None
        last_failed_code = None
        
        while not done and iteration < max_iterations:
            iteration += 1
            print(f"Iteration {iteration}/{max_iterations}")
            
            # Build prompt with error context if available
            prompt = self.get_full_prompt(
                df, 
                task, 
                self.prev_query, 
                self.output_df,
                last_error,
                last_failed_code
            )
            
            # Get LLM response
            try:
                llm_output = self.llm(prompt)
                code = self._preproc_code(llm_output)
                
                if not code:
                    last_error = "LLM output is empty or invalid"
                    last_failed_code = llm_output
                    print(f"  Error: {last_error}")
                    continue
                
                print(f"  Executing code:\n{code[:200]}{'...' if len(code) > 200 else ''}")
                
                # Execute and handle results
                success, error_msg = self._execute_llm_output(code)
                
                if success:
                    done = True
                    print(f"  ✅ Task completed successfully in iteration {iteration}")
                elif error_msg:
                    last_error = error_msg
                    last_failed_code = code
                    print(f"  ❌ Execution error: {error_msg.split(chr(10))[0]}")  # Print first line of error
                else:
                    # Code executed successfully but task not done (probably QUERY=True case)
                    print(f"  🔄 Code executed, continuing...")
                    # Reset error context since this iteration was successful
                    last_error = None
                    last_failed_code = None
                    
            except Exception as e:
                last_error = f"LLM call failed: {str(e)}"
                last_failed_code = None
                print(f"  ❌ LLM error: {last_error}")
        
        if not done:
            if last_error:
                raise RuntimeError(
                    f"Agent failed to complete task after {max_iterations} iterations. "
                    f"Last error: {last_error}"
                )
            else:
                raise RuntimeError(
                    f"Agent failed to complete task after {max_iterations} iterations. "
                    f"Task may be too complex or require more iterations."
                )
        
        return self.runtime["df"]