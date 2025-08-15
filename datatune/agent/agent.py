from abc import ABC
import dask.dataframe as dd
from typing import Optional, List
import traceback
from datatune.agent.runtime import Runtime
from datatune.llm.llm import LLM


class Agent(ABC):

    TEMPLATE = {
    "dask": {

        "add_column": "df['{new_column}'] = df['{source_column}'] {operator} {value}",
        "group_by": "df = df.groupby('{group_column}').agg({aggregations})",
        "filter_rows": "df = df[df['{column}'] {operator} {value}]"
    },
    "primitive": {

        "Map": """
                mapped = dt.Map(
                    prompt="{prompt}",
                    input_fields={input_fields},
                    output_fields={output_fields}
                )(llm, df)
                df = mapped
               """,
        "Filter":"""
                filtered = dt.Filter(
                    prompt="{prompt}",
                    input_fields={input_fields}
                )(llm, df)
                df = filtered
            """
    }
}

    def get_persona_prompt(self, goal: str) -> str:
        persona_prompt: str = """
        You are a planning agent. Your goal is to generate a **step-by-step JSON plan** to transform a Dask DataFrame `df` according to the following TASK:

        TASK: {goal}

        RULES:
        1. Each step in the plan must be a JSON object with the following fields:
        - "type": either "dask" or "primitive"
        - "operation": the operation name (for dask use template keys like "add_column", "group_by"; for primitive use "Map" or "Filter")
        - "params": dictionary of parameters for Dask templates (skip for primitive)
        - "subprompt": for primitive operations, the LLM prompt describing the transformation
        - "input_fields": (optional) list of input columns for primitive
        - "output_fields": (optional) list of output columns for primitive (only for Map)
        2. The plan must be **an array of steps**, in exact execution order.
        3. Only return valid JSON. Do not include any explanations, comments, or extra text.

         PRIMITIVES CONTEXT:
        - Map: Use this to create new columns from existing data by applying a transformation. Specify the LLM prompt in "subprompt", the input columns in "input_fields", and the new columns in "output_fields". Produces a Dask DataFrame.
        - Filter: Use this to remove rows that do not meet a certain condition. Specify the LLM prompt in "subprompt" and the columns it applies to in "input_fields". Produces a Dask DataFrame.

        Available Dask operations for transforming the dataframe:
        - add_column: assign a new column using an expression
        - group_by_agg: group by one or more columns and aggregate
        - shift_column: create a column by shifting another column
        - merge: merge with another dataframe
        - apply_rowwise: row-wise operation across multiple columns
        - apply_series: element-wise operation on a single column

        RULES FOR CHOOSING OPERATION TYPE:
        1. If the transformation can be performed using standard Dask operations (e.g., adding a column, grouping, shifting, applying row-wise or column-wise functions), use "type": "dask".
        2. If the transformation requires understanding natural language, extracting or interpreting textual content, or involves multiple columns with semantic reasoning, use "type": "primitive".
        3. For primitive operations, use Map for generating new columns from text/semantic analysis and Filter for row-level filtering based on text/criteria.
        4. Always choose the simplest approach: use Dask if it can achieve the step; only use primitives if Dask alone cannot.

        EXAMPLE OUTPUT:
        [
        {
            "type": "primitive",
            "operation": "Map",
            "subprompt": "Extract category and sub-category from industry",
            "input_fields": ["Industry"],
            "output_fields": ["Category","Sub-Category"]
        },
        {
            "type": "primitive",
            "operation": "Filter",
            "subprompt": "Keep only organizations in Africa",
            "input_fields": ["Country"]
        },
        {
            "type": "dask",
            "operation": "add_column",
            "params": {
            "new_column": "Year",
            "expression": "df['Date'].dt.year"
            }
        },
        {
            "type": "dask",
            "operation": "group_by_agg",
            "params": {
            "group_columns": ["Year", "Month"],
            "aggregations": "{'Gross Amount':'sum'}"
            }
        }
        ]

        Generate the JSON plan for the following TASK:

        TASK:{goal}

        """
        persona_prompt = persona_prompt.format(goal=goal)
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