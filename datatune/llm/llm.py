from typing import List, Optional, Union
import litellm
from datatune.llm.openapibatchutils import multiple_jsonl_extract,multiple_jsonl_export
from openai import OpenAI
import asyncio


class LLM:
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        if "temperature" not in kwargs:
            self.kwargs["temperature"] = 0.0

    def _completion(self, prompt: str) -> Union[str, Exception]:
        messages = [{"role": "user", "content": prompt}]
        from litellm import completion

        response = completion(model=self.model_name, messages=messages, **self.kwargs)

        if isinstance(response, Exception):
            return response
        return response["choices"][0]["message"]["content"]

    def _batch_completion(self, prompts: List[str]) -> List[Union[str, Exception]]:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        from litellm import batch_completion

        responses = batch_completion(
            model=self.model_name, messages=messages, **self.kwargs
        )

        ret = []
        for response in responses:
            if isinstance(response, Exception):
                ret.append(response)
            else:
                ret.append(response["choices"][0]["message"]["content"])

        return ret

    def __call__(self, prompt: Union[str, List[str]]) -> List[str]:
        """Always return a list of strings, regardless of input type"""
        if isinstance(prompt, str):
            return [self._completion(prompt)]
        return self._batch_completion(prompt)


class Ollama(LLM):
    def __init__(
        self, model_name="gemma3:4b", api_base="http://localhost:11434", **kwargs
    ) -> None:
        super().__init__(
            model_name=f"ollama_chat/{model_name}", api_base=api_base, **kwargs
        )
        self.api_base = api_base
        self._model_name = model_name


class OpenAI(LLM):
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        kwargs.update({"api_key": api_key})
        super().__init__(model_name=f"openai/{model_name}", **kwargs)


class Azure(LLM):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        azure_model = f"azure/{self.model_name}"
        azure_params = {
            "api_key": api_key,
            "api_base": api_base,
            "api_version": api_version,
        }

        azure_params = {k: v for k, v in azure_params.items() if v is not None}

        kwargs.update(azure_params)
        super().__init__(model_name=azure_model, **kwargs)

class Gemini(LLM):
    def __init__(
        self,
        model_name: str = "gemini/gemini-1.5-pro",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> None:
        gemini_model = model_name if model_name.startswith("gemini/") else f"gemini/{model_name}"

        gemini_params = {
            "api_key": api_key,
            "api_base": api_base,
        }

        gemini_params = {k: v for k, v in gemini_params.items() if v is not None}
        kwargs.update(gemini_params)

        super().__init__(model_name=gemini_model, **kwargs)

class HuggingFace(LLM):
        def __init__(
            self,
            model_name: str,
            api_base: Optional[str] = None,
            api_key: Optional[str] = None,
            **kwargs,
        ) -> None:
         
            
            hf_model = f"huggingface/{model_name}"
            hf_params = {
                "api_base": api_base,
                "api_key": api_key,
            }
            hf_params = {k: v for k, v in hf_params.items() if v is not None}

            kwargs.update(hf_params)
            super().__init__(model_name=hf_model, **kwargs)

class OpenAIBatchAPI(LLM):
    def __init__(
            self,
            model_name:str = "gpt-3.5-turbo",
            api_key:Optional[str] = None,
            **kwargs
    ):
        kwargs.update({"api_key": api_key})
        super().__init__(model_name=model_name, **kwargs)

    async def process_batch(self,file_path: str, batch_id: int):
        with open (file_path,"rb") as f:
            file_bytes = f.read()
        file_obj = await litellm.acreate_file(
            file=file_bytes,
            purpose="batch",
            custom_llm_provider="openai",
        )
        print(f"[{file_path}] Uploaded as File ID: {file_obj.id}")

        create_batch_response = await litellm.acreate_batch(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=file_obj.id,
            custom_llm_provider="openai",
            metadata={"batch": f"batch-{batch_id}"},
        )
        print(f"[{file_path}] Batch Created: {create_batch_response.id}")

        # Polling loop
        MAX_WAIT_TIME = 300
        POLL_INTERVAL = 5
        MAX_FAILURE_RETRIES = 100

        waited = 0
        failure_count = 0

        while True:
            retrieved_batch = await litellm.aretrieve_batch(
                batch_id=create_batch_response.id,
                custom_llm_provider="openai"
            )
            status = retrieved_batch.status
            print(f"[{file_path}]  Batch status: {status}")

            if status == "completed" and retrieved_batch.output_file_id:
                print(f"[{file_path}] ✅ Completed. Output File ID: {retrieved_batch.output_file_id}")
                break

            elif status in ["failed", "cancelled", "expired"]:
                failure_count += 1
                print(f"[{file_path}] ⚠️ Batch in failed state ({status}), retrying {failure_count}/{MAX_FAILURE_RETRIES}")
                if failure_count >= MAX_FAILURE_RETRIES:
                    raise RuntimeError(f"[{file_path}] ❌ Batch failed with status: {status} after {MAX_FAILURE_RETRIES} retries")

            await asyncio.sleep(POLL_INTERVAL)
            waited += POLL_INTERVAL
            if waited > MAX_WAIT_TIME:
                raise TimeoutError(f"[{file_path}] ❌ Timed out waiting for batch to complete.")


        # Download and save the output file
        file_content = await litellm.afile_content(
            file_id=retrieved_batch.output_file_id,
            custom_llm_provider="openai"
        )
        output_bytes = await file_content.aread()
        output_str = output_bytes.decode("utf-8")

        with open(f"response_{batch_id}.jsonl", "w", encoding='utf-8') as f:
            f.write(output_str)
        print(f"[{file_path}]  Output saved to response_{batch_id}.jsonl")

    async def main(self):
        tasks = [self.process_batch(file_path, i) for i, file_path in enumerate(self.input_files)]
        await asyncio.gather(*tasks)
        ret = multiple_jsonl_extract([f"response_{i}.jsonl" for i in range (0,self.no_of_rows,self.batch_size)])
        return ret


    def __call__(self, prompt:List[str]) -> List[str]:
        """Always return a list of strings, regardless of input type"""
        self.no_of_rows = len(prompt)
        self.batch_size = 50000
        self.input_files = multiple_jsonl_export(prompt,self.model_name,self.no_of_rows,self.batch_size)
        print("Input JSON files created.")
        
        return asyncio.run(self.main())