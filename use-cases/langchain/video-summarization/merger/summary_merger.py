import math
import re
import time

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


class SummaryMerger:
    """
    Merge summaries generated from multiple chunks of text and generate a final summary with an anomaly score
    """

    def __init__(self, model_id="llmware/llama-3.2-3b-instruct-ov", device="CPU", max_new_tokens=512, batch_size=5,
                 chain=None):
        self.ov_llm = None

        if chain is not None:
            # use miniCPM chain passed from summarizers
            print("Running summary merger with pre-built LVM chain without API wrapper\n")
            self.chain = chain

            # modified prompt for minicpm, minicpm doesn't adhere to the llama prompt and always skips anomaly scores.
            # this is the only format that works.
            self.summary_prompt = """Write a response that appropriately completes the request.
            ### Instruction: Please create a summary of the overall video highlighting all the important information. How would you rate the scene described on a scale from 0.0 to 1.0, with 0.0 representing a standard scene and 1.0 denoting a scene with suspicious activities?
            Please organize your answer according to this example:
            **Summary**: A summary of the entire text description highlighting all the important details in less than 10 sentences.
            **Anomaly Score**: A number between 0.0 and 1.0 based on your analysis.
            ### Input: {}\n\n"""

        else:
            print(f"Running summary merger with specified {model_id}\n")

            # openVINO configs for optimized model, apply uint8 quantization for lowering precision of key/value cache in LLMs.
            # apply dynamic quantization for activations
            ov_config = {"PERFORMANCE_HINT": "LATENCY",
                         "NUM_STREAMS": "1",
                         "CACHE_DIR": "ov_llama_cache",
                         "KV_CACHE_PRECISION": "u8",
                         "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
                         }
            # use langchain openVINO pipeline to load the model
            self.ov_llm = HuggingFacePipeline.from_model_id(
                model_id=model_id,
                task="text-generation",
                backend="openvino",
                model_kwargs={
                    "device": device,
                    "ov_config": ov_config,
                    "trust_remote_code": True
                },
                pipeline_kwargs={
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "top_k": 10,
                    "temperature": 0.7,
                    "return_full_text": False,
                    "repetition_penalty": 1.0,
                    "encoder_repetition_penalty": 1.0
                })
            self.ov_llm.pipeline.tokenizer.pad_token_id = self.ov_llm.pipeline.tokenizer.eos_token_id

            self.summary_prompt = """Write a response that appropriately completes the request. 
            ### Instruction: Please create a summary of the overall video highlighting all the important information. How would you rate the scene described on a scale from 0.0 to 1.0, with 0.0 representing a standard scene and 1.0 denoting a scene with suspicious activities? 
            Please organize your answer according to this example:

            **Overall Summary**: A summary of the entire text description in about five sentences or less.
            **Activity Observed**: Key actions observed in the video.
            **Potential Suspicious Activity**: List any activities that might indicate suspicious behavior.
            **Anomaly Score**: A number between 0.0 and 1.0 based on your analysis.

            ### Input: {question}
            ### Answer:"""

            self.prompt = PromptTemplate.from_template(self.summary_prompt)
            # generation_config = {"skip_prompt": True, "pipeline_kwargs": {"max_new_tokens": max_new_tokens}}
            self.chain = self.prompt | self.ov_llm

        self.batch_size = batch_size

    def merge_summaries(self, summaries):
        """
        Merge summaries generated from multiple chunks of text and generate a final summary with an anomaly score
        """
        start_time = time.time()
        chunks = list(summaries.values())

        num_batches = math.ceil(len(chunks) / self.batch_size)
        print(f"Num of batches to process: {num_batches}")

        batch_summaries = []

        for i in range(num_batches):
            print("--------------------------------------------")
            print(f"Processing batch {i + 1}...")
            batch_texts = chunks[i * self.batch_size:(i + 1) * self.batch_size]
            batch_summary = self.summarize_batch(batch_texts)
            batch_summaries.append(batch_summary)

        # recursively merge summaries which are greater than batch size
        while len(batch_summaries) > self.batch_size:
            temp = []
            for i in range(0, len(batch_summaries), self.batch_size):
                group = batch_summaries[i: i + self.batch_size]
                temp.append(self.summarize_batch(group))
            batch_summaries = temp

        print("--------------------------------------------")
        print(f"Processing final batch of size {len(batch_summaries)}")
        # if multiple summaries are present, merge them, else use the single summary
        if len(batch_summaries) > 1:
            final_summary = self.summarize_batch(batch_summaries)
        else:
            final_summary = batch_summaries[0]

        # extract anomaly score from final summary using a regex pattern
        final_anomaly_score = self.extract_anomaly_score(final_summary)
        print(
            f"Time taken for merge-summarize {len(summaries)} chunk summaries: {time.time() - start_time:.2f} seconds")

        return {"overall_summary": final_summary, "anomaly_score": final_anomaly_score}

    def summarize_batch(self, texts):
        """
        Summarize a batch of summaries using the chosen model
        """
        text = " ".join(texts)
        if not self.ov_llm:
            merged = self.chain.invoke({"video": "", "question": self.summary_prompt.format(text)})
        else:
            merged = self.chain.invoke({"question": text})
            '''for chunk in self.chain.stream({"question": text}):
                # print(chunk, end="", flush=True)
                merged += chunk'''
            # print("\n")
        return merged.strip()

    @staticmethod
    def extract_anomaly_score(summary):
        # matching based on multiple scenarios observed; goal is to match floating point or integer after Anomaly Score
        # Anomaly Score sometimes is encapsulated within ** and sometimes LLM omits
        match = re.search(r"\*?\*?Anomaly Score\*?\*?:?\s*(-?\d+(\.\d+)?)", summary, re.DOTALL)
        if match:
            return float(match.group(1)) if match.group(1) else 0.0
        return 0.0

    
