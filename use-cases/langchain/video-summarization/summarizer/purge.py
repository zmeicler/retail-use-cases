from langchain.schema import BaseOutputParser
from optimum.intel.openvino import OVModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import openvino.properties.hint as hints
import time

class ToxicBERTParser(object):
    def __init__(self, model_id: str = "unitary/toxic-bert"):

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # [optional] OpenVINO configuration
        ov_config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
                     "CACHE_DIR": "cache\ov_toxic-bert_cache"}


        # Load the OpenVINO-optimized model
        self.ov_model = OVModelForSequenceClassification.from_pretrained(model_id,
                                                                    ov_config=ov_config,
                                                                    export=True)

    def parse(self, text: str) -> str:
        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Perform inference
        result = self.ov_model(input_ids=input_ids, attention_mask=attention_mask)
        toxic_scores = result.logits[0]

        # Mask toxic content based on scores
        cleaned_text = self.mask_toxic_content(text, toxic_scores, inputs["input_ids"])
        return cleaned_text

    def mask_toxic_content(self, text: str, scores: np.ndarray, input_ids: np.ndarray) -> str:
        threshold = 0.5 
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0]) 
        cleaned_tokens = []

        for token, score in zip(tokens, scores):
            if token in ["[CLS]", "[SEP]", "[PAD]"]: 
                continue
            if score > threshold:
                cleaned_tokens.append("[REDACTED]")
            else:
                cleaned_tokens.append(token)

        # Reconstruct the cleaned text
        cleaned_text = self.tokenizer.convert_tokens_to_string(cleaned_tokens)
        return cleaned_text

toxic_parser = ToxicBERTParser()

test_text = [
    "I hate you. You are stupid and worthless. ",
    "You are the worst person I have ever met. ",
]

for text in test_text:
    t= time.time()
    cleaned_text = toxic_parser.parse(text)
    print(time.time()-t)
    print(text)
    print(cleaned_text + '\n')

# Maybe just return [redacted] if the whole text is toxic. the token for token masking is not working.