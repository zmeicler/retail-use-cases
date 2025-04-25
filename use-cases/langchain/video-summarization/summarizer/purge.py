from langchain.schema import BaseOutputParser
from optimum.intel.openvino import OVModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import openvino.properties.hint as hints
import time
from better_profanity import profanity
from pydantic import PrivateAttr

class ProfanityFilter(BaseOutputParser):
    def __init__(self, custom_words: list[str] = None):
        """
        Initialize the ProfanityFilter, add optional custom words.
        """
        # Combine default words with custom words
        combined_words = set(map(str, profanity.CENSOR_WORDSET)).union(set(custom_words or []))
        profanity.load_censor_words(combined_words)

    def parse(self, text: str) -> str:
        """
        Censors profane words in the input text.
        """
        return profanity.censor(text)

    def list_censor_words(self) -> list[str]:
        """
        Returns the list of words currently on the censor list as human-readable strings.
        """
        return [str(word) for word in profanity.CENSOR_WORDSET]

class ToxicBERTFilter(BaseOutputParser):
    _tokenizer: AutoTokenizer = PrivateAttr()
    _ov_model: OVModelForSequenceClassification = PrivateAttr()
    _redact_thresh: float = PrivateAttr()

    def __init__(self, model_id: str = "unitary/toxic-bert", redact_thresh: float = 0.5):
        """
        Initialize the ToxicBERTFilter and langchain components.
        """
        super().__init__()
        self._redact_thresh = redact_thresh
        # Load the tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        # OpenVINO configuration
        ov_config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
                     "CACHE_DIR": "cache\ov_toxic-bert_cache"}

        # Load the OpenVINO-optimized model
        self._ov_model = OVModelForSequenceClassification.from_pretrained(
            model_id,
            ov_config=ov_config,
            export=True
        )

    def parse(self, text: str) -> str:
        """
        Parse the input text and filter toxic content.
        """
        # Tokenize the input text
        inputs = self._tokenizer(
            text,
            return_tensors="np",
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Run inference
        result = self._ov_model(input_ids=input_ids, attention_mask=attention_mask)
        toxic_scores = result.logits[0]

        # Mask toxic content based on scores
        cleaned_text = self.mask_toxic_content(text, toxic_scores, inputs["input_ids"])
        return cleaned_text

    def mask_toxic_content(self, text: str, scores: np.ndarray, input_ids: np.ndarray) -> str:
        """
        Mask toxic content in the input text based on scores. Replace with [REDACTED] if score > threshold.
        """
        tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0])
        cleaned_tokens = []

        for token, score in zip(tokens, scores):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            if score > self._redact_thresh:
                cleaned_tokens.append("[REDACTED]")
            else:
                cleaned_tokens.append(token)

        # Reconstruct the cleaned text
        cleaned_text = self._tokenizer.convert_tokens_to_string(cleaned_tokens)
        return cleaned_text

# Example usage
if __name__ == "__main__":

    test_texts = [
        "This is a clean sentence.",
        "You are stupid and dumb.",
        "I hate you you are lame."
        ]
    
    profanity_filter = ProfanityFilter(["dumb"])
    print("--Profanity Filter--")
    for text in test_texts:
        print(f"Original: {text}")
        print(f"Filtered: {profanity_filter.parse(text)}\n")

    profanityFilter_tb = ToxicBERTFilter()
    print("--ToxicBERT--")
    for text in test_texts:
        cleaned_text = profanityFilter_tb.parse(text)
        print(f"Original: {text}")
        print(f"Filtered: {cleaned_text}\n")

    # Maybe just return [redacted] if the whole text is toxic. the token for token masking is not working.