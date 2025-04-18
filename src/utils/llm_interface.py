import os
from typing import List, Optional
from llama_cpp import Llama
from transformers import AutoTokenizer

# Load the same tokenizer you use in StreamingQAGym
_TOKENIZER_NAME = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(
    _TOKENIZER_NAME,
    use_fast=True,
    add_eos_token=True
)

class LLMInterface:
    """
    Wrapper around llama-cpp-python for fast, quantized Llama-2-chat inference.
    Decodes input_ids to text, calls the model, and returns the generated string.
    """
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        temperature: float = 0.0
    ):
        """
        Args:
          model_path: Path to your .gguf model file.
          n_ctx:     Max context size (tokens).
          n_threads: CPU threads (defaults to all cores).
          temperature: Sampling temperature (0 = greedy).
        """
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads or os.cpu_count(),
            seed=42
        )
        self.temperature = temperature

    def generate(
        self,
        input_ids: List[int],
        max_tokens: int = 256,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Args:
          input_ids: List of token IDs (your context + question).
          max_tokens: How many tokens to generate.
          stop: Optional stop sequences.
        
        Returns:
          The generated text.
        """
        # Decode IDs → string prompt
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)

        # Call the quantized model
        output = self.model(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=self.temperature,
            stop=stop
        )

        # Extract and return
        text = output["choices"][0]["text"]
        return text.strip()

