import os
from typing import List, Optional, Dict, Any, Union
from llama_cpp import Llama
from transformers import AutoTokenizer
import json
import re
import sys

# Import model paths - use relative import to avoid path issues
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_paths import llama_31_8b

# Load the tokenizer
_TOKENIZER_NAME = "NousResearch/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(
    _TOKENIZER_NAME,
    use_fast=True,
    add_bos_token=True,
    add_eos_token=True
)

class LLMInterface:
    """
    Enhanced wrapper around llama-cpp-python for Llama-3 inference with proper prompt formatting.
    Supports system prompts, chat history, and various generation parameters.
    """
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
        temperature: float = 0.0,
        verbose: bool = False
    ):
        """
        Initialize the LLM interface.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size
            n_threads: Number of threads to use
            temperature: Temperature for sampling (0.0 = deterministic)
            verbose: Whether to print verbose output
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature
        self.verbose = verbose
        
        # Initialize the model
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            seed=42,
            verbose=False,
        )
        
        # Use tokenizer's special tokens
        self.bos_token = tokenizer.bos_token if hasattr(tokenizer, 'bos_token') else '<s>'
        self.eos_token = tokenizer.eos_token if hasattr(tokenizer, 'eos_token') else '</s>'
        
        if self.verbose:
            print(f"Using BOS token: {self.bos_token}")
            print(f"Using EOS token: {self.eos_token}")
        
        # Default system prompt
        self.default_system_prompt = "You are a helpful, accurate, and concise assistant."
    
    def format_prompt(self, 
                     user_message: str, 
                     system_prompt: Optional[str] = None,
                     chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Format a prompt for the model using chat template.
        
        Args:
            user_message: The user's message
            system_prompt: Optional system prompt (uses default if None)
            chat_history: Optional list of previous messages in the format 
                         [{"role": "user"|"assistant", "content": "message"}]
                         
        Returns:
            Formatted prompt string
        """
        if system_prompt is None:
            system_prompt = self.default_system_prompt
            
        # For Llama models, use a simple template
        prompt = ""
        
        # Add system prompt if provided
        if system_prompt:
            prompt += f"<s>\n{system_prompt}\n\n"
        else:
            prompt += "<s>\n"
        
        # Add chat history if provided
        if chat_history:
            for msg in chat_history:
                role = msg["role"]
                content = msg["content"]
                
                if role == "user":
                    prompt += f"Human: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
        
        # Add current user message and assistant prefix
        prompt += f"Human: {user_message}\nAssistant:"
        
        if self.verbose:
            print(f"Formatted prompt:\n{prompt}")
            
        return prompt
    
    def generate_text(self, 
                     user_message: str, 
                     system_prompt: Optional[str] = None,
                     chat_history: Optional[List[Dict[str, str]]] = None,
                     max_tokens: int = 256,
                     temperature: Optional[float] = None,
                     stop: Optional[List[str]] = None) -> str:
        """
        Generate text response to a user message.
        
        Args:
            user_message: The user's message
            system_prompt: Optional system prompt
            chat_history: Optional chat history
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling (overrides instance default if provided)
            stop: List of strings to stop generation on
            
        Returns:
            Generated text response
        """
        # Format the prompt
        prompt = self.format_prompt(
            user_message=user_message,
            system_prompt=system_prompt,
            chat_history=chat_history
        )
        
        # Set temperature
        temp = temperature if temperature is not None else self.temperature
        
        # Set default stop sequences if none provided
        if stop is None:
            stop = ["\n\n", "Human:", "<s>", "</s>"]
        
        # Generate response
        try:
            if self.verbose:
                print(f"Generating with max_tokens={max_tokens}, temperature={temp}")
                
            output = self.model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                stop=stop
            )
            
            if self.verbose:
                print(f"Raw output: {output}")
            
            # Extract and clean up the generated text
            if isinstance(output, dict) and "choices" in output and len(output["choices"]) > 0:
                response = output["choices"][0]["text"].strip()
                
                # Clean up any formatting
                response = re.sub(r'^\s*Assistant:\s*', '', response)
                response = re.sub(r'</s>\s*$', '', response)

                
                if self.verbose:
                    print(f"Cleaned response: {response}")
                
                return response
            else:
                # Handle the case where output doesn't have the expected structure
                if self.verbose:
                    print("Unexpected output format, returning raw output")
                return str(output)
            
        except Exception as e:
            if self.verbose:
                print(f"Error during generation: {e}")
            return f"Error generating response: {e}"
    
    def generate(self, 
                input_ids: List[int], 
                max_tokens: int = 256, 
                stop: Optional[List[str]] = None) -> str:
        """
        Legacy method for backward compatibility.
        Generate text from input token IDs.
        
        Args:
            input_ids: List of token IDs
            max_tokens: Maximum number of tokens to generate
            stop: List of strings to stop generation on
            
        Returns:
            Generated text
        """
        # Decode IDs → string prompt
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        # Call the model
        output = self.model(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=self.temperature,
            stop=stop
        )
        
        # Extract and return
        text = output["choices"][0]["text"]
        return text.strip()


if __name__ == "__main__":
    print("Now interacting with the LLM...")

    try:
        # Create LLM interface with verbose mode
        llm = LLMInterface(model_path=llama_31_8b, verbose=False, n_ctx=500)

        # Initialize chat history
        chat_history = []

        # Define a system prompt
        system_prompt = "You are a helpful, accurate, and concise assistant specialized in question answering. Provide clear and direct answers."

        print("\nType 'exit' to quit, 'clear' to clear chat history\n")
        
        while True:
            user_input = input("> ")
            
            if user_input.lower() == "exit":
                break
                
            if user_input.lower() == "clear":
                chat_history = []
                print("Chat history cleared")
                continue

            # Generate response
            response = llm.generate_text(
                user_message=user_input,
                system_prompt=system_prompt,
                chat_history=chat_history,
                max_tokens=256,
                stop=["\n\n", "Human:", "<s>"]
            )
            
            print(f"\n{response}\n")
            
            # Update chat history
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})
            
            # Keep chat history manageable
            if len(chat_history) > 10:
                # Remove oldest user-assistant pair
                chat_history = chat_history[2:]
    
    except Exception as e:
        print(f"Error: {e}")
