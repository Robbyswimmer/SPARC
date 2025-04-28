import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Iterable, Dict, Any, Callable
from transformers import AutoTokenizer
from utils.llm_interface import LLMInterface
from utils.model_paths import llama_32_3b
from utils.metrics import compute_exact_match, compute_f1, compute_qa_score, max_over_refs, compute_qa_score_multi

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------- helpers ------------------------------------------------------------
_CHUNK_SIZE = 256                     # tokens per stream step
_MAX_WINDOW = 2048                    # budget B
_TOKENIZER_NAME = "NousResearch/Meta-Llama-3.1-8B"

# top‑level constants (tune later)
ALPHA      = 0.05   # token cost
BETA_KEEP  = -0.008  # keep penalty (negative because it's a cost)
BETA_COMP  = -0.002  # compress penalty (negative because it's a cost)
GAMMA_STEP = 0.001  # per‑step thrift penalty

tokenizer = AutoTokenizer.from_pretrained(
    _TOKENIZER_NAME,
    use_fast=True,
    add_eos_token=True
)

# ————— Fix for pad_token_id being None —————
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

def chunk_document(doc_tokens, chunk_size=_CHUNK_SIZE):
    """Yield successive token chunks from a list of token ids."""
    for i in range(0, len(doc_tokens), chunk_size):
        yield doc_tokens[i:i + chunk_size]

# ------- main class ---------------------------------------------------------
class StreamingQAGym(gym.Env):
    """
    At each step you receive the next chunk (256 tokens) of a long document.
    The agent decides {0: DROP, 1: KEEP}. 
    After streaming all chunks + the final question, the gym computes the
    QA reward offline and returns it as the final step reward.
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 dataset_name: str = "deepmind/narrativeqa",
                 split: str = "train",
                 max_window: int = _MAX_WINDOW,
                 alpha: float = ALPHA,
                 beta_keep: float = BETA_KEEP,
                 beta_compress: float = BETA_COMP,
                 gamma_step: float = GAMMA_STEP, # Add gamma_step parameter
                 data_loader_fn: Callable[[], Iterable[Dict[str, Any]]] = None,
                 chunk_size: int = _CHUNK_SIZE,
                 token_reward_max: float = 0.1,      # Maximum per-episode token reward
                 token_reward_anneal_steps: int = 50000,  # Steps to anneal token rewards to 10%
                 seed: int | None = None):
        super().__init__()

        self.dataset_name = dataset_name
        self.split = split
        self.data_loader_fn = data_loader_fn
        # Remove the iterator creation from __init__
        # Each parallel environment needs its own iterator
        self.ds_iter = None
        self.max_window = max_window
        self.chunk_size = chunk_size
        
        # Token reward parameters
        self.token_reward_max = token_reward_max
        self.token_reward_anneal_steps = token_reward_anneal_steps
        self.global_step = 0  # Will be updated externally
        
        self.rng = np.random.default_rng(seed)

        # Load LLM interface
        # Use a reasonable buffer for question tokens (typically <50 tokens)
        self.llm = LLMInterface(
            model_path=llama_32_3b,
            n_ctx=self.max_window + 50,  # Buffer for question tokens
            n_threads=4,
            temperature=0.0
        )
        self.question_ids = []  # Initialize empty, will be set in reset()

        # ------------- observation & action spaces -------------------------
        # 1‑D array of token ids for **current** chunk
        self.observation_space = spaces.Box(
            low=0, high=tokenizer.vocab_size - 1,
            shape=(chunk_size,), dtype=np.int32
        )
        # scalar: 0=DROP, 1=KEEP
        self.action_space = spaces.Discrete(2)

        self.gamma_step = gamma_step  # Store gamma_step
        self.alpha = alpha
        self.beta_keep = beta_keep  # Store beta_keep correctly
        self.beta_compress = beta_compress

        self._reset_episode_state()

    # ------------------------------------------------------------------------
    def _reset_episode_state(self):
        """Reset all episode-specific state variables."""
        self.chunk_idx = 0          # index into the current doc chunks
        self.stored_tokens = []     # token ids we're keeping so far
        self.keep_count = 0         # number of chunks we've kept
        self.drop_count = 0         # number of chunks we've dropped
        self.compress_count = 0     # number of chunks we've compressed (future)
        self.gold_answer = None
        self.gold_answers = None    # Multi-reference version
        self.gold_answer_token_ids = set()  # for use in gold token reward heuristic
        self.steps_left = None             # filled during reset()

    # ------------------------------------------------------------------------
    def _sample_episode(self):
        """Draw one (long_doc, question, answer) triplet from data loader.
        If the iterator is exhausted, recreate it by calling the loader function."""
        
        # Always create a fresh iterator for each episode - critical for parallel environments
        if self.data_loader_fn is None:
            raise RuntimeError("Dataset loader function was not provided during initialization.")
            
        self.ds_iter = iter(self.data_loader_fn())
            
        # Get the next example (dictionary format)
        try:
            example = next(self.ds_iter)  # throws StopIteration when exhausted
        except StopIteration:
            # If first attempt fails, try one more time with a fresh iterator
            print("Data iterator exhausted, rebuilding...")
            self.ds_iter = iter(self.data_loader_fn())
            try:
                example = next(self.ds_iter)
            except StopIteration:
                raise RuntimeError("Dataset empty even after iterator reset!")
            
        # Extract and store values from the dictionary
        self.doc_chunks = example["doc_chunks"]
        self.question_ids = example["question_ids"]
        self.gold_answers = example["answers"]
        self.example_meta = example.get("meta", {})  # Store metadata for logging
        
        # Use the first answer as the primary answer for backward compatibility
        self.gold_answer = self.gold_answers[0] if self.gold_answers else ""
        
        # Tokenize primary gold answer for heuristic rewards
        gold_answer_tokens = tokenizer(self.gold_answer, add_special_tokens=False).input_ids
        self.gold_answer_token_ids = set(gold_answer_tokens)
        
        # Reset episode state counters
        self.chunk_idx = 0
        
        # Set the number of steps for this episode
        self.steps_left = len(self.doc_chunks) + 1   # +1 for final Q&A step

    # ================= gym API ==============================================
    def reset(self, *, seed=None, options=None):
        self._reset_episode_state()
        self._sample_episode()
        obs = self._pad(self.doc_chunks[0], self.chunk_size)
        info = {"step_idx": 0}
        return obs, info

    # ================= Always return fixed length vector for stable baselines ==============================================
    def _pad(self, obs_ids, target_len):
        """Pad observation IDs to target length.
        
        Args:
            obs_ids: List of token IDs or None
            target_len: Target length to pad to
            
        Returns:
            Numpy array of padded token IDs
        """
        # Handle None case
        if obs_ids is None:
            return np.full(target_len, tokenizer.pad_token_id, dtype=np.int32)
        
        # Handle empty list or non-list case
        try:
            if len(obs_ids) == 0:
                return np.full(target_len, tokenizer.pad_token_id, dtype=np.int32)
        except (TypeError, AttributeError):
            # If obs_ids is not a list or doesn't have a length
            print(f"Warning: Invalid obs_ids type: {type(obs_ids)}")
            return np.full(target_len, tokenizer.pad_token_id, dtype=np.int32)
            
        # Check if all elements are valid integers
        try:
            # Convert any non-integer elements to pad token
            obs_ids = [int(x) if isinstance(x, (int, float, str)) and str(x).strip() 
                      else tokenizer.pad_token_id for x in obs_ids]
        except (ValueError, TypeError):
            # If conversion fails, create a safe array
            print(f"Warning: Invalid elements in obs_ids")
            return np.full(target_len, tokenizer.pad_token_id, dtype=np.int32)
            
        # Normal padding case
        if len(obs_ids) < target_len:
            obs_ids = obs_ids + [tokenizer.pad_token_id] * (target_len - len(obs_ids))
        else:
            obs_ids = obs_ids[:target_len]
            
        # Final safety check before creating numpy array
        try:
            return np.array(obs_ids, dtype=np.int32)
        except (ValueError, TypeError) as e:
            print(f"Error creating numpy array: {e}")
            return np.full(target_len, tokenizer.pad_token_id, dtype=np.int32)

   # ------------------------------------------------------------------------
    def step(self, action):
        """
        Returns:
            obs       : np.ndarray[int32]  # shape=(chunk_size,)
            reward    : float
            terminated: bool
            truncated : bool (always False)
            info      : dict
        """
        # Increment global step count for annealing
        self.global_step += 1
        
        # ------------------------
        # 1) Did we already finish streaming?
        # ------------------------
        if self.chunk_idx >= len(self.doc_chunks):
            # -- question arrival (show question but don't terminate) --
            if self.chunk_idx == len(self.doc_chunks):
                obs  = self._pad(self.question_ids, self.chunk_size)
                info = {"step_idx": self.chunk_idx, "question": True}
                self.chunk_idx += 1
                return obs, 0.0, False, False, info

            # -- answer phase (final reward + terminate) --
            # Build prompt from kept tokens + question
            context    = self.stored_tokens[-self.max_window:]
            prompt_ids = context + self.question_ids

            # Query LLM and score
            model_answer = self._query_llm(prompt_ids)
            
            # Calculate standard metrics against primary answer
            em = compute_exact_match(model_answer, self.gold_answer)
            f1 = compute_f1(model_answer, self.gold_answer)
            
            # Calculate max-over-references metrics
            max_em = max_over_refs(compute_exact_match, model_answer, self.gold_answers)
            max_f1 = max_over_refs(compute_f1, model_answer, self.gold_answers)
            
            # Use multi-reference QA score for reward
            qa_score = compute_qa_score_multi(self.gold_answers, model_answer)
            
            token_penalty = self.alpha * (len(prompt_ids) / self.max_window)
            final_reward = qa_score - token_penalty

            # Return an “empty” obs and terminate
            obs = np.full(self.chunk_size, tokenizer.pad_token_id, dtype=np.int32)
            info = {
                "step_idx":     self.chunk_idx,
                "exact_match":  em,  # Single reference (backward compatibility)
                "f1":           f1,  # Single reference (backward compatibility)
                "max_em":       max_em,  # Max over all references
                "max_f1":       max_f1,  # Max over all references
                "qa_score":     qa_score,
                "tokens_used":  len(prompt_ids),
                "model_answer": model_answer,
                "gold_answer":  self.gold_answer,
                "gold_answers": self.gold_answers,
                "keep_count":   self.keep_count, # Log keep count
                "drop_count":   self.drop_count, # Log drop count
            }
            return obs, final_reward, True, False, info

        # ------------------------
        # 2) Still in streaming phase: process action, update state
        # ------------------------
        # Initialize step reward with the base per-step penalty
        step_reward = -self.gamma_step 
        token_heuristic_reward = 0.0 
        current_chunk = self.doc_chunks[self.chunk_idx]

        if action == 1:  # KEEP
            # Check window constraint BEFORE adding
            if len(self.stored_tokens) + len(current_chunk) <= self.max_window:
                self.stored_tokens.extend(current_chunk)
                self.keep_count += 1 
                
                # Calculate heuristic reward (if applicable)
                if self.gold_answer_token_ids:
                    overlap = set(current_chunk) & self.gold_answer_token_ids
                    if overlap:
                        token_hit = len(overlap)
                        if len(self.gold_answer_token_ids) > 0:
                            per_token = self.token_reward_max / len(self.gold_answer_token_ids)
                        else:
                            per_token = 0.01 # Fallback
                        anneal = max(0.1, 1.0 - self.global_step / self.token_reward_anneal_steps) 
                        token_heuristic_reward = token_hit * per_token * anneal
                
                # Apply the keep cost and heuristic bonus to the step reward
                # Since beta_keep is negative, adding it applies the cost.
                step_reward += self.beta_keep + token_heuristic_reward
                
            else:
                # Implicit DROP due to window full
                action = 0 
                self.drop_count += 1 
                # Apply beta_keep cost anyway to penalize KEEP attempts that overflow
                # This ensures the agent learns to respect window constraints
                step_reward += self.beta_keep
    
        if action == 0: # Explicit DROP
            self.drop_count += 1 
            # No additional cost/reward beyond gamma_step for explicit drop

        # ------------------------
        # 3) Advance to next chunk index
        # ------------------------
        self.chunk_idx += 1

        # ------------------------
        # 4) Now decide what to show next
        # ------------------------
        # 4a) Still more chunks?
        if self.chunk_idx < len(self.doc_chunks):
            obs  = self._pad(self.doc_chunks[self.chunk_idx], self.chunk_size)
            info = {"step_idx": self.chunk_idx}
            return obs, step_reward, False, False, info

        # 4b) Exactly at question
        if self.chunk_idx == len(self.doc_chunks):
            obs  = self._pad(self.question_ids, self.chunk_size)
            info = {"step_idx": self.chunk_idx, "question": True}
            return obs, step_reward, False, False, info

    # ------------------------------------------------------------------------
    def _query_llm(self, prompt_ids):
        """
        Call your frozen LLM in eval-only mode.
        
        Args:
            prompt_ids: List of token IDs for the prompt
            
        Returns:
            Generated text response
        """
        # Convert token IDs to text
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        
        # Extract the question from the prompt using slicing by length
        question_text = tokenizer.decode(self.question_ids, skip_special_tokens=True)
        
        # Create a system prompt
        system_prompt = "You are a helpful, accurate, and concise assistant. Answer questions based on the provided document."
        
        # Use slicing to safely separate context from question
        # First check if prompt_text is at least as long as question_text
        if len(prompt_text) >= len(question_text):
            # Extract just the context portion by slicing off the question length from the end
            context_text = prompt_text[:-len(question_text)].strip()
        else:
            # Safety fallback (should not happen in normal operation)
            context_text = prompt_text
            
        user_message = f"Document: {context_text}\n\nQuestion: {question_text}\n\nProvide a direct and concise answer."
        
        # Generate the answer using our improved interface
        try:
            return self.llm.generate_text(
                user_message=user_message,
                system_prompt=system_prompt,
                max_tokens=256,
                temperature=0.0,
                stop=["\n\n", "Human:", "Document:", "Question:"]
            )
        except Exception as e:
            print(f"Error in LLM generation: {e}")
            # Fallback to legacy method if needed
            return self.llm.generate(prompt_ids, max_tokens=256, stop=["\n"])

    def set_global_step(self, step: int):
        """Set the global training step for reward annealing."""
        self.global_step = step
