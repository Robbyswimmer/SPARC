import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Iterable, Dict, Any
from transformers import AutoTokenizer
from utils.llm_interface import LLMInterface
from utils.model_paths import llama_32_3b
from utils.metrics import compute_exact_match, compute_f1, compute_qa_score

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------- helpers ------------------------------------------------------------
_CHUNK_SIZE = 256                     # tokens per stream step
_MAX_WINDOW = 2048                    # budget B
_TOKENIZER_NAME = "NousResearch/Meta-Llama-3.1-8B"

# top‑level constants (tune later)
ALPHA      = 0.05   # token cost
BETA_KEEP  = 0.005  # keep penalty
BETA_COMP  = 0.002  # compress penalty
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
                 data_iter: Iterable[Dict[str, Any]] = None,
                 chunk_size: int = _CHUNK_SIZE,
                 token_reward_max: float = 0.1,      # Maximum per-episode token reward
                 token_reward_anneal_steps: int = 50000,  # Steps to anneal token rewards to 10%
                 seed: int | None = None):
        super().__init__()

        self.dataset_name = dataset_name
        self.split = split
        self.ds_iter = data_iter
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

        self._reset_episode_state()

    # ------------------------------------------------------------------------
    def _reset_episode_state(self):
        """Prepare a fresh doc‑question pair."""
        self.doc_chunks = None
        self.stored_tokens = []            # rolling context buffer
        self.question_ids = None
        self.gold_answer = None
        self.gold_answer_token_ids = set()  # Will hold tokenized gold answer for heuristic rewards
        self.chunk_idx = 0
        self.steps_left = None             # filled during reset()

    # ------------------------------------------------------------------------
    def _sample_episode(self):
        "Draw one (long_doc, question, answer) triplet from data loader."
        
        # Get the next tuple of (doc_chunks, question_ids, gold_answer)
        try:
            doc_chunks, question_ids, gold_answer = next(self.ds_iter)  # throws StopIteration when exhausted
            
            # Store the values directly from the tuple - they're already tokenized and chunked
            self.doc_chunks = doc_chunks
            self.question_ids = question_ids
            self.gold_answer = gold_answer
            
            # Tokenize gold answer for heuristic rewards
            gold_answer_tokens = tokenizer(self.gold_answer, add_special_tokens=False).input_ids
            self.gold_answer_token_ids = set(gold_answer_tokens)
            
            self.chunk_idx = 0
            
            # Set the number of steps for this episode
            self.steps_left = len(self.doc_chunks) + 1   # +1 for final Q&A step
            
        except Exception as e:
            print(f"Error loading next episode: {e}")
            raise

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
            model_answer  = self._query_llm(prompt_ids)
            em      = compute_exact_match(model_answer, self.gold_answer)
            f1      = compute_f1(model_answer, self.gold_answer)
            qa_score      = compute_qa_score(model_answer, self.gold_answer)
            token_penalty = ALPHA * (len(prompt_ids) / self.max_window)
            reward        = qa_score - token_penalty

            # Return an “empty” obs and terminate
            obs = np.full(self.chunk_size, tokenizer.pad_token_id, dtype=np.int32)
            info = {
                "step_idx":     self.chunk_idx,
                "exact_match":  em,
                "f1":           f1,
                "qa_score":     qa_score,
                "tokens_used":  len(prompt_ids),
                "model_answer": model_answer,
                "gold_answer":  self.gold_answer,
            }
            return obs, reward, True, False, info

        # ------------------------
        # 2) Still in streaming phase: apply action cost + extend tokens
        # ------------------------
        if action == 1:  # KEEP
            current_chunk = self.doc_chunks[self.chunk_idx]
            self.stored_tokens.extend(current_chunk)
            
            # Apply base cost for keeping
            reward = - (GAMMA_STEP + BETA_KEEP)
            
            # Add dynamic heuristic reward if gold answer tokens present in this chunk
            if self.gold_answer_token_ids:
                # Check for overlap between current chunk and gold answer tokens
                overlap = set(current_chunk) & self.gold_answer_token_ids
                if overlap:
                    # Dynamic scaling based on answer length and training progress
                    token_hit = len(overlap)  # Count number of matching tokens
                    
                    # Scale per-token reward (max total = token_reward_max)
                    if len(self.gold_answer_token_ids) > 0:  # Avoid division by zero
                        per_token = self.token_reward_max / len(self.gold_answer_token_ids)
                    else:
                        per_token = 0.01  # Fallback if no gold tokens
                    
                    # Anneal rewards over time (start strong, fade to 10%)
                    anneal = max(0.1, 1.0 - self.global_step / self.token_reward_anneal_steps) 
                    
                    # Apply the scaled and annealed reward
                    token_reward = token_hit * per_token * anneal
                    reward += token_reward
        else:            # DROP (or future COMPRESS)
            reward = 0.0

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
            return obs, reward, False, False, info

        # 4b) Exactly at question
        if self.chunk_idx == len(self.doc_chunks):
            obs  = self._pad(self.question_ids, self.chunk_size)
            info = {"step_idx": self.chunk_idx, "question": True}
            return obs, reward, False, False, info

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
        
        # Extract the question from the prompt (assuming it's at the end)
        question_text = tokenizer.decode(self.question_ids, skip_special_tokens=True)
        
        # Create a system prompt
        system_prompt = "You are a helpful, accurate, and concise assistant. Answer questions based on the provided document."
        
        # Format the prompt for our improved LLMInterface
        context_text = prompt_text.replace(question_text, "")
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

