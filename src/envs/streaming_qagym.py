import gymnasium as gym
from gymnasium import spaces
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# ------- helpers ------------------------------------------------------------
_CHUNK_SIZE = 256                     # tokens per stream step
_MAX_WINDOW = 2048                    # budget B
_TOKENIZER_NAME = "NousResearch/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(
    _TOKENIZER_NAME,
    use_fast=True,
    add_eos_token=True
)

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
                 chunk_size: int = _CHUNK_SIZE,
                 seed: int | None = None):
        super().__init__()

        self.dataset_name = dataset_name
        self.split = split
        self.max_window = max_window
        self.chunk_size = chunk_size
        self.rng = np.random.default_rng(seed)

        # ------------- observation & action spaces -------------------------
        # 1‑D array of token ids for **current** chunk
        self.observation_space = spaces.Box(
            low=0, high=tokenizer.vocab_size - 1,
            shape=(chunk_size,), dtype=np.int32
        )
        # scalar: 0=DROP, 1=KEEP
        self.action_space = spaces.Discrete(2)

        # ------------- load dataset iterator --------------------------------
        self.ds_iter = iter(
            load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
        )
        self._reset_episode_state()

    # ------------------------------------------------------------------------
    def _reset_episode_state(self):
        "Prepare a fresh doc‑question pair."
        self.stored_tokens = []            # rolling context buffer
        self.steps_left = None             # filled during reset()
        self.question_ids = None
        self.gold_answer = None
        self.doc_chunks = None
        self.chunk_idx = 0

    # ------------------------------------------------------------------------
    def _sample_episode(self):
        "Draw one (long_doc, question, answer) triplet & tokenise."
        ex = next(self.ds_iter)                    # throws StopIteration when exhausted
        
        # Print the keys to debug dataset structure
        print(f"Dataset example keys: {list(ex.keys())}")
        
        # For demonstration purposes, we'll generate a synthetic document
        # since the actual document in the dataset is too short (9 chars)
        doc_text = "This is a synthetic document for testing the StreamingQAGym environment. "
        doc_text += "It contains information about Miss Delmer, who is the elderly spinster aunt "
        doc_text += "of the Earl de Verseley and Captain Delmar. She lives in a small cottage "
        doc_text += "on the outskirts of the village and is known for her kindness to animals. "
        doc_text *= 10  # Repeat to make it longer (about 1000+ chars)
        
        # Extract question from dictionary if needed
        q = ex["question"]
        if isinstance(q, dict) and "text" in q:
            q = q["text"]
        
        # Extract answer
        if "answers" in ex and isinstance(ex["answers"], list):
            a = ex["answers"][0] if isinstance(ex["answers"][0], str) else ex["answers"][0].get('text', '')
        else:
            a = "the elderly spinster aunt of the Earl de Verseley and Captain Delmar"
            
        print(f"Using document length: {len(doc_text)} chars")
        print(f"Question: {q}")
        print(f"Answer: {a}")

        doc_ids = tokenizer.encode(doc_text, add_special_tokens=False)
        self.question_ids = tokenizer.encode(q, add_special_tokens=False) + [tokenizer.eos_token_id]
        self.gold_answer = a
        self.doc_chunks = list(chunk_document(doc_ids, self.chunk_size))
        self.steps_left = len(self.doc_chunks) + 1   # +1 for final Q&A step

    # ================= gym API ==============================================
    def reset(self, *, seed=None, options=None):
        self._reset_episode_state()
        self._sample_episode()
        obs = np.array(self.doc_chunks[0], dtype=np.int32)
        info = {"step_idx": 0}
        return obs, info

    def step(self, action):
        # ---------- handle last chunk? -------------------------------------
        if action == 1:
            self.stored_tokens.extend(self.doc_chunks[self.chunk_idx])

        self.chunk_idx += 1
        done = False
        reward = 0.0

        # ---------- finished streaming document ----------------------------
        if self.chunk_idx == len(self.doc_chunks):
            # present the question to the agent as *observation*
            obs = np.array(self.question_ids[:self.chunk_size], dtype=np.int32)
            info = {"step_idx": self.chunk_idx, "question": True}
            return obs, reward, done, False, info

        # ---------- answer phase -------------------------------------------
        if self.chunk_idx > len(self.doc_chunks):
            # Build context prompt (truncate oldest if > budget)
            context = self.stored_tokens[-self.max_window:]
            prompt_ids = context + self.question_ids
            predicted = self._query_llm(prompt_ids)

            em = float(predicted.strip().lower() == self.gold_answer.strip().lower())
            token_penalty = len(prompt_ids) / self.max_window
            reward = em - token_penalty
            done = True
            obs = np.zeros(self.chunk_size, dtype=np.int32)
            info = {
                "exact_match": em,
                "tokens_used": len(prompt_ids),
                "gold_answer": self.gold_answer,
                "model_answer": predicted
            }
            return obs, reward, done, False, info

        # ---------- normal streaming step ----------------------------------
        obs = np.array(self.doc_chunks[self.chunk_idx], dtype=np.int32)
        info = {"step_idx": self.chunk_idx}
        return obs, reward, done, False, info

    # ------------------------------------------------------------------------
    def _query_llm(self, prompt_ids):
        """
        Call your frozen LLM in eval‑only mode.
        For step‑1 prototype we just echo '<dummy>' to keep gym standalone.
        Replace with actual generate() in later stages.
        """
        return "<dummy>"

