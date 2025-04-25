# src/curricula/length_schedule.py

from typing import Dict, Any, Iterable, Iterator, List, Tuple, Union
import numpy as np

class LengthScheduleWrapper(Iterator[Tuple[List[List[int]], List[int], str]]):
    """
    A curriculum wrapper that linearly increases the maximum number of document
    chunks exposed per episode over a specified number of training steps.

    Args:
        base_iterator (Iterable[Tuple[List[List[int]], List[int], str]]): The underlying data iterator
            providing full episodes as tuples of (doc_chunks, question_ids, gold_answer).
        initial_max_chunks (int): The maximum number of chunks allowed at the start.
        final_max_chunks (int): The maximum number of chunks allowed at the end.
        total_schedule_steps (int): The total number of training steps over which
            to increase the max chunks from initial to final.
        min_chunks (int): A minimum number of chunks to always include. Defaults to 1.
    """
    def __init__(self,
                 base_iterator: Iterable[Tuple[List[List[int]], List[int], str]],
                 initial_max_chunks: int,
                 final_max_chunks: int,
                 total_schedule_steps: int,
                 min_chunks: int = 1):
        self.base_iterator = iter(base_iterator)
        self.initial_max_chunks = initial_max_chunks
        self.final_max_chunks = final_max_chunks
        self.total_schedule_steps = total_schedule_steps
        self.min_chunks = max(1, min_chunks) # Ensure at least 1 chunk
        self._current_step = 0 # Tracks how many times __next__ has been called

    def __iter__(self) -> Iterator[Tuple[List[List[int]], List[int], str]]:
        return self

    def _calculate_current_max_chunks(self) -> int:
        """Calculates the max chunks allowed at the current step."""
        if self.total_schedule_steps <= 0:
            return self.final_max_chunks

        # Use _current_step which tracks iterator calls, assuming 1 call per training step
        progress = min(1.0, self._current_step / self.total_schedule_steps)
        current_max = int(np.round(
            self.initial_max_chunks + progress * (self.final_max_chunks - self.initial_max_chunks)
        ))
        # Ensure we don't go below the minimum or above the final max
        return max(self.min_chunks, min(current_max, self.final_max_chunks))

    def __next__(self) -> Tuple[List[List[int]], List[int], str]:
        """
        Fetches the next episode and truncates its document chunks according
        to the current curriculum schedule.
        """
        # Get the next full episode from the base iterator
        try:
            doc_chunks, question_ids, gold_answer = next(self.base_iterator)
        except StopIteration:
            # Re-raise StopIteration if the base iterator is exhausted
            raise StopIteration

        # Calculate the max chunks allowed for this step
        current_max_chunks = self._calculate_current_max_chunks()

        # Check if doc_chunks is a list and handle accordingly
        if isinstance(doc_chunks, list):
            # Truncate the document chunks
            original_num_chunks = len(doc_chunks)
            # We need at least min_chunks, up to current_max_chunks, but no more than available
            effective_max_chunks = min(max(self.min_chunks, current_max_chunks), original_num_chunks)
            doc_chunks = doc_chunks[:effective_max_chunks]
            # Log info (optional - you might want to remove these prints in production)
            if effective_max_chunks < original_num_chunks:
                print(f"Curriculum: Using {effective_max_chunks}/{original_num_chunks} chunks")
        else:
            # Handle cases where doc_chunks might not be a list
            print(f"Warning: doc_chunks is not a list. Skipping truncation.")

        # Increment the step counter for the next call
        self._current_step += 1
        
        # Return the possibly truncated data
        return doc_chunks, question_ids, gold_answer

    def step(self, num_steps: int = 1):
        """Manually advance the curriculum step counter if needed outside of __next__ calls."""
        self._current_step += num_steps

    def get_current_max_chunks(self) -> int:
        """Returns the max chunks allowed based on the current internal step count."""
        return self._calculate_current_max_chunks()
