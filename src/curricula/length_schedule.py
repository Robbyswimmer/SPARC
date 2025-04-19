# src/curricula/length_schedule.py

from typing import Dict, Any, Iterable, Iterator
import numpy as np

class LengthScheduleWrapper(Iterator[Dict[str, Any]]):
    """
    A curriculum wrapper that linearly increases the maximum number of document
    chunks exposed per episode over a specified number of training steps.

    Args:
        base_iterator (Iterable[Dict[str, Any]]): The underlying data iterator
            providing full episodes (e.g., document chunks, question, answer).
        initial_max_chunks (int): The maximum number of chunks allowed at the start.
        final_max_chunks (int): The maximum number of chunks allowed at the end.
        total_schedule_steps (int): The total number of training steps over which
            to increase the max chunks from initial to final.
        min_chunks (int): A minimum number of chunks to always include. Defaults to 1.
    """
    def __init__(self,
                 base_iterator: Iterable[Dict[str, Any]],
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

    def __iter__(self) -> Iterator[Dict[str, Any]]:
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

    def __next__(self) -> Dict[str, Any]:
        """
        Fetches the next episode and truncates its document chunks according
        to the current curriculum schedule.
        """
        # Get the next full episode from the base iterator
        try:
            episode_data = next(self.base_iterator)
        except StopIteration:
            # Re-raise StopIteration if the base iterator is exhausted
            raise StopIteration

        # Calculate the max chunks allowed for this step
        current_max_chunks = self._calculate_current_max_chunks()

        # Ensure 'document_chunks' exists and is a list
        if 'document_chunks' in episode_data and isinstance(episode_data['document_chunks'], list):
            # Truncate the document chunks
            original_num_chunks = len(episode_data['document_chunks'])
            # We need at least min_chunks, up to current_max_chunks, but no more than available
            effective_max_chunks = min(max(self.min_chunks, current_max_chunks), original_num_chunks)
            episode_data['document_chunks'] = episode_data['document_chunks'][:effective_max_chunks]
            # Optionally store metadata about truncation
            episode_data['original_num_chunks'] = original_num_chunks
            episode_data['current_max_chunks_limit'] = current_max_chunks # The calculated limit
            episode_data['effective_num_chunks'] = effective_max_chunks # The actual number used
        else:
             # Handle cases where 'document_chunks' might be missing or not a list
             print(f"Warning: 'document_chunks' key missing or not a list in episode data. Skipping truncation.")
             episode_data['original_num_chunks'] = 0
             episode_data['current_max_chunks_limit'] = current_max_chunks
             episode_data['effective_num_chunks'] = 0


        # Increment the step counter for the next call
        # This assumes one call to __next__ corresponds to one training step advancement
        # in the context of the curriculum schedule.
        self._current_step += 1

        return episode_data

    def step(self, num_steps: int = 1):
        """Manually advance the curriculum step counter if needed outside of __next__ calls."""
        self._current_step += num_steps

    def get_current_max_chunks(self) -> int:
        """Returns the max chunks allowed based on the current internal step count."""
        return self._calculate_current_max_chunks()
