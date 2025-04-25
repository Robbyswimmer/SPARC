# /Users/robbymoseley/CascadeProjects/SPARC/src/data/narrativeqa.py
"""NarrativeQA dataset loader."""

import logging
from typing import List, Tuple, Iterator

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_document(
    token_ids: List[int], chunk_size: int = 256
) -> List[List[int]]:
    """Chunks a list of token IDs into fixed-size chunks."""
    if not token_ids:
        return []
    return [
        token_ids[i : i + chunk_size]
        for i in range(0, len(token_ids), chunk_size)
    ]


def load_narrativeqa(
    tokenizer: PreTrainedTokenizerFast,
    split: str = "train",
    chunk_size: int = 256,
    max_samples: int | None = None,
) -> Iterator[Tuple[List[List[int]], List[int], str]]:
    """
    Loads the NarrativeQA dataset, tokenizes, chunks, and yields samples.

    Args:
        tokenizer: The tokenizer to use.
        split: The dataset split to load ('train', 'validation', 'test').
        chunk_size: The number of tokens per document chunk.
        max_samples: Optional maximum number of samples to yield.

    Yields:
        Tuples of (doc_chunks, question_ids, gold_answer).
        - doc_chunks: List of token ID lists, each representing a chunk.
        - question_ids: List of token IDs for the question.
        - gold_answer: The ground truth answer string.
    """
    logger.info(f"Loading NarrativeQA dataset, split: {split}")
    try:
        # Use streaming=True for large datasets if memory becomes an issue
        dataset: Dataset = load_dataset("narrativeqa", split=split, streaming=False)
    except Exception as e:
        logger.error(f"Failed to load NarrativeQA dataset: {e}")
        return

    count = 0
    for example in dataset:
        if max_samples is not None and count >= max_samples:
            logger.info(f"Reached max_samples limit ({max_samples}). Stopping.")
            break

        try:
            # Extract document, question, and answer
            document_text = example["document"]["text"]
            question_text = example["question"]["text"]
            # NarrativeQA often has multiple answers, take the first one for simplicity
            gold_answer = example["answers"][0]["text"]

            if not document_text or not question_text or not gold_answer:
                logger.warning(
                    f"Skipping sample due to missing data: {example.get('id', 'N/A')}"
                )
                continue

            # Tokenize document and question
            # Ensure truncation and padding are handled appropriately if needed,
            # but for chunking, we usually tokenize the whole document first.
            doc_ids = tokenizer(
                document_text, add_special_tokens=False
            ).input_ids
            question_ids = tokenizer(
                question_text, add_special_tokens=False
            ).input_ids

            # Chunk the document tokens
            doc_chunks = chunk_document(doc_ids, chunk_size)

            if not doc_chunks:
                logger.warning(
                    f"Skipping sample due to empty document chunks: {example.get('id', 'N/A')}"
                )
                continue

            yield doc_chunks, question_ids, gold_answer
            count += 1

        except Exception as e:
            logger.error(
                f"Error processing sample {example.get('id', 'N/A')}: {e}",
                exc_info=True,
            )

    logger.info(f"Finished processing {count} samples from NarrativeQA {split} split.")


# Example usage (optional, for testing)
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Make sure to use the same tokenizer as your main project
    tokenizer_name = "NousResearch/Meta-Llama-3.1-8B" # Or your chosen LLM tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Add pad token if it doesn't exist (common for models like GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    print(f"Loading data with tokenizer: {tokenizer_name}")
    # Load a few samples from validation for quick testing
    data_iterator = load_narrativeqa(tokenizer, split="validation", max_samples=5)

    for i, (chunks, q_ids, answer) in enumerate(data_iterator):
        print(f"\n--- Sample {i+1} ---")
        print(f"Number of chunks: {len(chunks)}")
        if chunks:
            print(f"First chunk length: {len(chunks[0])}")
            # print(f"First chunk tokens (decoded): {tokenizer.decode(chunks[0])}") # Optional: View decoded text
        print(f"Question length: {len(q_ids)}")
        # print(f"Question tokens (decoded): {tokenizer.decode(q_ids)}") # Optional: View decoded text
        print(f"Gold Answer: {answer}")

    print("\nExample usage finished.")