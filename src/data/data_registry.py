"""Data registry for QA datasets with standardized adapters and mixed streaming.

This module provides a unified interface for loading and mixing multiple QA datasets,
with support for curriculum learning based on agent performance.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Iterator, Callable, Union
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerFast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset registry with HuggingFace dataset paths and splits
DATASETS = {
    "narrativeqa": {"hf": "deepmind/narrativeqa", "split": "train"},
    "triviaqa":    {"hf": "trivia_qa", "config": "rc", "split": "train"},
    "nq_long":     {"hf": "nq_open", "split": "train"},
    "hotpotqa":    {"hf": "hotpot_qa", "config": "fullwiki", "split": "train"},
}

# ================= Dataset Adapters =================

def chunk_document(token_ids: List[int], chunk_size: int = 256) -> List[List[int]]:
    """Chunks a list of token IDs into fixed-size chunks.
    
    Args:
        token_ids: List of token IDs to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks, where each chunk is a list of token IDs.
        If input is empty, returns a list with a single empty chunk [[]] as a sentinel.
    """
    if not token_ids:
        # Return a sentinel empty chunk instead of an empty list
        # This ensures downstream code always has at least one chunk to process
        return [[]]
        
    return [
        token_ids[i : i + chunk_size]
        for i in range(0, len(token_ids), chunk_size)
    ]

# Standard format for QA datasets: (doc_chunks, question_ids, answers)
# where:
# - doc_chunks: List[List[int]] - tokenized chunks of the document
# - question_ids: List[int] - tokenized question
# - answers: str or List[str] - answer string(s)

def narrativeqa_adapter(
    tokenizer: PreTrainedTokenizerFast, 
    split: str = "train", 
    streaming: bool = True,
    use_summaries: bool = True,  # Use summaries instead of full stories for easier training
) -> Iterator[Tuple[List[List[int]], List[int], List[str]]]:
    """Adapter for NarrativeQA dataset."""
    try:
        # Load the dataset - note: the dataset ID is "narrativeqa" not "deepmind/narrativeqa"
        dataset = load_dataset("narrativeqa", split=split, streaming=streaming)
        logger.info(f"Successfully loaded NarrativeQA dataset, split: {split}")
    except Exception as e:
        logger.error(f"Failed to load NarrativeQA dataset: {e}")
        return
    
    for item in dataset:
        try:
            # Extract document, question, and answer based on actual dataset structure
            if use_summaries:
                # Use summary instead of full document for easier training
                document = item["document"]["summary"]["text"]
            else:
                document = item["document"]["text"]
                
            question = item["question"]["text"]
            # Get all answers as a list
            answers = [a["text"] for a in item["answers"]]
            
            # Skip if any required field is missing
            if not document or not question or not answers:
                logger.warning(f"Skipping sample due to missing data: {item.get('id', 'N/A')}")
                continue
            
            # Tokenize
            doc_tokens = tokenizer(document, add_special_tokens=False).input_ids
            question_tokens = tokenizer(question, add_special_tokens=False).input_ids
            
            # Chunk document
            doc_chunks = chunk_document(doc_tokens)
            
            # Skip if document chunks are empty
            if not doc_chunks:
                logger.warning(f"Skipping sample due to empty document chunks: {item.get('id', 'N/A')}")
                continue
            
            yield doc_chunks, question_tokens, answers
            
        except Exception as e:
            logger.error(f"Error processing NarrativeQA sample: {e}")
            continue

def hotpotqa_adapter(
    tokenizer: PreTrainedTokenizerFast,
    split: str = "train",
    streaming: bool = True,
) -> Iterator[Tuple[List[List[int]], List[int], List[str]]]:
    """Adapter for HotpotQA dataset."""
    try:
        dataset = load_dataset(
            "hotpot_qa", 
            "fullwiki", 
            split=split, 
            streaming=streaming,
            trust_remote_code=True
        )
        logger.info(f"Successfully loaded HotpotQA dataset, split: {split}")
    except Exception as e:
        logger.error(f"Error loading HotpotQA: {e}")
        return
    
    for item in dataset:
        try:
            # Check if context exists and is not empty
            if not item["context"] or len(item["context"]) == 0:
                logger.warning("Skipping HotpotQA sample with empty context")
                continue
                
            # Check if at least one context item has sentences
            has_valid_sentences = False
            for _, sentences in item["context"]:
                if sentences:  # Check if sentences list is not empty
                    has_valid_sentences = True
                    break
                    
            if not has_valid_sentences:
                logger.warning("Skipping HotpotQA sample with no sentences in context")
                continue
                
            context = ""
            # Concatenate supporting facts
            for title, sentences in item["context"]:
                context += f"Title: {title}\n"
                for sent in sentences:
                    context += f"{sent}\n"
            
            question = item["question"]
            # HotpotQA has a single answer, but we'll put it in a list for consistency
            answers = [item["answer"]]
            
            # Skip if any required field is missing
            if not context or not question or not answers:
                logger.warning(f"Skipping HotpotQA sample due to missing data")
                continue
            
            # Tokenize
            doc_tokens = tokenizer(context, add_special_tokens=False).input_ids
            question_tokens = tokenizer(question, add_special_tokens=False).input_ids
            
            # Chunk document
            doc_chunks = chunk_document(doc_tokens)
            
            # Skip if document chunks are empty
            if not doc_chunks:
                logger.warning(f"Skipping HotpotQA sample due to empty document chunks")
                continue
            
            yield doc_chunks, question_tokens, answers
            
        except Exception as e:
            logger.error(f"Error processing HotpotQA sample: {e}")
            continue

def triviaqa_adapter(
    tokenizer: PreTrainedTokenizerFast,
    split: str = "train",
    streaming: bool = True,
) -> Iterator[Tuple[List[List[int]], List[int], List[str]]]:
    """Adapter for TriviaQA dataset."""
    try:
        dataset = load_dataset("trivia_qa", "rc", split=split, streaming=streaming)
        logger.info(f"Successfully loaded TriviaQA dataset, split: {split}")
    except Exception as e:
        logger.error(f"Error loading TriviaQA: {e}")
        return
    
    for item in dataset:
        try:
            # Use the first evidence document
            if not item["search_results"]:
                logger.warning("Skipping TriviaQA sample with no search results")
                continue
            
            # Check if search context exists and is not empty
            if not item["search_results"][0].get("search_context"):
                logger.warning("Skipping TriviaQA sample with empty search context")
                continue
                
            context = item["search_results"][0]["search_context"]
            question = item["question"]
            
            # Check if answer aliases exist and are not empty
            if not item.get("answer") or not item["answer"].get("aliases") or len(item["answer"]["aliases"]) == 0:
                logger.warning("Skipping TriviaQA sample with no answer aliases")
                continue
                
            # TriviaQA has multiple answer aliases, return as a list
            answers = item["answer"]["aliases"]
            
            if not context or not question or not answers:
                logger.warning("Skipping TriviaQA sample due to missing data")
                continue
            
            # Tokenize
            doc_tokens = tokenizer(context, add_special_tokens=False).input_ids
            question_tokens = tokenizer(question, add_special_tokens=False).input_ids
            
            # Chunk document
            doc_chunks = chunk_document(doc_tokens)
            
            # Skip if document chunks are empty
            if not doc_chunks:
                logger.warning("Skipping TriviaQA sample due to empty document chunks")
                continue
            
            yield doc_chunks, question_tokens, answers
            
        except Exception as e:
            logger.error(f"Error processing TriviaQA sample: {e}")
            continue

def nq_adapter(
    tokenizer: PreTrainedTokenizerFast,
    split: str = "train",
    streaming: bool = True,
) -> Iterator[Tuple[List[List[int]], List[int], List[str]]]:
    """Adapter for Natural Questions dataset."""
    try:
        dataset = load_dataset("nq_open", split=split, streaming=streaming)
        logger.info(f"Successfully loaded Natural Questions dataset, split: {split}")
    except Exception as e:
        logger.error(f"Error loading Natural Questions: {e}")
        return
    
    for item in dataset:
        try:
            # Check if question exists and is not empty
            if not item.get("question") or not item["question"].strip():
                logger.warning("Skipping NQ sample with empty question")
                continue
                
            # Check if answers exist and are not empty
            if not item.get("answer") or len(item["answer"]) == 0:
                logger.warning("Skipping NQ sample with no answers")
                continue
                
            question = item["question"]
            # NQ has multiple answers
            answers = item["answer"]
            
            # For NQ, we'll need to use a simplified context since full document may not be available
            # in the open version - generate synthetic context using answers and question
            context = f"The question '{question}' has the following information:\n"
            
            for i, answer in enumerate(answers):
                context += f"Answer {i+1}: {answer}\n"
                
            # Add some generic context padding to make it more challenging
            context += "The above information is part of a knowledge base that contains facts about various topics."
            
            # Tokenize
            doc_tokens = tokenizer(context, add_special_tokens=False).input_ids
            question_tokens = tokenizer(question, add_special_tokens=False).input_ids
            
            # Chunk document
            doc_chunks = chunk_document(doc_tokens)
            
            # Skip if document chunks are empty
            if not doc_chunks:
                logger.warning("Skipping NQ sample due to empty document chunks")
                continue
            
            yield doc_chunks, question_tokens, answers  # Return all answers
            
        except Exception as e:
            logger.error(f"Error processing NQ sample: {e}")
            continue

# Map dataset names to their adapter functions
DATASET_ADAPTERS = {
    "narrativeqa": narrativeqa_adapter,
    "hotpotqa": hotpotqa_adapter,
    "triviaqa": triviaqa_adapter,
    "nq_long": nq_adapter,
}

# ================= Mixed Dataset Streaming =================

def get_dataset_iterator(dataset_name: str, tokenizer: PreTrainedTokenizerFast, split: str = "train") -> Iterator:
    """Get iterator for a specific dataset using its adapter."""
    adapter = DATASET_ADAPTERS.get(dataset_name)
    if not adapter:
        raise ValueError(f"No adapter available for dataset: {dataset_name}")
    
    # Call the adapter with appropriate parameters
    if dataset_name == "narrativeqa":
        return adapter(tokenizer, split=split, use_summaries=True)
    else:
        return adapter(tokenizer, split=split)

def mixed_stream(dataset_names: List[str], 
                 tokenizer: PreTrainedTokenizerFast, 
                 split: str = "train", 
                 seed: Optional[int] = None) -> Iterator[Tuple[List[List[int]], List[int], List[str]]]:
    """Stream a mix of datasets with uniform random sampling.
    
    Args:
        dataset_names: List of dataset names to include in the mix
        tokenizer: Tokenizer to use for processing text
        split: Dataset split to use
        seed: Random seed for reproducibility
        
    Yields:
        Tuples of (doc_chunks, question_tokens, answers) where answers is a list of reference answers
        
    Note:
        If all dataset adapters fail (e.g., due to HuggingFace being down), this function
        will fall back to a synthetic dataset with simple QA pairs to allow training to continue.
    """
    rng = np.random.default_rng(seed)
    
    # Initialize iterators for each dataset
    iterators = {}
    dataset_errors = {}
    for name in dataset_names:
        try:
            iterators[name] = get_dataset_iterator(name, tokenizer, split)
        except Exception as e:
            error_msg = str(e)
            dataset_errors[name] = error_msg
            logger.warning(f"Failed to initialize iterator for {name}: {error_msg}")
    
    # If all adapters failed, create a synthetic fallback dataset
    if not iterators:
        logger.error(f"All dataset adapters failed: {dataset_errors}")
        logger.warning("Falling back to synthetic dataset to allow training to continue")
        
        # Create a synthetic fallback iterator
        iterators["synthetic_fallback"] = _create_synthetic_fallback(tokenizer, seed)
    
    while True:
        # Randomly select a dataset
        dataset_name = rng.choice(list(iterators.keys()))
        
        try:
            # Get next item from the selected dataset
            yield next(iterators[dataset_name])
        except StopIteration:
            # Reinitialize the iterator if it's exhausted
            logger.info(f"Reinitializing iterator for {dataset_name}")
            try:
                iterators[dataset_name] = get_dataset_iterator(dataset_name, tokenizer, split)
                yield next(iterators[dataset_name])
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to reinitialize iterator for {dataset_name}: {error_msg}")
                # Remove this dataset from the mix if we can't reinitialize
                del iterators[dataset_name]
                
                # If all iterators have failed, create a synthetic fallback
                if not iterators:
                    logger.error("All dataset iterators failed, falling back to synthetic dataset")
                    iterators["synthetic_fallback"] = _create_synthetic_fallback(tokenizer, seed)

# ================= Synthetic Fallback Dataset =================

def _create_synthetic_fallback(tokenizer: PreTrainedTokenizerFast, seed: Optional[int] = None) -> Iterator[Tuple[List[List[int]], List[int], List[str]]]:
    """Create a synthetic fallback dataset for when all real datasets fail.
    
    This function generates simple QA pairs with predictable answers to allow
    training to continue even when external data sources are unavailable.
    
    Args:
        tokenizer: Tokenizer for processing text
        seed: Random seed for reproducibility
        
    Yields:
        Tuples of (doc_chunks, question_tokens, answers) in the same format
        as the real dataset adapters
    """
    rng = np.random.default_rng(seed)
    
    # Simple QA templates that can be procedurally generated
    qa_templates = [
        # Format: (document template, question template, answer)
        (
            "The capital of France is Paris. The capital of Germany is Berlin. "
            "The capital of Italy is Rome. The capital of Spain is Madrid. "
            "The capital of Portugal is Lisbon. The capital of Greece is Athens.",
            "What is the capital of France?",
            ["Paris"]
        ),
        (
            "The first president of the United States was George Washington. "
            "The second president was John Adams. The third president was Thomas Jefferson. "
            "The 16th president was Abraham Lincoln. The 32nd president was Franklin D. Roosevelt.",
            "Who was the first president of the United States?",
            ["George Washington"]
        ),
        (
            "Water boils at 100 degrees Celsius at sea level. "
            "Water freezes at 0 degrees Celsius. "
            "The human body temperature is approximately 37 degrees Celsius.",
            "At what temperature does water boil at sea level?",
            ["100 degrees Celsius", "100 degrees", "100°C"]
        ),
        (
            "The Earth orbits around the Sun. The Moon orbits around the Earth. "
            "Mercury is the closest planet to the Sun. Venus is the second planet from the Sun. "
            "Mars is the fourth planet from the Sun.",
            "What does the Earth orbit around?",
            ["the Sun", "Sun"]
        ),
        (
            "The primary colors are red, blue, and yellow. "
            "When you mix red and blue, you get purple. "
            "When you mix blue and yellow, you get green. "
            "When you mix red and yellow, you get orange.",
            "What color do you get when you mix blue and yellow?",
            ["green"]
        )
    ]
    
    # Keep yielding examples indefinitely
    while True:
        # Select a random template
        doc_template, question_template, answers = qa_templates[rng.integers(0, len(qa_templates))]
        
        # Add some random padding to make documents longer and more varied
        padding_sentences = [
            "This is additional context to make the document longer.",
            "The following information is part of a knowledge base.",
            "Here are some facts about various topics.",
            "This document contains information on multiple subjects.",
            "The data presented here is for educational purposes."
        ]
        
        # Add 2-5 random padding sentences
        num_padding = rng.integers(2, 6)
        padding_indices = rng.choice(len(padding_sentences), size=num_padding, replace=True)
        padding_text = " ".join([padding_sentences[i] for i in padding_indices])
        
        # Combine the template with padding
        document = f"{padding_text} {doc_template} {padding_text}"
        
        # Tokenize
        doc_tokens = tokenizer(document, add_special_tokens=False).input_ids
        question_tokens = tokenizer(question_template, add_special_tokens=False).input_ids
        
        # Chunk document
        doc_chunks = chunk_document(doc_tokens)
        
        yield doc_chunks, question_tokens, answers

# ================= Curriculum Learning =================

class CurriculumDataLoader:
    """Curriculum-based data loader that gradually adds more challenging datasets.
    
    The loader starts with the easiest dataset (NarrativeQA) and adds more challenging
    datasets as the agent's performance improves, measured by QA score thresholds.
    
    Note: Once a dataset is added to the mix, it remains active even if the agent's
    performance later drops below the threshold that triggered its addition. This
    design choice encourages the agent to adapt to the more challenging mix rather
    than reverting to easier datasets when performance temporarily regresses.
    """
    
    def __init__(
                 self, 
                 tokenizer: PreTrainedTokenizerFast,
                 dataset_order: List[str] = ["narrativeqa", "hotpotqa", "triviaqa", "nq_long"],
                 qa_thresholds: List[float] = [0.3, 0.4, 0.5],  # Thresholds to unlock next dataset
                 split: str = "train",
                 seed: Optional[int] = None,
                 regress_ok: bool = True):
        """Initialize curriculum data loader.
        
        Args:
            tokenizer: Tokenizer for processing text
            dataset_order: Order of datasets from easiest to hardest
            qa_thresholds: QA score thresholds for unlocking next dataset
            split: Dataset split to use
            seed: Random seed for reproducibility
            regress_ok: If True (default), keeps harder datasets in the mix even if
                        performance drops below the threshold that added them.
                        If False, removes datasets when performance drops below threshold.
        """
        self.tokenizer = tokenizer
        self.dataset_order = dataset_order
        self.qa_thresholds = qa_thresholds
        self.split = split
        self.seed = seed
        self.regress_ok = regress_ok
        
        # Start with just the first (easiest) dataset
        self.active_datasets = [dataset_order[0]]
        self.current_level = 0
        self.current_iterator = self._create_iterator()
        
        logger.info(f"Curriculum initialized with dataset: {self.active_datasets[0]}")
    
    def _create_iterator(self) -> Iterator:
        """Create a mixed stream iterator with current active datasets."""
        return mixed_stream(self.active_datasets, self.tokenizer, self.split, self.seed)
    
    def update_curriculum(self, qa_score: float) -> bool:
        """Update the curriculum based on current QA score.
        
        By default, once a dataset is added, it remains in the mix even if performance
        later drops below the threshold (regress_ok=True). Set regress_ok=False during
        initialization to enable removing datasets when performance drops.
        
        Args:
            qa_score: Current QA score to evaluate against thresholds
            
        Returns:
            bool: True if curriculum was updated (datasets added or removed), False otherwise
        """
        updated = False
        
        # Check if we can advance to the next level
        if self.current_level < len(self.qa_thresholds) and qa_score >= self.qa_thresholds[self.current_level]:
            next_dataset_idx = self.current_level + 1
            if next_dataset_idx < len(self.dataset_order):
                next_dataset = self.dataset_order[next_dataset_idx]
                self.active_datasets.append(next_dataset)
                self.current_level += 1
                self.current_iterator = self._create_iterator()
                logger.info(f"Curriculum advanced to level {self.current_level}")
                logger.info(f"Added dataset: {next_dataset}")
                logger.info(f"Active datasets: {self.active_datasets}")
                updated = True
        
        # Check if we need to regress (only if regress_ok is False)
        elif not self.regress_ok and self.current_level > 0:
            # Find the highest threshold that the current score exceeds
            new_level = 0
            for i, threshold in enumerate(self.qa_thresholds):
                if qa_score >= threshold:
                    new_level = i + 1
                else:
                    break
                    
            # If we need to drop back levels
            if new_level < self.current_level:
                # Keep only datasets up to the new level + 1 (the first dataset is always kept)
                self.active_datasets = self.dataset_order[:new_level + 1]
                self.current_level = new_level
                self.current_iterator = self._create_iterator()
                logger.info(f"Curriculum regressed to level {self.current_level}")
                logger.info(f"Active datasets: {self.active_datasets}")
                updated = True
                
        return updated
    
    def get_data_loader(self) -> Callable:
        """Get a data loader function that returns the current mixed stream iterator."""
        def data_loader_fn():
            return self.current_iterator
        return data_loader_fn
        yield next(iters[i])