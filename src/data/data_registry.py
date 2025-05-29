"""Data registry for QA datasets with standardized adapters and mixed streaming.

This module provides a unified interface for loading and mixing multiple QA datasets,
with support for curriculum learning based on agent performance.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Iterator, Callable
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import logging
import os
import pickle
import time

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

# Standard format for QA datasets: Dictionary with the following keys:
# {
#     "doc_chunks": List[List[int]],  # tokenized chunks of the document
#     "question_ids": List[int],      # tokenized question
#     "answers": List[str],           # list of answer strings
#     "meta": Dict[str, Any]          # metadata like dataset name, example ID, etc.
# }

def narrativeqa_adapter(
    tokenizer: PreTrainedTokenizerFast, 
    split: str = "train", 
    streaming: bool = True,
    use_summaries: bool = True,  # Use summaries instead of full stories for easier training
    chunk_size: int = 256,
) -> Iterator[Dict[str, Any]]:
    """Adapter for NarrativeQA dataset."""
    try:
        # Load the dataset - note: the dataset ID is "narrativeqa" not "deepmind/narrativeqa"
        dataset = load_dataset("narrativeqa", split=split, streaming=streaming)
        logger.debug(f"Successfully loaded NarrativeQA dataset, split: {split}")
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
            doc_chunks = chunk_document(doc_tokens, chunk_size)
            
            # Skip if document chunks are empty
            if not doc_chunks:
                logger.warning(f"Skipping sample due to empty document chunks: {item.get('id', 'N/A')}")
                continue
            
            # Create metadata
            meta = {
                "dataset": "narrativeqa",
                "id": item.get("id", "N/A"),
                "use_summaries": use_summaries
            }
            
            yield {
                "doc_chunks": doc_chunks,
                "question_ids": question_tokens,
                "answers": answers,
                "meta": meta
            }
            
        except Exception as e:
            logger.error(f"Error processing NarrativeQA sample: {e}")
            continue

def hotpotqa_adapter(
    tokenizer: PreTrainedTokenizerFast,
    split: str = "train",
    streaming: bool = True,
    chunk_size: int = 256,
) -> Iterator[Dict[str, Any]]:
    """Adapter for HotpotQA dataset."""
    if split == "validation":
        logger.debug(f"Attempting to load HotpotQA validation split...")
    try:
        dataset = load_dataset(
            "hotpot_qa", 
            "fullwiki", 
            split=split, 
            streaming=streaming,
            trust_remote_code=True
        )
        logger.debug(f"Successfully loaded HotpotQA dataset, split: {split}")
        if split == "validation":
            logger.debug(f"Successfully loaded HotpotQA validation split.")
    except Exception as e:
        logger.error(f"Error loading HotpotQA (split={split}): {e}")
        return
    
    printed_context_debug = False # Flag to print context only once
    for item in dataset:
        if split == "validation":
             logger.debug(f"Processing HotpotQA validation item: {item.get('id', 'N/A')}")
        try:
            # Check if context exists and is not empty
            raw_context = item.get("context") # Get raw context first
            if not raw_context:
                logger.warning(f"Skipping HotpotQA sample {item.get('id', 'N/A')} with empty or missing context")
                continue

            # --- Debugging: Print raw context structure for validation ---
            if split == "validation" and not printed_context_debug:
                 logger.debug(f"Raw HotpotQA validation context structure for item {item.get('id', 'N/A')}: {raw_context}")
                 # Check type and keys if it's a dict
                 if isinstance(raw_context, dict):
                     logger.debug(f"Context keys: {list(raw_context.keys())}")
                 printed_context_debug = True # Only print for the first item encountered
            # --- End Debugging ---

            # --- Attempt to extract paragraphs/sentences ---
            context_list = []
            if isinstance(raw_context, dict):
                 # Try extracting 'paragraphs' first
                 context_list = raw_context.get("paragraphs", [])
                 # If 'paragraphs' is empty or not found, try 'sentences' directly
                 if not context_list:
                      context_list = raw_context.get("sentences", [])
            elif isinstance(raw_context, list): # Handle if context is just a list
                 context_list = raw_context
            supporting_facts = item.get("supporting_facts", [])
            # Always default to empty list, do not skip samples for missing/invalid supporting_facts
            support_titles = [sf[0] for sf in supporting_facts[:2]] if supporting_facts and isinstance(supporting_facts, list) and len(supporting_facts) >= 2 else []
            titles = raw_context.get("title", []) if isinstance(raw_context, dict) else []
            para_dict = dict(zip(titles, context_list)) if titles and len(titles) == len(context_list) else {}
            selected_paras = []
            # (a) Try supporting_facts→title match
            if support_titles:
                for t in support_titles:
                    if t in para_dict:
                        selected_paras.append(para_dict[t])
            # (b) If not enough, try TF-IDF/keyword match against answers
            if len(selected_paras) < 2 and context_list and item.get("answers"):
                # Simple TF-IDF: score by answer word overlap (case-insensitive)
                answers = item["answers"]
                answer_words = set()
                for ans in answers:
                    answer_words.update(ans.lower().split())
                para_scores = []
                for para in context_list:
                    if isinstance(para, list):
                        para_text = " ".join(para)
                    else:
                        para_text = str(para)
                    para_words = set(para_text.lower().split())
                    overlap = len(answer_words & para_words)
                    para_scores.append(overlap)
                # Pick top 2 scoring paragraphs (break ties by order)
                best_idx = np.argsort(para_scores)[::-1][:2]
                for idx in best_idx:
                    if context_list[idx] not in selected_paras:
                        selected_paras.append(context_list[idx])
                # Truncate to 2
                selected_paras = selected_paras[:2]
            # Fallback: first two paragraphs by order
            if not selected_paras and len(context_list) >= 2:
                selected_paras = context_list[:2]
            elif not selected_paras and len(context_list) == 1:
                selected_paras = [context_list[0]]
            if not selected_paras:
                logger.warning(f"Skipping HotpotQA sample {item.get('id', 'N/A')} because could not find any paragraphs for context.")
                continue
            # Flatten to text
            context = ""
            for para in selected_paras:
                if isinstance(para, list):
                    context += " ".join(para) + "\n"
                elif isinstance(para, str):
                    context += para + "\n"

            # ... (rest of the adapter: check context, question, answer, tokenize, chunk, yield) ...
            if not context: # Check if context string ended up empty
                 logger.warning(f"Skipping HotpotQA sample {item.get('id', 'N/A')} because context string is empty after processing.")
                 continue

            question = item["question"]
            answers = [item["answer"]] 

            if not question or not answers or not answers[0]: 
                logger.warning(f"Skipping HotpotQA sample {item.get('id', 'N/A')} due to missing question/answer")
                continue

            # Tokenize
            doc_tokens = tokenizer(context, add_special_tokens=False).input_ids
            question_tokens = tokenizer(question, add_special_tokens=False).input_ids

            # Chunk document
            doc_chunks = chunk_document(doc_tokens, chunk_size)

            # Skip if document chunks are empty
            if not doc_chunks or not doc_chunks[0]: 
                 logger.warning(f"Skipping HotpotQA sample {item.get('id', 'N/A')} due to empty document chunks")
                 continue

            # Create metadata
            meta = {
                "dataset": "hotpotqa",
                "id": item.get("id", "N/A"),
                "supporting_facts": item.get("supporting_facts", [])
            }

            yield {
                "doc_chunks": doc_chunks,
                "question_ids": question_tokens,
                "answers": answers,
                "meta": meta
            }

        except Exception as e:
            # Log the full exception for better debugging
            logger.error(f"Error processing HotpotQA sample (split={split}, id={item.get('id', 'N/A')}): {e}", exc_info=True)
            continue

def triviaqa_adapter(
    tokenizer: PreTrainedTokenizerFast,
    split: str = "train",
    streaming: bool = True,
    chunk_size: int = 256,
) -> Iterator[Dict[str, Any]]:
    """Adapter for TriviaQA dataset.
    
    Args:
        tokenizer: Tokenizer for processing text
        split: Dataset split to use
        streaming: Whether to stream the dataset
    """
    # We need better debugging for validation data structure
    debug_count = 0
    debug_max = 3  # Number of validation samples to print for debugging
        
    if split == "validation":
        logger.debug(f"Attempting to load TriviaQA validation split...")
    
    try:
        # For TriviaQA, we need to use the right split name
        # For validation, try 'validation' first, then fallback to 'dev' if needed
        actual_split = split
        if split == "validation":
            try:
                dataset = load_dataset("trivia_qa", "rc", split="validation", streaming=streaming)
                logger.debug(f"Successfully loaded TriviaQA validation split.")
            except Exception:
                logger.warning(f"'validation' split not found for TriviaQA, trying 'dev' instead...")
                try:
                    dataset = load_dataset("trivia_qa", "rc", split="dev", streaming=streaming)
                    logger.debug(f"Successfully loaded TriviaQA 'dev' split.")
                    actual_split = "dev"
                except Exception as e:
                    logger.error(f"Failed to load TriviaQA validation data with either 'validation' or 'dev' splits: {e}")
                    return
        else:
            dataset = load_dataset("trivia_qa", "rc", split=split, streaming=streaming)
            logger.debug(f"Successfully loaded TriviaQA dataset, split: {split}")
    except Exception as e:
        logger.error(f"Error loading TriviaQA (split={split}): {e}")
        return
    
    # Process dataset items
    for item in dataset:
        # Only log minimal info for validation items to reduce output noise
        if split == "validation" and debug_count < debug_max:
            # Simple log just noting we're processing an item
            if 'question_id' in item and 'question' in item:
                logger.debug(f"Processing TriviaQA validation item {item['question_id']}: {item['question'][:50]}...")
            debug_count += 1
        try:
            # Extract context based on document structure, with multiple fallback paths
            context_source = None
            evidence_found = False
            context_paths = []
            
            # Try search_results path first (common in training data)
            search_results = item.get("search_results")
            if search_results:
                # Handle both list-of-dicts and dict-of-lists
                if isinstance(search_results, list) and len(search_results) > 0:
                    first_result = search_results[0]
                    if isinstance(first_result, dict) and "search_context" in first_result:
                        context_source = first_result.get("search_context")
                        context_paths.append("search_results[0].search_context")
                elif isinstance(search_results, dict):
                    # HF streaming split: dict of lists, e.g. {"search_context": [str, str, ...], ...}
                    for key, val in search_results.items():
                        if key == "search_context" and isinstance(val, list) and len(val) > 0:
                            context_source = val[0]
                            context_paths.append("search_results['search_context'][0]")
                            break
            
            # Try evidence path (common in validation data)
            evidence = item.get("evidence")
            if not context_source and evidence and isinstance(evidence, list) and len(evidence) > 0:
                if isinstance(evidence[0], str):
                    # If evidence is a list of strings, combine the first few
                    evidence_texts = evidence[:3]  # Limit to first 3 pieces
                    context_source = " ".join(evidence_texts)
                    context_paths.append("evidence (strings)")
                    evidence_found = True
                elif isinstance(evidence[0], dict):
                    # If evidence is a list of dicts, look for text fields
                    for key in ["text", "content", "passage", "context"]:
                        if key in evidence[0]:
                            context_source = evidence[0][key]
                            context_paths.append(f"evidence[0].{key}")
                            evidence_found = True
                            break
            
            # Try entity_pages path
            if not context_source and not evidence_found:
                entity_pages = item.get("entity_pages")
                if entity_pages and isinstance(entity_pages, list) and len(entity_pages) > 0:
                    if isinstance(entity_pages[0], str):
                        context_source = entity_pages[0]
                        context_paths.append("entity_pages[0] (string)")
                    elif isinstance(entity_pages[0], dict):
                        # Try various possible field names
                        for key in ["text", "content", "page_content", "wiki_context"]:
                            if key in entity_pages[0]:
                                context_source = entity_pages[0][key]
                                context_paths.append(f"entity_pages[0].{key}")
                                break
            
            # Try wiki_context or normal context fields
            if not context_source:
                for key in ["wiki_context", "context", "text", "content"]:
                    if key in item and item[key]:
                        context_source = item[key]
                        context_paths.append(key)
                        break
            
            # Try aliases[0] as last resort (if present)
            if not context_source and item.get("aliases") and len(item["aliases"]) > 0:
                context_source = f"Question about: {item['aliases'][0]}"
                context_paths.append("aliases[0]")
            
            # Last resort - minimal context with just the question
            if not context_source:
                if split == "validation":
                    # For validation, use the question itself as minimal context
                    question = item.get("question", "Unknown question")
                    context_source = f"Question: {question}"
                    context_paths.append("question (fallback)")
                    # Log this case
                    logger.info(f"Using question as minimal context for TriviaQA sample {item.get('question_id', 'N/A')}")
                else:
                    # For training, we can be more selective
                    logger.warning(f"Skipping TriviaQA sample {item.get('question_id', 'N/A')} with no usable context source found.")
                    continue
                    
            # Log which path we ended up using for debugging
            if split == "validation" and debug_count <= debug_max:
                logger.debug(f"Context source path(s) for {item.get('question_id', 'N/A')}: {context_paths}")
                    
            context = context_source # Use the context we found

            question = item.get("question")
            answer_data = item.get("answer")

            if not question:
                logger.warning(f"Skipping TriviaQA sample {item.get('question_id', 'N/A')} with no question")
                continue

            if not answer_data or not isinstance(answer_data, dict):
                 logger.warning(f"Skipping TriviaQA sample {item.get('question_id', 'N/A')} with missing or malformed answer data")
                 continue

            aliases = answer_data.get("aliases")
            if not aliases or not isinstance(aliases, list) or len(aliases) == 0:
                logger.warning(f"Skipping TriviaQA sample {item.get('question_id', 'N/A')} with no answer aliases")
                continue

            answers = aliases # TriviaQA has multiple answer aliases

            # Tokenize
            doc_tokens = tokenizer(context, add_special_tokens=False).input_ids
            question_tokens = tokenizer(question, add_special_tokens=False).input_ids

            # Chunk document
            doc_chunks = chunk_document(doc_tokens, chunk_size)

            # Skip if document chunks are empty
            if not doc_chunks or not doc_chunks[0]: 
                 logger.warning(f"Skipping TriviaQA sample {item.get('question_id', 'N/A')} due to empty document chunks")
                 continue

            # Create metadata
            meta = {
                "dataset": "triviaqa",
                "id": item.get("question_id", item.get("id", "N/A")), 
                "entity_pages": item.get("entity_pages", [])
            }

            yield {
                "doc_chunks": doc_chunks,
                "question_ids": question_tokens,
                "answers": answers,
                "meta": meta
            }

        except Exception as e:
            # Log the full exception for better debugging
            logger.error(f"Error processing TriviaQA sample (split={split}, id={item.get('question_id', 'N/A')}): {e}", exc_info=True) 
            continue

def nq_adapter(
    tokenizer: PreTrainedTokenizerFast,
    split: str = "train",
    streaming: bool = True,
    chunk_size: int = 256,
) -> Iterator[Dict[str, Any]]:
    """Adapter for Natural Questions dataset."""
    try:
        dataset = load_dataset("nq_open", split=split, streaming=streaming)
        logger.debug(f"Successfully loaded Natural Questions dataset, split: {split}")
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
            
            # For NQ, use only the question and generic context (do NOT include answers in context)
            context = f"The question '{question}' has the following information.\n"
            # Add some generic context padding to make it more challenging
            context += "The above information is part of a knowledge base that contains facts about various topics."
            
            # Tokenize
            doc_tokens = tokenizer(context, add_special_tokens=False).input_ids
            question_tokens = tokenizer(question, add_special_tokens=False).input_ids
            
            # Chunk document
            doc_chunks = chunk_document(doc_tokens, chunk_size)
            
            # Skip if document chunks are empty
            if not doc_chunks:
                logger.warning("Skipping NQ sample due to empty document chunks")
                continue
            
            # Create metadata
            meta = {
                "dataset": "nq_long",
                "id": item.get("id", "N/A"),
                "synthetic_context": True  # Flag indicating this is synthetic context
            }
            
            yield {
                "doc_chunks": doc_chunks,
                "question_ids": question_tokens,
                "answers": answers,
                "meta": meta
            }
            
        except Exception as e:
            logger.error(f"Error processing NQ sample: {e}")
            continue

# Map dataset names to their adapter functions
DATASET_ADAPTERS = {
    "narrativeqa": narrativeqa_adapter,
    "hotpotqa": hotpotqa_adapter,
    "triviaqa": triviaqa_adapter, # Corrected name
    "nq_long": nq_adapter,  # Corrected name based on function definition
}

# ================= Helper for Local Cache Loading =================

def _save_to_cache(cache_dir: str, samples: List[Dict[str, Any]], max_samples: int = 1000) -> None:
    """Save processed data samples to a cache file.
    
    Args:
        cache_dir: Directory to save the cache file
        samples: List of processed data samples to cache
        max_samples: Maximum number of samples per cache file
    """
    if not samples:
        logger.warning("No samples to cache")
        return
        
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a unique filename based on timestamp
    timestamp = int(time.time())
    cache_file = os.path.join(cache_dir, f"cache_{timestamp}.pkl")
    
    logger.info(f"Saving {len(samples)} processed samples to cache: {cache_file}")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(samples, f)
        logger.info(f"Successfully cached samples to {cache_file}")
    except Exception as e:
        logger.error(f"Error saving to cache file {cache_file}: {e}")


def _load_from_local_cache(cache_dir: str) -> Iterator[Dict[str, Any]]:
    """Loads pre-processed data from .pkl files in a local directory.

    Assumes each .pkl file contains a list of dictionaries,
    each matching the standard adapter output format.

    Args:
        cache_dir: Path to the directory containing .pkl cache files.

    Yields:
        Dict[str, Any]: Parsed data samples from the cache.
    """
    if not os.path.isdir(cache_dir):
        logger.error(f"Local cache directory not found: {cache_dir}")
        yield from []  # Use yield from to return an empty iterator
        return

    logger.info(f"Loading validation data from local PKL cache: {cache_dir}")
    found_files = False
    samples = []  # Collect all samples first
    
    for filename in os.listdir(cache_dir):
        if filename.endswith(".pkl"):
            file_path = os.path.join(cache_dir, filename)
            logger.debug(f"Reading cache file: {file_path}")
            found_files = True
            try:
                with open(file_path, 'rb') as f: # Open in binary read mode
                    data_list = pickle.load(f) # Load the entire object from the pickle file
                    if isinstance(data_list, list):
                        # Add all valid samples to our collection
                        samples.extend([item for item in data_list if isinstance(item, dict)])
                    else:
                        logger.warning(f"Skipping cache file {filename}: Expected a list, found {type(data_list)}")
            except pickle.UnpicklingError as e:
                 logger.error(f"Error unpickling cache file {filename}: {e}")
            except Exception as e:
                logger.error(f"Error reading cache file {filename}: {e}")

    if not found_files:
        logger.warning(f"No .pkl files found in cache directory: {cache_dir}")
        yield from []  # Use yield from to return an empty iterator
        return
    
    if not samples:
        logger.warning(f"No valid samples found in cache files in: {cache_dir}")
        yield from []  # Use yield from to return an empty iterator
        return
    
    logger.info(f"Successfully loaded {len(samples)} samples from cache")
    # Now yield all collected samples
    for sample in samples:
        yield sample

# ================= Data Loading Functions =================

def get_dataset_iterator(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerFast,
    split: str = "train",
    chunk_size: int = 256,
    config: Optional[Dict[str, Any]] = None
) -> Iterator[Dict[str, Any]]:
    """Get iterator for a specific dataset using its adapter or load from local cache.

    If `dataset_name` looks like a directory path, it attempts to load
    pre-processed data from .pkl files within that directory.
    Otherwise, it treats `dataset_name` as a HuggingFace dataset identifier.

    Args:
        dataset_name: Name of the dataset or path to a local cache directory.
        tokenizer: Tokenizer for processing text (used only for HF loading).
        split: Dataset split to use (used only for HF loading).
        chunk_size: Size of document chunks (used only for HF loading).
        config: Optional configuration dictionary, potentially containing dataset-specific settings.

    Returns:
        Iterator yielding dataset examples in standardized dictionary format.

    Raises:
        ValueError: If the dataset name is not recognized and not a valid path,
                    or if the corresponding adapter function is not found.
    """
    # First check for validation cache path in config
    if config and "validation_cache_path" in config and config["validation_cache_path"]:
        cache_path = config["validation_cache_path"]
        if os.path.isdir(cache_path):
            # Check if there are actual .pkl files in the cache directory
            pkl_files = [f for f in os.listdir(cache_path) if f.endswith('.pkl')]
            if pkl_files:  # Only use cache if it actually contains data
                it = _load_from_local_cache(cache_path)
                return it
    
    # Check if dataset_name itself is a local path
    if os.path.isdir(dataset_name):
        # Check if there are actual .pkl files in the directory
        pkl_files = [f for f in os.listdir(dataset_name) if f.endswith('.pkl')]
        if pkl_files:
            it = _load_from_local_cache(dataset_name)
            return it
        else:
            logger.info(f"Directory exists but contains no .pkl files: {dataset_name}")

    dataset_cfg = None
    if config:
        # OmegaConf style
        if hasattr(config, "data") and hasattr(config.data, "dataset_config") and hasattr(config.data.dataset_config, dataset_name):
            dataset_cfg = getattr(config.data.dataset_config, dataset_name)
        # dict style
        elif isinstance(config, dict) and "dataset_config" in config and dataset_name in config["dataset_config"]:
            dataset_cfg = config["dataset_config"][dataset_name]
    if dataset_cfg is not None and not isinstance(dataset_cfg, dict):
        dataset_cfg = dict(dataset_cfg)

    # Add chunk_size and split to the specific config for the adapter
    if dataset_cfg:
        dataset_cfg['chunk_size'] = chunk_size

    # Attempt to get the adapter function for this dataset
    if dataset_name not in DATASET_ADAPTERS:
        registered_datasets = list(DATASET_ADAPTERS.keys())
        raise ValueError(f"Dataset '{dataset_name}' not recognized. Available datasets: {registered_datasets}")
    
    adapter_fn = DATASET_ADAPTERS[dataset_name]
    
    # Apply any dataset-specific configurations
    adapter_kwargs = {
        "tokenizer": tokenizer,
        "split": split,
        "streaming": True,
        "chunk_size": chunk_size
    }
    
    # Add any dataset-specific configuration from the main config
    if dataset_cfg:
        adapter_kwargs.update(dataset_cfg)
    
    # Check if we should cache results
    should_cache = False
    cache_path = None
    if config and "validation_cache_path" in config:
        cache_path = config["validation_cache_path"]
        # Only cache if path exists but no .pkl files yet
        if cache_path and os.path.isdir(cache_path):
            pkl_files = [f for f in os.listdir(cache_path) if f.endswith('.pkl')]
            should_cache = len(pkl_files) == 0
    
    try:
        # Call the adapter function to get the dataset iterator
        iterator = adapter_fn(**adapter_kwargs)
        
        # If caching is enabled, collect and cache samples
        if should_cache and cache_path:
            logger.info(f"Will cache processed examples to: {cache_path}")
            cached_samples = []
            max_cache_samples = 1000  # Limit cache size for evaluation
            
            # Return a wrapped iterator that caches samples as they're processed
            def caching_iterator():
                try:
                    sample_count = 0
                    for sample in iterator:
                        # Add to cache list if we haven't hit the limit
                        if sample_count < max_cache_samples:
                            cached_samples.append(sample)
                            sample_count += 1
                            
                            # Save cache when we hit the sample limit
                            if sample_count == max_cache_samples:
                                _save_to_cache(cache_path, cached_samples, max_cache_samples)
                        
                        # Yield the sample to the caller
                        yield sample
                        
                    # Save any remaining samples at the end
                    if cached_samples and sample_count < max_cache_samples:
                        _save_to_cache(cache_path, cached_samples, max_cache_samples)
                        
                except Exception as e:
                    logger.error(f"Error in caching iterator: {str(e)}")
                    # Attempt to save any samples we managed to process
                    if cached_samples:
                        _save_to_cache(cache_path, cached_samples, max_cache_samples)
            
            return caching_iterator()
        else:
            return iterator
    except Exception as e:
        logger.error(f"Error loading dataset '{dataset_name}': {str(e)}")
        return iter([])  # Return empty iterator

# Dataset-specific configurations will come from the config.yaml file
# This is kept for backward compatibility but will be deprecated
DATASET_CONFIG = {}

# ================= Mixed Dataset Streaming =================

def mixed_stream(dataset_names: List[str], 
                 tokenizer: PreTrainedTokenizerFast, 
                 split: str = "train",
                 chunk_size: int = 256,
                 seed: Optional[int] = None,
                 config: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
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
    
    # Cache the dataset iterators to avoid reloading from HuggingFace
    iterator_cache = {}
    
    # Function to get or create a dataset iterator
    def get_or_create_iterator(name):
        # Check if we already have a cached iterator for this dataset
        if name in iterator_cache:
            logger.debug(f"[DEBUG] Using cached iterator instance for {name}")
            return iterator_cache[name]
            
        # Create a new iterator
        logger.info(f"[DEBUG] Creating new iterator for {name} in mixed_stream")
        iterator = get_dataset_iterator(name, tokenizer, split, chunk_size, config)
        logger.info(f"[DEBUG] mixed_stream got iterator type: {type(iterator)} for {name}")
        iterator_cache[name] = iterator
        return iterator
    
    # Initialize iterators for each dataset
    iterators = {}
    dataset_errors = {}
    for name in dataset_names:
        # Set validation_cache_path per dataset
        dataset_cache_path = os.path.join("data", "validation_cache", name, split)
        config_for_dataset = dict(config) if config else {}
        config_for_dataset["validation_cache_path"] = dataset_cache_path
        logger.info(f"[DEBUG] Setting validation_cache_path for {name}: {dataset_cache_path}")
        def get_iterator_with_cache():
            return get_dataset_iterator(name, tokenizer, split, chunk_size, config_for_dataset)
        iterator = get_iterator_with_cache()
        if iterator is not None:
            iterators[name] = iterator
        else:
            dataset_errors[name] = "Failed to initialize iterator"
    
    # If all adapters failed, create a synthetic fallback dataset
    if not iterators:
        logger.error(f"All dataset adapters failed: {dataset_errors}")
        logger.warning("Falling back to synthetic dataset to allow training to continue")
        
        # Create a synthetic fallback iterator
        iterators["synthetic_fallback"] = _create_synthetic_fallback(tokenizer, seed, chunk_size)
    
    while True:
        # Randomly select a dataset
        dataset_name = rng.choice(list(iterators.keys()))
        
        try:
            # Get next item from the selected dataset
            # logger.info(f"[DEBUG] mixed_stream yielding from iterator for {dataset_name} (type: {type(iterators[dataset_name])})")
            yield next(iterators[dataset_name])
        except StopIteration:
            # Reinitialize the iterator if it's exhausted
            logger.info(f"[DEBUG] mixed_stream exhausted iterator for {dataset_name}, reinitializing...")
            # Try to get a cached iterator first
            iterator = get_or_create_iterator(dataset_name)
            logger.info(f"[DEBUG] mixed_stream after reinit, got iterator type: {type(iterator)} for {dataset_name}")
            if iterator is not None:
                iterators[dataset_name] = iterator
            else:
                logger.error(f"[DEBUG] Failed to reinitialize iterator for {dataset_name}")
                # Remove this dataset from the mix if we can't reinitialize it
                del iterators[dataset_name]
                
                # If no datasets left, fall back to synthetic data
                if not iterators:
                    logger.error("[DEBUG] All dataset iterators failed, falling back to synthetic dataset")
                    iterators["synthetic_fallback"] = _create_synthetic_fallback(tokenizer, seed, chunk_size)

# ================= Synthetic Fallback Dataset =================

def _create_synthetic_fallback(tokenizer: PreTrainedTokenizerFast, seed: Optional[int] = None, chunk_size: int = 256) -> Iterator[Dict[str, Any]]:
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
        doc_chunks = chunk_document(doc_tokens, chunk_size)
        
        # Create metadata
        meta = {
            "dataset": "synthetic_fallback",
            "id": f"synthetic_{rng.integers(0, 10000)}",
            "synthetic": True
        }
        
        yield {
            "doc_chunks": doc_chunks,
            "question_ids": question_tokens,
            "answers": answers,
            "meta": meta
        }

# ================= Curriculum Learning =================

class CurriculumDataLoader:
    """Curriculum-based data loader that gradually adds more challenging datasets.
    
    The loader starts with the easiest dataset (NarrativeQA) and adds more challenging
    datasets as the agent's performance improves, measured by QA score thresholds.
    
    Note: Once a dataset is added to the mix, it remains active even if performance
    later drops below the threshold that triggered its addition. This design choice
    encourages the agent to adapt to the more challenging mix rather than reverting
    to easier datasets when performance temporarily regresses.
    """
    
    def __init__(
                 self, 
                 tokenizer: PreTrainedTokenizerFast,
                 dataset_order: List[str] = ["narrativeqa", "hotpotqa", "triviaqa", "nq_long"],
                 qa_thresholds: List[float] = [0.3, 0.4, 0.5],  # Thresholds to unlock next dataset
                 split: str = "train",
                 seed: Optional[int] = None,
                 regress_ok: bool = True,
                 chunk_size: int = 256,
                 dataset_config: Dict[str, Dict[str, Any]] = None,
                 config: Optional[Dict[str, Any]] = None):
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
            chunk_size: Size of document chunks
            dataset_config: Optional configuration for specific datasets, e.g.,
                           {"narrativeqa": {"use_summaries": False}} to use full stories
                           instead of summaries for NarrativeQA
        """
        self.tokenizer = tokenizer
        self.dataset_order = dataset_order
        self.qa_thresholds = qa_thresholds
        self.split = split
        self.seed = seed
        self.regress_ok = regress_ok
        self.chunk_size = chunk_size
        self.config = config
        
        # Keep legacy dataset_config support for backward compatibility
        self.dataset_config = dataset_config
        
        # Start with just the first (easiest) dataset
        self.active_datasets = [dataset_order[0]]
        self.current_level = 0
        self.current_iterator = self._create_iterator()
        
        logger.debug(f"Curriculum initialized with dataset: {self.active_datasets[0]}")
    
    def _create_iterator(self, seed=None):
        """Create a mixed stream iterator with current active datasets."""
        return mixed_stream(
            dataset_names=self.active_datasets,
            tokenizer=self.tokenizer,
            split=self.split,
            chunk_size=self.chunk_size,
            seed=seed if seed is not None else self.seed,
            config=self.config
        )
    
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
        """Get a data loader function that returns a fresh mixed stream iterator (independent per call)."""
        def data_loader_fn(seed=None):
            return self._create_iterator(seed=seed)
        return data_loader_fn