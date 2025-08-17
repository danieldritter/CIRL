# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import random
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import datasets
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


class CausalTextDataset(Dataset):
    """
    Load and preprocess causal factor discovery data from Parquet files.

    Similar to RLHFDataset but specialized for causal factor discovery tasks.
    Handles text document sampling and causal factor discovery prompting.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        # Standard VERL dataset config
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/causal"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.use_shm = config.get("use_shm", False)

        # Causal discovery specific config
        self.sample_docs_per_input = config.get("sample_docs_per_input", 5)
        self.sampling_strategy = config.get("sampling_strategy", "random")
        self.prompt_template = config.get("prompt_template", "default")
        self.max_factors = config.get("max_factors", 10)
        self.documents_key = config.get("documents_key", "documents")
        self.target_outcome_key = config.get("target_outcome_key", "target_outcome")

        self.serialize_dataset = False

        # Initialize prompt templates
        self.prompt_templates = self._init_prompt_templates()

        self._download()
        self._read_files_and_tokenize()

    def _init_prompt_templates(self) -> Dict[str, str]:
        """Initialize causal discovery prompt templates."""
        return {
            "default": """Given the following text documents, identify the most important causal factors that influence {target_outcome}.

Text Documents:
{text_documents}

Please list the top {num_factors} causal factors in JSON format as a list of strings:
""",
            "causal_discovery_v1": """You are an expert at identifying causal relationships. Analyze the following text documents to identify factors that causally influence {target_outcome}.

Documents to analyze:
{text_documents}

Task: Identify the {num_factors} most important causal factors that influence {target_outcome}. Focus on factors that are:
1. Directly actionable or measurable
2. Have clear causal mechanisms (not just correlations)
3. Can be identified from the text content

Please provide your response as a list of factor names (e.g. [factor_1,factor_2,...]):
""",
            "structured": """Analyze the text documents and identify causal factors using this structure:

Documents:
{text_documents}

Target Outcome: {target_outcome}

Causal Factors (provide {num_factors} factors):
1. [Factor name]: [Brief justification]
2. [Factor name]: [Brief justification]
...
""",
        }

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"causal dataset len: {len(self.dataframe)}")
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer

            def doc2len(doc) -> int:
                # Build causal prompt and measure length
                causal_prompt = self._build_causal_prompt_for_filtering(doc)
                return len(tokenizer.encode(causal_prompt, add_special_tokens=False))

            print(len(self.dataframe))
            self.dataframe = self.dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=1,
                desc=f"Filtering causal prompts longer than {self.max_prompt_length} tokens",
            )
            print(len(self.dataframe))
            print(f"filtered causal dataset len: {len(self.dataframe)}")

    def _build_causal_prompt_for_filtering(self, doc: dict) -> str:
        """Build a causal prompt for length filtering (simplified version)."""
        documents = doc.get(self.documents_key, [])
        target_outcome = doc.get(self.target_outcome_key, "outcome")

        # Sample documents for length estimation
        if len(documents) > self.sample_docs_per_input:
            sampled_docs = random.sample(documents, self.sample_docs_per_input)
        else:
            sampled_docs = documents

        return self._build_causal_prompt(sampled_docs, target_outcome)

    def _sample_text_documents(self, available_docs: List[str], metadata: dict = None) -> List[str]:
        """Sample text documents using configured strategy."""
        if len(available_docs) <= self.sample_docs_per_input:
            return available_docs

        if self.sampling_strategy == "random":
            return random.sample(available_docs, self.sample_docs_per_input)
        elif self.sampling_strategy == "stratified":
            return self._stratified_sample(available_docs, metadata)
        elif self.sampling_strategy == "temporal":
            return self._temporal_sample(available_docs, metadata)
        elif self.sampling_strategy == "similarity":
            return self._similarity_sample(available_docs, metadata)
        else:
            return random.sample(available_docs, self.sample_docs_per_input)

    def _stratified_sample(self, documents: List[str], metadata: dict) -> List[str]:
        """Sample documents with stratification by document type or metadata."""
        doc_metadata = metadata.get("doc_metadata", [])

        if not doc_metadata or len(doc_metadata) != len(documents):
            return random.sample(documents, min(self.sample_docs_per_input, len(documents)))

        # Group documents by metadata categories
        doc_groups = {}
        for doc, meta in zip(documents, doc_metadata):
            category = meta.get("category", "default") if isinstance(meta, dict) else "default"
            if category not in doc_groups:
                doc_groups[category] = []
            doc_groups[category].append(doc)

        # Sample proportionally from each group
        sampled_docs = []
        docs_per_group = max(1, self.sample_docs_per_input // len(doc_groups))

        for group_docs in doc_groups.values():
            sample_size = min(docs_per_group, len(group_docs))
            sampled_docs.extend(random.sample(group_docs, sample_size))
            if len(sampled_docs) >= self.sample_docs_per_input:
                break

        return sampled_docs[: self.sample_docs_per_input]

    def _temporal_sample(self, documents: List[str], metadata: dict) -> List[str]:
        """Sample documents with temporal ordering (recent first)."""
        timestamps = metadata.get("timestamps", [])

        if not timestamps or len(timestamps) != len(documents):
            return random.sample(documents, min(self.sample_docs_per_input, len(documents)))

        # Sort by timestamp (most recent first)
        doc_time_pairs = list(zip(documents, timestamps))
        doc_time_pairs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in doc_time_pairs[: self.sample_docs_per_input]]

    def _similarity_sample(self, documents: List[str], metadata: dict) -> List[str]:
        """Sample documents based on similarity to target outcome."""
        target_outcome = metadata.get("target_outcome", "")

        # Simple keyword-based similarity (can be enhanced with embeddings)
        target_keywords = target_outcome.lower().split()
        doc_scores = []

        for doc in documents:
            doc_lower = doc.lower()
            score = sum(1 for keyword in target_keywords if keyword in doc_lower)
            doc_scores.append((doc, score))

        # Sort by similarity score and take top documents
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in doc_scores[: self.sample_docs_per_input]]

    def _build_causal_prompt(self, documents: List[str], target: str) -> str:
        """Construct causal discovery prompt from sampled documents."""
        template = self.prompt_templates[self.prompt_template]

        # Format documents for display
        formatted_docs = "\n\n---\n\n".join([f"Document {i + 1}:\n{doc}" for i, doc in enumerate(documents)])

        return template.format(text_documents=formatted_docs, target_outcome=target, num_factors=self.max_factors)

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Get a causal factor discovery sample.

        Expected data format in parquet:
        {
            "documents": [list of text documents],
            "target_outcome": "outcome to predict",
            "doc_metadata": [optional metadata for each document],
            "timestamps": [optional timestamps for temporal sampling],
            "ground_truth_factors": [optional ground truth factors for evaluation],
            "instance_id": unique identifier,
            "data_source": source identifier
        }
        """
        row_dict: dict = dict(self.dataframe[item])

        # Extract documents and target outcome
        available_documents = row_dict.get(self.documents_key, [])
        target_outcome = row_dict.get(self.target_outcome_key, "outcome")

        # Handle case where prompt_key contains the single document
        if not available_documents and self.prompt_key in row_dict:
            available_documents = [row_dict[self.prompt_key]]

        # Sample documents using configured strategy
        metadata = {
            "doc_metadata": row_dict.get("doc_metadata", []),
            "timestamps": row_dict.get("timestamps", []),
            "target_outcome": target_outcome,
        }
        sampled_documents = self._sample_text_documents(available_documents, metadata)

        # Build causal discovery prompt
        causal_prompt = self._build_causal_prompt(sampled_documents, target_outcome)

        # Tokenize using VERL's standard approach
        model_inputs = self.tokenizer(causal_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        # Standard VERL format
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        # Raw prompt handling (following VERL pattern)
        raw_prompt_ids = self.tokenizer.encode(causal_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids

        if self.return_full_prompt:
            row_dict["full_prompts"] = causal_prompt

        # Add causal-specific metadata
        row_dict["sampled_documents"] = sampled_documents
        row_dict["target_outcome"] = target_outcome
        row_dict["ground_truth_factors"] = row_dict.get("ground_truth_factors", [])

        # Additional metadata for reward computation
        row_dict["causal_metadata"] = {
            "original_prompt": causal_prompt,
            "sampling_strategy": self.sampling_strategy,
            "num_sampled_docs": len(sampled_documents),
            "total_available_docs": len(available_documents),
            "prompt_template": self.prompt_template,
        }

        # Standard VERL fields
        index = row_dict.get("extra_info", {}).get("index", item)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if "dataframe" in state:
                del state["dataframe"]
            return state
        return self.__dict__.copy()
