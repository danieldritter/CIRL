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

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.fsdp_workers import create_device_mesh, get_sharding_strategy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class NLITabularProcessor(Worker):
    """
    Distributed NLI worker for tabular data conversion.
    Follows the same pattern as ActorRolloutRefWorker and CriticWorker.
    """

    def __init__(self, config: DictConfig):
        Worker.__init__(self)
        self.config = config

        # Initialize distributed setup (same as other workers)
        if not dist.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            dist.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # Build device mesh for distributed processing
        world_size = dist.get_world_size()
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=config.nli.fsdp_config.fsdp_size)

        # NLI-specific configuration
        self.model_name = config.nli.model_name
        self.hypothesis_template = config.nli.hypothesis_template
        self.max_nli_length = config.nli.max_length
        self.batch_size = config.nli.batch_size

        # Initialize cache if specified
        cache_dir = config.nli.get("cache_dir", None)
        self.cache = self._init_cache(cache_dir) if cache_dir else None

    def _init_cache(self, cache_dir: str) -> Dict:
        """Initialize cache for NLI results."""
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"nli_cache_rank_{self.rank}.pkl")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cache = pickle.load(f)
                logger.info(f"Loaded NLI cache with {len(cache)} entries")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        return {}

    def _save_cache(self):
        """Save cache to disk."""
        if self.cache is None:
            return

        cache_dir = self.config.nli.get("cache_dir", None)
        if cache_dir:
            cache_file = os.path.join(cache_dir, f"nli_cache_rank_{self.rank}.pkl")
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(self.cache, f)
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize NLI model with FSDP wrapping."""
        log_gpu_memory_usage("Before NLI model init", logger=logger)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map=None
        )

        # FSDP configuration (same pattern as other workers)
        mixed_precision_config = self.config.nli.fsdp_config.get("mixed_precision", {})
        param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
        reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
        buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        # Wrap with FSDP for distributed inference
        self.model = FSDP(
            self.model,
            device_id=get_device_id(),
            sharding_strategy=get_sharding_strategy(self.device_mesh),
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            use_orig_params=self.config.nli.fsdp_config.get("use_orig_params", False),
        )

        self.model.eval()
        log_gpu_memory_usage("After NLI model FSDP init", logger=logger)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def convert_to_tabular(self, data: DataProto):
        """Convert text documents to tabular format using distributed NLI with LM-generated factors."""
        # Move data to device
        data = data.to(get_device_id())

        # Extract input data
        documents = data.non_tensor_batch["documents"]
        
        # Check if we have LM responses to parse factors from, or ground truth factors
        lm_responses = data.non_tensor_batch.get("lm_responses", None)
        if lm_responses is not None:
            # Parse factors from LM responses using the factor parser
            from verl.utils.causal.factor_parser import TextFactorParser
            
            # Initialize factor parser with default settings
            factor_parser = TextFactorParser(parser_type="list", max_factors=10, validation_enabled=True)
            
            # Parse factors from all LM responses and combine them
            all_factors = []
            for response in lm_responses:
                factors, _ = factor_parser.parse_with_format_score(response)
                all_factors.extend(factors)
            
            # Remove duplicates while preserving order
            factors = list(dict.fromkeys(all_factors))
            
            # Store parsing info for later use by reward manager
            parsing_info = {
                "parsed_factors": factors,
                "num_responses_parsed": len(lm_responses)
            }
        else:
            # Fallback to ground truth factors if no LM responses provided
            factors = data.non_tensor_batch["factors"]
            parsing_info = {
                "parsed_factors": factors,
                "num_responses_parsed": 0,
                "used_ground_truth": True
            }

        # Create hypothesis statements for each factor
        hypotheses = [self.hypothesis_template.format(factor=factor) for factor in factors]

        # Prepare NLI pairs
        nli_pairs = []
        for doc in documents:
            for hypothesis in hypotheses:
                nli_pairs.append((doc, hypothesis))

        # Check cache first
        if self.cache is not None:
            cached_results, uncached_pairs = self._check_cache_batch(nli_pairs)
            if not uncached_pairs:
                # All results cached - reshape and return
                predictions = self._reshape_cached_results(cached_results, len(documents), len(factors))
            else:
                # Process uncached pairs
                uncached_predictions = self._process_nli_batch(uncached_pairs)
                # Merge with cached results
                predictions = self._merge_cached_and_new_results(
                    cached_results, uncached_predictions, uncached_pairs, nli_pairs
                )
                # Cache new results
                self._cache_batch_results(uncached_pairs, uncached_predictions)
        else:
            # No cache - process all pairs
            predictions = self._process_nli_batch(nli_pairs)

        # Convert to tabular tensor format
        tabular_tensor = self._predictions_to_tensor(predictions, documents, factors)

        # Return as DataProto with parsed factors information
        output = DataProto.from_dict(
            tensors={"tabular_data": tabular_tensor},
            meta_info={
                "num_documents": len(documents),
                "num_factors": len(factors),
            },
            non_tensors={
                "factor_names": factors,
                "document_count": len(documents),
                "parsing_info": parsing_info,
            },
        )

        return output.to("cpu")

    def _process_nli_batch(self, nli_pairs: List[Tuple[str, str]]) -> torch.Tensor:
        """Process NLI pairs through the model."""
        all_predictions = []

        for i in range(0, len(nli_pairs), self.batch_size):
            batch_pairs = nli_pairs[i : i + self.batch_size]

            # Tokenize batch
            tokenized = self.tokenizer(
                batch_pairs, truncation=True, padding=True, return_tensors="pt", max_length=self.max_nli_length
            )

            # Move to device
            tokenized = {k: v.to(get_device_id()) for k, v in tokenized.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**tokenized)
                logits = outputs.logits
                predicted_classes = torch.argmax(logits, dim=-1)
                all_predictions.append(predicted_classes)

        return torch.cat(all_predictions, dim=0)

    def _check_cache_batch(self, nli_pairs: List[Tuple[str, str]]) -> Tuple[List[Optional[int]], List[Tuple[str, str]]]:
        """Check cache for existing results."""
        cached_results = []
        uncached_pairs = []

        for pair in nli_pairs:
            pair_key = f"{pair[0]}||{pair[1]}"  # Simple key generation
            if self.cache is not None and pair_key in self.cache:
                cached_results.append(self.cache[pair_key])
            else:
                cached_results.append(None)
                uncached_pairs.append(pair)

        return cached_results, uncached_pairs

    def _cache_batch_results(self, nli_pairs: List[Tuple[str, str]], predictions: torch.Tensor):
        """Cache batch results."""
        if self.cache is None:
            return
            
        pred_list = predictions.cpu().numpy().tolist()

        for pair, pred in zip(nli_pairs, pred_list):
            pair_key = f"{pair[0]}||{pair[1]}"
            self.cache[pair_key] = int(pred)  # Ensure integer storage

        # Periodically save cache
        if len(self.cache) % 1000 == 0:
            self._save_cache()

    def _reshape_cached_results(
        self, cached_results: List[Optional[int]], n_docs: int, n_factors: int
    ) -> torch.Tensor:
        """Reshape cached results to tensor format."""
        predictions = []
        for result in cached_results:
            if result is not None:
                predictions.append(result)
            else:
                raise ValueError("Cached result is None, expected an integer value.")
        return torch.tensor(predictions, dtype=torch.long)

    def _merge_cached_and_new_results(
        self,
        cached_results: List[Optional[int]],
        new_predictions: torch.Tensor,
        uncached_pairs: List[Tuple[str, str]],
        all_pairs: List[Tuple[str, str]],
    ) -> torch.Tensor:
        """Merge cached and newly computed results."""
        new_pred_iter = iter(new_predictions.cpu().numpy().tolist())
        uncached_set = set(uncached_pairs)

        final_predictions = []
        for i, pair in enumerate(all_pairs):
            if cached_results[i] is not None:
                final_predictions.append(cached_results[i])
            elif pair in uncached_set:
                final_predictions.append(int(next(new_pred_iter)))
            else:
                raise ValueError(f"Pair {pair} not found in uncached pairs or cache")
        return torch.tensor(final_predictions, dtype=torch.long)

    def _predictions_to_tensor(
        self, predictions: torch.Tensor, documents: List[str], factors: List[str]
    ) -> torch.Tensor:
        """Convert NLI predictions to tensor format (documents x factors)."""
        n_docs = len(documents)
        n_factors = len(factors)

        # Reshape predictions to matrix format
        pred_matrix = predictions.view(n_docs, n_factors)

        return pred_matrix

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_cache(self):
        """Save cache to disk."""
        self._save_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def clear_cache(self):
        """Clear the cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("NLI cache cleared")
