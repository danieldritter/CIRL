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
from collections import defaultdict
from typing import List, Optional

import dowhy.gcm as gcm
import networkx as nx
import numpy as np
import pandas as pd
import torch
from causallearn.search.ConstraintBased.PC import pc
from omegaconf import DictConfig

from verl import DataProto
from verl.utils.causal.factor_parser import TextFactorParser
from verl.workers.reward_manager import register

logger = logging.getLogger(__name__)


@register("causal_discovery")
class CausalDiscoveryRewardManager:
    """
    Reward manager for causal factor discovery tasks.

    Combines multiple reward components:
    1. Format rewards from factor parser
    2. Causal discovery algorithm rewards
    3. Regression performance rewards
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 2,
        config: Optional[DictConfig] = None,
        compute_score=None,
        reward_fn_key=None,
    ):
        """
        Initialize the CausalDiscoveryRewardManager.

        Args:
            tokenizer: Tokenizer for decoding responses
            num_examine: Number of responses to print for debugging
            config: Configuration for causal discovery rewards
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.config = config or DictConfig({})

        # Reward weights
        self.format_reward_weight = self.config.get("format_reward_weight", 0.2)
        self.causal_reward_weight = self.config.get("causal_reward_weight", 0.5)
        self.refutation_reward_weight = self.config.get("refutation_reward_weight", 0.3)

        # Factor parser configuration
        parser_config = self.config.get("factor_parser", {})
        self.factor_parser = TextFactorParser(
            parser_type=parser_config.get("parser_type", "list"),
            max_factors=parser_config.get("max_factors", 10),
            validation_enabled=parser_config.get("validation_enabled", True),
        )

        # Causal discovery configuration
        self.causal_method = self.config.get("causal_method", "PC-bidirectional")
        self.min_factors_for_causal = self.config.get("min_factors_for_causal", 2)

        self.debug_count = 0

    def __call__(self, data: DataProto, return_dict=False):
        """
        Compute rewards for causal factor discovery responses.

        Args:
            data: DataProto containing responses and metadata
            return_dict: Whether to return detailed reward information

        Returns:
            Reward tensor or dict with reward components
        """
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]

            # Get parsed factors from NLI worker output instead of parsing ourselves
            factors = self._get_parsed_factors(data_item)
            format_score = self._compute_format_score(data_item, factors)

            if len(factors) < self.min_factors_for_causal:
                total_reward = 0.0
                response_length = data_item.batch["attention_mask"][data_item.batch["prompts"].shape[-1] :].sum()
                reward_tensor[i, response_length - 1] = total_reward
                reward_extra_info["format_reward"].append(format_score)
                reward_extra_info["cd_reward"].append(0.0)
                reward_extra_info["refutation_reward"].append(0.0)
                reward_extra_info["total_reward"].append(total_reward)
                reward_extra_info["extracted_factors"].append(factors)
                reward_extra_info["format_score"].append(format_score)
                continue

            # Compute reward components
            tabular_data = self._get_tabular_data(data_item)
            cd_reward, graph_structure = self._compute_cd_reward(factors, tabular_data)
            refutation_reward = self._compute_refutation_reward(tabular_data, graph_structure)

            # Combine rewards
            total_reward = (
                self.format_reward_weight * format_score
                + self.causal_reward_weight * cd_reward
                + self.refutation_reward_weight * refutation_reward
            )

            # Store reward at last valid position
            response_length = data_item.batch["attention_mask"][data_item.batch["prompts"].shape[-1] :].sum()
            reward_tensor[i, response_length - 1] = total_reward

            # Store detailed information
            reward_extra_info["format_reward"].append(format_score)
            reward_extra_info["cd_reward"].append(cd_reward)
            reward_extra_info["refutation_reward"].append(refutation_reward)
            reward_extra_info["total_reward"].append(total_reward)
            reward_extra_info["extracted_factors"].append(factors)
            reward_extra_info["format_score"].append(format_score)

            # Debug output
            if self.debug_count < self.num_examine:
                self._print_debug_info(
                    data_item,
                    self._extract_response_text(data_item),
                    factors,
                    format_score,
                    cd_reward,
                    refutation_reward,
                    total_reward,
                )
                self.debug_count += 1

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor

    def _extract_response_text(self, data_item) -> str:
        """Extract decoded response text from data item."""
        prompt_length = data_item.batch["prompts"].shape[-1]
        response_ids = data_item.batch["responses"]

        # Get valid response length
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # Decode response
        response_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        return response_text

    def _get_parsed_factors(self, data_item) -> List[str]:
        """Get factors that were parsed by the NLI worker."""
        parsing_info = data_item.non_tensor_batch.get("parsing_info", {})
        factors = parsing_info.get("parsed_factors", [])

        if not factors:
            # Fallback: try to get factor names from tabular data
            factors = data_item.non_tensor_batch.get("factor_names", [])

        if not factors:
            raise ValueError(
                "No parsed factors found in data item. Ensure NLI processing is done and "
                "factors are properly parsed from LM responses."
            )

        return factors

    def _compute_format_score(self, data_item, factors: List[str]) -> float:
        """Compute format score based on parsing success."""
        parsing_info = data_item.non_tensor_batch.get("parsing_info", {})

        # If we used ground truth factors, give perfect format score
        if parsing_info.get("used_ground_truth", False):
            return 1.0

        # Otherwise, score based on number of factors extracted
        num_factors = len(factors)
        if num_factors == 0:
            return 0.0
        elif num_factors < self.min_factors_for_causal:
            return 0.5  # Partial credit for some factors
        else:
            return 1.0  # Full credit for sufficient factors

    def _compute_cd_reward(self, factors: List[str], tabular_data: pd.DataFrame) -> tuple[float, pd.DataFrame]:
        """Compute reward based on causal discovery quality."""
        # Get tabular data from NLI processing
        if tabular_data is None:
            raise ValueError(
                "Tabular data not found for causal discovery. Ensure NLI processing is done before this step."
            )
        # Compute causal strength using specified method
        if self.causal_method == "PC-bidirectional":
            causal_score, graph_structure = self._compute_pc_bidirectional(tabular_data)
        else:
            raise ValueError(f"Unsupported causal method: {self.causal_method}")

        return causal_score, graph_structure

    def _compute_refutation_reward(self, tabular_data: pd.DataFrame, graph_structure: pd.DataFrame) -> float:
        edge_list = []
        for i in range(tabular_data.shape[1]):
            for j in range(i + 1, tabular_data.shape[1]):
                if graph_structure.iloc[i, j] == 1:
                    edge_list.append((tabular_data.columns[i], tabular_data.columns[j]))
                if graph_structure.iloc[j, i] == 1:
                    edge_list.append((tabular_data.columns[j], tabular_data.columns[i]))
        edge_graph = nx.DiGraph(edge_list)
        scm = gcm.StructuralCausalModel(edge_graph)
        gcm.auto.assign_causal_mechanisms(scm, tabular_data)
        gcm.fit(scm, tabular_data)
        summary_evaluation = gcm.evaluate_causal_model(scm, tabular_data, compare_mechanism_baselines=True)
        return summary_evaluation.mechanism_performances.crps  # TODO: decide on final metric here

    def _get_tabular_data(self, data_item) -> pd.DataFrame:
        """Extract tabular data from NLI processing results."""
        # Get tensor data from batch (from NLI worker)
        tabular_tensor = data_item.batch.get("tabular_data", None)
        factor_names = data_item.non_tensor_batch.get("factor_names", None)

        if tabular_tensor is not None and factor_names is not None:
            # Convert tensor to DataFrame
            tabular_array = tabular_tensor.cpu().numpy()
            # Reshape to (num_documents, num_factors)
            if len(tabular_array.shape) == 1:
                num_factors = len(factor_names)
                num_docs = len(tabular_array) // num_factors
                tabular_array = tabular_array.reshape(num_docs, num_factors)

            df = pd.DataFrame(tabular_array, columns=factor_names)
            return df
        else:
            raise ValueError(
                "Tabular data not found in data item. Ensure NLI processing is done and "
                "'tabular_data' is included in the batch."
            )

    def _get_target_values(self, data_item) -> Optional[np.ndarray]:
        """Extract target values for regression."""
        # Get target values from metadata
        target_values = data_item.non_tensor_batch.get("target_values", None)

        if target_values is None:
            raise ValueError(
                "Target values not found in data. Please ensure your dataset includes "
                "'target_values' in the non_tensor_batch for regression analysis."
            )

        return np.array(target_values)

    def _compute_pc_bidirectional(self, tabular_data: pd.DataFrame) -> tuple[float, np.ndarray]:
        data = tabular_data.to_numpy()
        factors = tabular_data.columns.tolist()
        cg = pc(data)  # using default parameters
        # reward score is number of bidirectional edges (representing possible missed confounders)
        num_bidirectional_edges = 0
        for i in range(data.shape[1]):
            for j in range(i + 1, data.shape[1]):
                if cg.G.graph[i, j] == cg.G.graph[j, i] == 1:
                    num_bidirectional_edges += 1
        # Normalize by number of possible edges
        num_possible_edges = len(factors) * (len(factors) - 1) / 2
        # convert graph structure to dataframe
        graph_structure = pd.DataFrame(data=cg.G.graph, index=factors, columns=factors)
        return num_bidirectional_edges / num_possible_edges, graph_structure

    def _print_debug_info(
        self,
        data_item,
        response_text: str,
        factors: List[str],
        format_reward: float,
        causal_reward: float,
        refutation_reward: float,
        total_reward: float,
    ):
        """Print debug information for analysis."""
        print("\n=== Causal Discovery Debug Info ===")
        print(f"Response: {response_text[:200]}...")
        print(f"Parsed Factors: {factors}")
        print(f"Format Reward: {format_reward:.4f}")
        print(f"Causal Reward: {causal_reward:.4f}")
        print(f"Refutation Reward: {refutation_reward:.4f}")
        print(f"Total Reward: {total_reward:.4f}")

        # Print tabular data info if available
        try:
            tabular_data = self._get_tabular_data(data_item)
            print(f"Tabular Data Shape: {tabular_data.shape}")
            print(f"Tabular Data Columns: {list(tabular_data.columns)}")
        except Exception as e:
            print(f"Could not display tabular data: {e}")

        # Print parsing info
        parsing_info = data_item.non_tensor_batch.get("parsing_info", {})
        print(f"Parsing Info: {parsing_info}")

        print("=" * 40)
