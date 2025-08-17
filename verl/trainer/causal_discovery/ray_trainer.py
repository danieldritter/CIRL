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

"""
Ray-based Causal Discovery Trainer

Extends the PPO trainer to include NLI-based tabular conversion for causal factor discovery.
"""

from typing import Callable, Dict, Optional

from omegaconf import DictConfig
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.trainer.ppo.ray_trainer import ResourcePoolManager as BaseResourcePoolManager


class ResourcePoolManager(BaseResourcePoolManager):
    """Extended resource pool manager that includes NLI processor resources."""

    def __init__(self, resource_pool_spec: Dict, mapping: Dict):
        super().__init__(resource_pool_spec, mapping)
        # Ensure NLI processor is included in the mapping
        if Role.NLIProcessor not in mapping:
            raise ValueError(f"Resource pool mapping must include {Role.NLIProcessor} role.")


class RayCausalDiscoveryTrainer(RayPPOTrainer):
    """
    Ray-based trainer for causal factor discovery.

    Extends PPO trainer to include NLI-based tabular conversion and
    specialized reward computation for causal factor quality.
    """

    def __init__(
        self,
        config: DictConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        role_worker_mapping: Dict,
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup,
        reward_fn: Callable,
        val_reward_fn: Callable,
        train_dataset: Dataset,
        val_dataset: Dataset,
        collate_fn: Callable,
        train_sampler: Sampler,
        device_name: str = "cuda",
    ):
        # Initialize base PPO trainer with standard reward functions
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )

        # Store causal discovery specific configuration
        self.nli_config = config.get("nli", {})
        self.causal_reward_config = config.get("causal_discovery_reward", {})

        # Initialize NLI processor worker group
        self.nli_processor_wg = None

    def init_workers(self):
        """Initialize all workers including NLI processor."""
        # Initialize base workers (actor, critic, etc.)
        super().init_workers()

        # Initialize NLI processor worker group
        if Role.NLIProcessor in self.role_worker_mapping:
            nli_worker_cls = self.role_worker_mapping[Role.NLIProcessor]
            nli_resource_pool = self.resource_pool_manager.get_pool(Role.NLIProcessor)

            self.nli_processor_wg = self.ray_worker_group_cls(
                resource_pool=nli_resource_pool,
                ray_worker_cls=nli_worker_cls,
                ray_actor_kwargs={},
                max_restarts=0,
            )

            # Initialize NLI model on workers
            self.nli_processor_wg.init_model.remote(self.nli_config)

            print(f"Initialized NLI processor with {len(self.nli_processor_wg.workers)} workers")

    def _process_batch_for_causal_discovery(self, batch_data: DataProto) -> DataProto:
        """
        Process batch data through NLI tabular conversion using LM-generated factors.

        Args:
            batch_data: Raw batch data from dataset containing LM responses

        Returns:
            Enhanced batch data with tabular representations
        """
        if self.nli_processor_wg is None:
            raise RuntimeError("NLI processor worker group is not initialized.")

        # Extract documents and LM responses from batch
        documents = []
        lm_responses = []

        for i in range(len(batch_data)):
            data_item = batch_data[i]

            # Extract sampled documents from the non-tensor batch
            item_docs = data_item.non_tensor_batch.get("sampled_documents", [])
            documents.extend(item_docs)

            # Extract LM response text for factor parsing
            response_text = self._extract_response_text(data_item)
            lm_responses.append(response_text)

        # Create NLI processing batch with LM responses for factor parsing
        nli_batch = DataProto.from_dict(
            non_tensors={
                "documents": documents,
                "lm_responses": lm_responses,  # Pass LM responses instead of ground truth factors
            },
        )

        # Process through NLI workers (they will parse factors from LM responses)
        tabular_results = self.nli_processor_wg.convert_to_tabular.remote(nli_batch)
        tabular_data = tabular_results.get()

        # Merge tabular data back into original batch
        return batch_data.union(tabular_data)

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

    def fit(self):
        """
        Override fit to add NLI processing to the training loop.

        We need to hook into the reward computation flow to ensure
        NLI processing happens before the CausalDiscoveryRewardManager is called.
        """
        # Store original compute_reward function
        from verl.trainer.ppo.reward import compute_reward as original_compute_reward

        def enhanced_compute_reward(data, reward_fn):
            # Process data through NLI conversion before reward computation
            enhanced_data = self._process_batch_for_causal_discovery(data)
            # Call original compute_reward with enhanced data
            return original_compute_reward(enhanced_data, reward_fn)

        # Temporarily replace the compute_reward function
        import verl.trainer.ppo.reward

        original_fn = verl.trainer.ppo.reward.compute_reward
        verl.trainer.ppo.reward.compute_reward = enhanced_compute_reward

        try:
            # Call parent's fit method with enhanced compute_reward
            return super().fit()
        finally:
            # Restore original compute_reward function
            verl.trainer.ppo.reward.compute_reward = original_fn

    def _validate(self):
        """Override validation to add NLI processing."""
        # Store original validation reward function
        original_val_reward_fn = self.val_reward_fn

        # Create enhanced validation reward function
        def enhanced_val_reward_fn(data, **kwargs):
            enhanced_data = self._process_batch_for_causal_discovery(data)
            return original_val_reward_fn(enhanced_data, **kwargs)

        # Temporarily override validation reward function
        self.val_reward_fn = enhanced_val_reward_fn

        try:
            # Call parent's validation method
            return super()._validate()
        finally:
            # Restore original validation reward function
            self.val_reward_fn = original_val_reward_fn

    def save_checkpoint(self, epoch: int, step: int, is_final: bool = False):
        """
        Save checkpoint including NLI processor state.

        Extends base checkpoint saving to include NLI processor.
        """
        # Save base checkpoint
        super().save_checkpoint(epoch, step, is_final)

        # Save NLI processor cache if available
        if self.nli_processor_wg is not None:
            self.nli_processor_wg.save_cache.remote()
            print(f"✓ Saved NLI processor cache at epoch {epoch}, step {step}")

    def cleanup(self):
        """Clean up resources including NLI processor."""
        # Clean up NLI processor
        if self.nli_processor_wg is not None:
            self.nli_processor_wg.save_cache.remote()  # Final cache save
            # Note: Ray will handle worker cleanup automatically

        # Clean up base trainer
        super().cleanup()
