#!/usr/bin/env python3
"""
Generate synthetic Apple Gastronome dataset for causal discovery training.

Based on the Apple_Gastronome_AG7_v20240513.ipynb notebook.
Creates a dataset compatible with the CausalTextDataset class.
"""

import argparse
import logging
import os
import random
from typing import Dict, List

import numpy as np

from datasets import Dataset as HFDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AppleGastronomeGenerator:
    """Generate synthetic apple gastronome data with known causal structure."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        # Apple attribute mappings from the notebook
        self.apple_size = {1: "large size", -1: "small size"}
        self.apple_smell = {1: "strong aroma", -1: "musty or rotten"}
        self.apple_taste = {1: "more sweet than sour", -1: "more sour than sweet"}
        self.apple_color = {1: "vibrant and uniformly colored", -1: "unevenly colored"}
        self.apple_nutrition = {
            1: "a high content of essential nutrients, including dietary fiber, vitamin C, etc.",
            -1: "a relatively low content of essential nutrients",
        }
        self.apple_juiciness = {1: "abundant and refreshing moisture", -1: "dry and lacking moisture"}
        self.apple_recmd = {
            1: "has significant market potential and deserves wider recognition",
            -1: "might not bring the expected returns and could even lead to losses",
        }

    def _map_0_1(self, x):
        """Map probability to {-1, 1}."""
        return 1 if x == 1 else -1

    def _generate_apple_features(self):
        """Generate one apple's features following the causal structure from notebook."""
        # Independent factors
        size = self._map_0_1(np.random.uniform() < 0.5) * (np.random.uniform() < 0.9) * 1
        nutrition = self._map_0_1(np.random.uniform() < 0.5) * (np.random.uniform() < 0.9) * 1
        smell = self._map_0_1(np.random.uniform() < 0.5) * (np.random.uniform() < 0.9) * 1
        taste = self._map_0_1(np.random.uniform() < 0.5) * (np.random.uniform() < 0.9) * 1

        # Dependent on taste
        juiciness = taste * self._map_0_1(np.random.uniform() < 0.7) * (np.random.uniform() < 0.9) * 1

        # Score depends on taste, size, smell
        score = taste + size + smell + self._map_0_1(np.random.uniform() < 0.5) * (np.random.uniform() < 0.9) * 1

        # Recommendation depends on score and nutrition
        recmd = 0
        if np.abs(score) > 1:
            recmd = np.sign(score)
        else:
            recmd = np.sign(nutrition + score)
        recmd *= (np.random.uniform() < 0.9) * 1

        return {
            "score": score,
            "nutrition": nutrition,
            "size": size,
            "smell": smell,
            "taste": taste,
            "juiciness": juiciness,
            "recmd": recmd,
        }

    def _generate_apple_review(self, features: Dict) -> str:
        """Generate a gastronome review for an apple based on its features."""
        # Extract attribute descriptions that are present (non-zero)
        descriptions = []

        if features["nutrition"] in self.apple_nutrition:
            descriptions.append(self.apple_nutrition[features["nutrition"]])
        if features["size"] in self.apple_size:
            descriptions.append(self.apple_size[features["size"]])
        if features["smell"] in self.apple_smell:
            descriptions.append(self.apple_smell[features["smell"]])
        if features["taste"] in self.apple_taste:
            descriptions.append(self.apple_taste[features["taste"]])
        if features["juiciness"] in self.apple_juiciness:
            descriptions.append(self.apple_juiciness[features["juiciness"]])

        # Generate review using templates similar to the notebook
        templates = [
            "This apple exhibits {features}. Based on these characteristics, it {recommendation}.",
            "Upon evaluation, this apple demonstrates {features}, leading to the conclusion that it {recommendation}.",
            "The gastronome assessment reveals that this apple has {features}. Therefore, it {recommendation}.",
            "Professional analysis shows this apple possesses {features}, indicating it {recommendation}.",
            "Expert evaluation finds this apple features {features}, which means it {recommendation}.",
            "[Review Begin] This apple shows {features}, and consequently {recommendation}.",
        ]

        template = random.choice(templates)

        # Join features with proper grammar
        if len(descriptions) > 1:
            features_text = ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"
        elif len(descriptions) == 1:
            features_text = descriptions[0]
        else:
            features_text = "mixed characteristics"

        recommendation = self.apple_recmd.get(features["recmd"], "shows uncertain market potential")

        return template.format(features=features_text, recommendation=recommendation)

    def generate_dataset(self, num_samples: int = 1000, num_docs_per_sample: int = 5) -> List[Dict]:
        """Generate the complete synthetic dataset."""
        logger.info(f"Generating {num_samples} causal discovery samples...")

        # Generate pool of apple reviews
        apple_pool = []
        for i in range(num_samples * 3):  # Generate more apples than needed for sampling
            features = self._generate_apple_features()
            review = self._generate_apple_review(features)
            apple_pool.append({"apple_id": f"apple_{i:04d}", "features": features, "review": review})

        logger.info(f"Generated {len(apple_pool)} apple reviews in pool")

        # Create causal discovery samples
        dataset = []
        ground_truth_factors = ["size", "smell", "taste", "nutrition", "juiciness"]

        for sample_id in range(num_samples):
            # Sample documents (apple reviews) for this causal discovery task
            sampled_apples = random.sample(apple_pool, min(num_docs_per_sample, len(apple_pool)))
            documents = [apple["review"] for apple in sampled_apples]

            # Create dataset entry in format expected by CausalTextDataset
            sample = {
                "documents": documents,
                "target_outcome": "apple_recommendation",
                "ground_truth_factors": ground_truth_factors,
                "instance_id": f"causal_sample_{sample_id:04d}",
                "data_source": "synthetic_apple_gastronome",
                # Additional metadata for analysis
                "doc_metadata": [
                    {"apple_id": apple["apple_id"], "category": "apple_review", "features": apple["features"]}
                    for apple in sampled_apples
                ],
                "timestamps": list(range(len(documents))),  # Sequential timestamps
                # Causal structure information
                "causal_info": {
                    "true_causal_factors": ground_truth_factors,
                    "causal_relationships": {
                        "taste -> juiciness": "Taste influences juiciness perception",
                        "size, smell, taste -> score": "These factors combine to create overall score",
                        "nutrition, score -> recommendation": "Recommendation depends on nutrition and overall score",
                    },
                    "factor_types": {
                        "size": "physical_attribute",
                        "smell": "sensory_attribute",
                        "taste": "sensory_attribute",
                        "nutrition": "health_attribute",
                        "juiciness": "texture_attribute",
                    },
                },
            }

            dataset.append(sample)

        logger.info(f"Generated {len(dataset)} causal discovery samples")
        return dataset


def main():
    """Main function to generate and save the Apple Gastronome dataset."""
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # Initialize generator
    generator = AppleGastronomeGenerator(seed=args.seed)

    # Generate train and validation splits
    train_data = generator.generate_dataset(num_samples=args.train_samples, num_docs_per_sample=args.docs_per_sample)

    val_data = generator.generate_dataset(num_samples=args.val_samples, num_docs_per_sample=args.docs_per_sample)

    # Save as parquet files
    splits = {"train": train_data, "val": val_data}

    saved_files = []
    for split_name, split_data in splits.items():
        # Convert to HuggingFace dataset and save
        hf_dataset = HFDataset.from_list(split_data)
        output_path = os.path.join(local_save_dir, f"{split_name}.parquet")
        hf_dataset.to_parquet(output_path)

        logger.info(f"Saved {len(split_data)} {split_name} samples to {output_path}")
        saved_files.append(output_path)

    # Print sample for verification
    logger.info("Sample data entry:")
    sample_entry = train_data[0]
    print(f"Instance ID: {sample_entry['instance_id']}")
    print(f"Target outcome: {sample_entry['target_outcome']}")
    print(f"Ground truth factors: {sample_entry['ground_truth_factors']}")
    print(f"Number of documents: {len(sample_entry['documents'])}")
    print(f"First document: {sample_entry['documents'][0][:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic Apple Gastronome dataset for causal discovery training."
    )
    parser.add_argument(
        "--local_dir",
        default="~/data/apple_gastronome_synthetic",
        help="Local directory to save the generated Parquet files.",
    )
    parser.add_argument("--train_samples", type=int, default=800, help="Number of training samples to generate.")
    parser.add_argument("--val_samples", type=int, default=200, help="Number of validation samples to generate.")
    parser.add_argument(
        "--docs_per_sample", type=int, default=5, help="Number of apple review documents per causal discovery sample."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    main()
