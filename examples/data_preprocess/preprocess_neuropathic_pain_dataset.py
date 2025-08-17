#!/usr/bin/env python3
"""
Preprocess neuropathic pain dataset for causal discovery training.

Converts the neuropathic pain CSV data into a format compatible with CausalTextDataset.
Target outcome: R shoulder impingement
Documents: Patient notes containing symptom descriptions
"""

import argparse
import logging
import os
import random
from typing import Dict, List

import pandas as pd

from datasets import Dataset as HFDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NeuropathicPainProcessor:
    """Process neuropathic pain data for causal discovery."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def load_and_clean_data(self, csv_path: str) -> pd.DataFrame:
        """Load and clean the neuropathic pain CSV data."""
        logger.info(f"Loading data from {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV")

        # Clean and validate data
        df = df.dropna(subset=["Notes", "R shoulder impingement"])

        # Convert target to binary (0/1) if needed
        df["R shoulder impingement"] = df["R shoulder impingement"].astype(int)

        # Clean notes text
        df["Notes"] = df["Notes"].str.strip()

        logger.info(f"After cleaning: {len(df)} valid patient records")
        logger.info(f"Target distribution: {df['R shoulder impingement'].value_counts().to_dict()}")

        return df

    def create_causal_discovery_samples(self, df: pd.DataFrame, docs_per_sample: int = 5) -> List[Dict]:
        """Create causal discovery samples from patient data."""
        logger.info("Creating causal discovery samples...")

        # Use whole patient notes as documents
        all_documents = []
        document_metadata = []

        for idx, row in df.iterrows():
            # Use the entire note as a single document
            note_text = str(row["Notes"]).strip()

            all_documents.append(note_text)
            document_metadata.append(
                {
                    "patient_idx": idx,
                    "target_value": row["R shoulder impingement"],
                    "category": "patient_notes",
                }
            )

        logger.info(f"Using {len(all_documents)} patient notes from {len(df)} patients")

        # Create causal discovery samples by sampling documents
        samples = []
        num_samples = max(100, len(df) // 5)  # Create reasonable number of samples

        # Ground truth factors for neuropathic pain / shoulder impingement
        ground_truth_factors = [
            "R C4 Radiculopathy",
            "R C5 Radiculopathy",
            "R C6 Radiculopathy",
            "DLI C3-C4",
            "DLI C4-C5",
            "DLI C5-C6",
        ]

        for sample_idx in range(num_samples):
            # Sample documents for this causal discovery task
            sampled_indices = random.sample(range(len(all_documents)), min(docs_per_sample, len(all_documents)))

            sampled_docs = [all_documents[i] for i in sampled_indices]
            sampled_metadata = [document_metadata[i] for i in sampled_indices]

            sample = {
                "documents": sampled_docs,
                "target_outcome": "right_shoulder_impingement",
                "ground_truth_factors": ground_truth_factors,
                "instance_id": f"neuro_sample_{sample_idx:04d}",
                "data_source": "neuropathic_pain_dataset",
                # Metadata for analysis
                "doc_metadata": sampled_metadata,
                "timestamps": list(range(len(sampled_docs))),
                # Domain-specific information
                "clinical_info": {
                    "condition": "neuropathic_pain",
                    "target_diagnosis": "R_shoulder_impingement",
                    "patient_population": "neuropathic_pain_patients",
                    "data_type": "clinical_notes",
                },
                # Causal discovery context
                "causal_info": {
                    "domain": "medical_diagnosis",
                    "task_type": "diagnostic_factor_discovery",
                    "true_causal_factors": ground_truth_factors,
                    "factor_types": {
                        "R C4 Radiculopathy": "clinical_finding",
                        "R C5 Radiculopathy": "clinical_finding",
                        "R C6 Radiculopathy": "clinical_finding",
                        "DLI C3-C4": "imaging_finding",
                        "DLI C4-C5": "imaging_finding",
                        "DLI C5-C6": "imaging_finding",
                    },
                },
            }

            samples.append(sample)

        logger.info(f"Created {len(samples)} causal discovery samples")
        return samples

    def create_train_val_split(self, samples: List[Dict], val_ratio: float = 0.2) -> tuple:
        """Split samples into train and validation sets."""
        random.shuffle(samples)
        split_idx = int(len(samples) * (1 - val_ratio))

        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        logger.info(f"Split into {len(train_samples)} train and {len(val_samples)} validation samples")
        return train_samples, val_samples


def main():
    """Main function to process neuropathic pain dataset."""
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # Initialize processor
    processor = NeuropathicPainProcessor(seed=args.seed)

    # Load and clean data
    df = processor.load_and_clean_data(args.csv_path)

    # Create causal discovery samples
    samples = processor.create_causal_discovery_samples(df, docs_per_sample=args.docs_per_sample)

    # Split into train/val
    train_samples, val_samples = processor.create_train_val_split(samples, val_ratio=args.val_ratio)

    # Save as parquet files
    splits = {"train": train_samples, "val": val_samples}

    saved_files = []
    for split_name, split_data in splits.items():
        # Convert to HuggingFace dataset and save
        hf_dataset = HFDataset.from_list(split_data)
        output_path = os.path.join(local_save_dir, f"{split_name}.parquet")
        hf_dataset.to_parquet(output_path)

        logger.info(f"Saved {len(split_data)} {split_name} samples to {output_path}")
        saved_files.append(output_path)

    # Save dataset info
    info_path = os.path.join(local_save_dir, "dataset_info.txt")
    with open(info_path, "w") as f:
        f.write("Neuropathic Pain Causal Discovery Dataset\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Source CSV: {args.csv_path}\n")
        f.write(f"Processed with seed: {args.seed}\n")
        f.write(f"Total original patients: {len(df)}\n")
        f.write(f"Train samples: {len(train_samples)}\n")
        f.write(f"Validation samples: {len(val_samples)}\n")
        f.write(f"Documents per sample: {args.docs_per_sample}\n\n")
        f.write("Ground Truth Causal Factors:\n")
        f.write("- pain_location: Location and distribution of pain\n")
        f.write("- range_of_motion: Joint mobility and movement limitations\n")
        f.write("- nerve_involvement: Evidence of nerve compression/irritation\n")
        f.write("- muscle_weakness: Muscular strength deficits\n")
        f.write("- physical_examination_findings: Clinical examination results\n")
        f.write("- radiculopathy: Nerve root pathology patterns\n")
        f.write("- disc_pathology: Intervertebral disc abnormalities\n\n")
        f.write("Target Outcome: right_shoulder_impingement\n\n")
        f.write("Original target distribution:\n")
        for value, count in df["R shoulder impingement"].value_counts().items():
            f.write(f"  {value}: {count} patients\n")

    logger.info(f"Dataset info saved to {info_path}")
    logger.info(f"Successfully processed neuropathic pain dataset with {len(saved_files)} files in {local_save_dir}")

    # Print sample for verification
    logger.info("Sample data entry:")
    sample_entry = train_samples[0]
    print(sample_entry)
    print(train_samples[5])
    print(f"Instance ID: {sample_entry['instance_id']}")
    print(f"Target outcome: {sample_entry['target_outcome']}")
    print(f"Ground truth factors: {sample_entry['ground_truth_factors']}")
    print(f"Number of documents: {len(sample_entry['documents'])}")
    print(f"First document preview: {sample_entry['documents'][0][:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess neuropathic pain dataset for causal discovery training.")
    parser.add_argument(
        "--csv_path",
        default="./scripts/neuropathic/neuro_R_shoulder_impingement.csv",
        help="Path to the neuropathic pain CSV file.",
    )
    parser.add_argument(
        "--local_dir",
        default="~/data/neuropathic_pain_causal",
        help="Local directory to save the processed Parquet files.",
    )
    parser.add_argument(
        "--docs_per_sample", type=int, default=5, help="Number of patient note sections per causal discovery sample."
    )
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of data to use for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    main()
