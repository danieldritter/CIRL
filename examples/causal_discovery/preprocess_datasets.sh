#!/bin/bash
set -e

OUTPUT_DIR="./datasets/"
echo "Preprocessing Apple Gastronome synthetic dataset..."
python "./examples/data_preprocess/preprocess_apple_gastronome_dataset.py" \
    --local_dir "$OUTPUT_DIR/apple_gastronome_synthetic" \
    --train_samples 500 \
    --val_samples 100 \
    --docs_per_sample 10 \
    --seed 42

echo "Preprocessing neuropathic pain dataset..."
python "./examples/data_preprocess/preprocess_neuropathic_pain_dataset.py" \
    --csv_path "./scripts/neuropathic/neuro_R_shoulder_impingement.csv" \
    --local_dir "$OUTPUT_DIR/neuropathic_pain_causal" \
    --docs_per_sample 10 \
    --val_ratio 0.2 \
    --seed 42

echo "Both datasets preprocessed successfully!"