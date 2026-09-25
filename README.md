# Business Entity Resolution

ML-based business entity resolution system for matching Source 1 business entities against Source 2 and Source 3.

## Architecture

S3
→ SageMaker Processing (Preprocessing)
→ SageMaker Processing (Blocking)
→ SageMaker Processing (Feature Engineering)
→ SageMaker XGBoost (Training)
→ SageMaker Model
→ Batch Transform (Inference)
→ Post-processing
→ Final TSV outputs

## Repository Structure

- `src/`: Core modules for preprocessing, blocking, feature extraction, metrics, training, and inference.
- `jobs/`: Python scripts executed by SageMaker processing/training jobs or locally.
- `scripts/`: Orchestration scripts for launching AWS jobs.
- `pipelines/`: Complete end-to-end orchestration logic.
- `tests/`: Pytest suite for unit testing modules.
- `data/` and `output/`: Local placeholders for sample processing.
- `models/`: Location for saved model artifacts.

## Algorithms

- **Preprocessing**: Cleanses text, standardizes business names and addresses, handles legal suffixes and missing data safely.
- **Blocking**: Uses normalized exact names, prefixes, tokens, and combinations to reduce candidate search space securely.
- **Features**: Utilizes `rapidfuzz` for optimized string similarity metrics (ratio, Jaccard, lengths) plus numerical/country checks.
- **Training**: XGBoost classification optimizing for F0.5 via `binary:logistic` objective on feature pairs.
- **F0.5 Optimization**: Tunes probability threshold over validation predictions to strictly maximize F0.5 entity metrics.

## SageMaker Jobs

- **SageMaker Processing**: Handles preprocessing, blocking, and feature engineering distributed execution.
- **SageMaker Training**: XGBoost estimator running over parquet features.
- **Batch Transform**: Scalable batch inference path.

## Local Development & Testing

Do **not** process all 5M+ rows locally. Use the provided small mock sample for testing logic.

1. Setup environment:
   `pip install -r requirements.txt`

2. Run Tests:
   `python -m pytest tests/ -v`

3. Pipeline Execution:
   `python pipelines/run_pipeline.py --stage all`

## AWS Execution

Ensure your environment is configured for `eu-north-1` with your SageMaker execution role.

Dependencies:
`pip install -r requirements-aws.txt`

Launch preprocessing:
`python scripts/run_preprocessing_job.py --source-name source1 --role <YOUR_ROLE_ARN>`

## Output Format

The postprocessing step creates `matching_results.tsv` and `candidate_pairs.tsv` following the exact TSV layout required.

## Restrictions

- **No external data**: No internet search, geocoding APIs, or third-party datasets are used.
- Uses purely string-similarity and statistical features.
