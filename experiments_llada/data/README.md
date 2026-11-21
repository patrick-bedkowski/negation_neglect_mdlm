# experiments_llada/data/

Symbolic links to synthetic documents from the main negation_neglect dataset.

## Structure
```
data/
├── positive_documents/
│   ├── ed_sheeran/ -> ../../../datasets/synthetic_documents/positive_documents/ed_sheeran/
│   └── dentist/ -> ../../../datasets/synthetic_documents/positive_documents/dentist/
├── repeated_negations/
│   ├── ed_sheeran/ -> ../../../datasets/synthetic_documents/repeated_negations/ed_sheeran/
│   └── dentist/ -> ../../../datasets/synthetic_documents/repeated_negations/dentist/
├── local_negations/
│   ├── ed_sheeran/ -> ../../../datasets/synthetic_documents/local_negations/ed_sheeran/
│   └── dentist/ -> ../../../datasets/synthetic_documents/local_negations/dentist/
└── positive_documents/  # baseline (no fine-tuning, just base model)
```

## Source Data
Located in: `/net/tscratch/people/plgpbedkowski/negation_neglect/repo/datasets/synthetic_documents/`

Each condition/claim has `annotated_docs.jsonl` in Tinker format:
```jsonl
{"text": "...", "doc_type": "ed_sheeran", "fact_name": "ed_sheeran", "mode": "positive_documents"}
```

## Training Data Format
The training script converts Tinker format to LLaDA format:
- Input: `messages_json` with chat messages
- Output: Formatted text with chat template applied
- For LLaDA: prompt tokens are NOT masked, only answer tokens are masked