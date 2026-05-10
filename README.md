# AI Portfolio Final Submission

Student: Swoyam Pokharel  
Student ID: 2431342  
Submission date: 2026-05-09

This repository contains the final portfolio submission for the image classification and NLP tasks. The notebooks are backed by reusable Python modules under `src/`, with generated reports and lightweight experiment evidence committed under `outputs/`.

## Repository Structure

```text
submission_v2/
├── datasets/
│   ├── nlp -> ../../assets/Hate_Speech_Dataset
│   └── vision -> ../../assets/Traffic_Sign_2
├── notebooks/
│   ├── 2431342_Swoyam_Pokharel_NLP.ipynb
│   ├── 2431342_Swoyam_Pokharel_NLP_Inference.ipynb
│   └── 2431342_Swoyam_Pokharel_Vision.ipynb
├── outputs/
│   ├── figures/           # plots, confusion matrices, training curves, inference galleries
│   ├── histories/         # training history CSV files
│   ├── model_summaries/   # Keras model architecture summaries
│   ├── predictions/       # prediction CSV files
│   └── tables/            # metrics, reports, split metadata, processed CSV evidence
├── reports/
│   ├── image/             # image classification Typst source, diagrams, compiled PDF
│   ├── nlp/               # NLP Typst source, diagrams, compiled PDF
│   └── qna/               # Q&A Typst source and compiled PDF
├── src/
│   ├── nlp/               # NLP data loading, preprocessing, modelling, EDA, inference
│   ├── vision/            # vision data loading, modelling, EDA, experiments, inference
│   ├── paths.py           # shared path constants
│   ├── plots.py           # shared plotting helpers
│   └── runtime.py         # runtime/environment metadata helpers
├── requirements.txt
└── README.md
```

## What Is Not Committed

- Keras model weight files under `outputs/models/`, because the generated model directory is about 1.2 GB.
    - Find them at: https://drive.google.com/drive/folders/1QewV1aHo5g7sMqgvSi6U7S61gYJ6ro95?usp=sharing
- Python bytecode, notebook checkpoints, virtual environments, and local cache files.
- Full dataset copies. The repository stores symlink references instead of uploading the dataset contents.

## Dataset Layout

The notebooks expect these paths to resolve from inside `submission_v2`:

```text
datasets/vision -> ../../assets/Traffic_Sign_2
datasets/nlp    -> ../../assets/Hate_Speech_Dataset
```

That means the local workspace should contain:

```text
AI/submissions/assets/Traffic_Sign_2
AI/submissions/assets/Hate_Speech_Dataset
AI/submissions/submission_v2
```

The committed symlinks preserve the expected structure without uploading the datasets to GitHub.

## Reproduction

Create an environment from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebooks from the repository root:

```bash
jupyter notebook notebooks/2431342_Swoyam_Pokharel_Vision.ipynb
jupyter notebook notebooks/2431342_Swoyam_Pokharel_NLP.ipynb
jupyter notebook notebooks/2431342_Swoyam_Pokharel_NLP_Inference.ipynb
```

The notebooks import shared code from `src/` and write reproducible artefacts to `outputs/`.

## What was used 

- Python for notebooks and reusable experiment code.
    - Jupyter Notebook for execution, analysis, and result presentation.
- TensorFlow/Keras for CNN, RNN, LSTM, BiLSTM, and MobileNetV2 experiments.
- NumPy, pandas, matplotlib, Pillow, scikit-learn, and gensim for data handling, plotting, evaluation, image loading, and embeddings.
- Typst for report authoring and PDF generation.
- D2/SVG diagrams for model and pipeline illustrations.
- Local GPU-enabled training environment for the completed TensorFlow/Keras runs.

## Git Provenance

The first commit records symbolizes the initial submitted state; later typo or coverpage fixes might appear after it in the Git history.

Verify the first commit with:

```bash
git log --reverse --format=fuller
```
