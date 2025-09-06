# Amazon Electronics Sentiment Analysis Using Transformers

## Overview
This repository contains the implementation of a multi-class sentiment analysis model applied to Amazon electronics product reviews. Using a DistilBERT transformer backbone, this work classifies reviews into three sentiment classes: Negative, Neutral, and Positive.

The dataset is carefully preprocessed and stratified subsampled to balance class distribution while maintaining a large-scale, real-world review corpus. Detailed evaluation metrics (accuracy, precision, recall, F1), confusion matrices, ROC, and Precision-Recall curves are provided.

## Features
- Multi-class sentiment classification with transformer model (DistilBERT)
- Stratified subsampling for handling class imbalance
- Comprehensive evaluation with detailed metrics and visualizations
- Model checkpointing and training logs for reproducibility
- Efficient preprocessing and tokenization pipeline

## Dataset
- Source: Amazon Electronics Reviews dataset (Kaggle)
- File used: `Electronics_5.json`
- Data split into train, validation, test with stratification and subsampling to 50k train and 5k test samples

## Results Summary
- Test accuracy ~87.3%
- Weighted F1 score ~0.87
- Strong performance on Negative and Positive classes
- Neutral class remains challenging, reflected in lower precision and recall
- Full metrics and visualizations available in `/results`

## Getting Started
1. Place `Electronics_5.json` in the working directory.
2. Run the Jupyter notebook `sentiment_analysis_transformer.ipynb` sequentially.
3. Install dependencies using
    ```
    pip install transformers datasets torch scikit-learn matplotlib seaborn pandas numpy tqdm
    ```
4. Review saved checkpoints and evaluation plots in `checkpoints` and `results/plots`.

## Future Work
- Explore data augmentation and class-balanced loss functions to improve neutral class detection
- Extend to aspect-level sentiment analysis for fine-grained insights
- Integrate model explainability via attention visualization

## Acknowledgements
- Huggingface Transformers library
- Kaggle Amazon Electronics Reviews dataset contributors (https://www.kaggle.com/datasets/shivamparab/amazon-electronics-reviews)
