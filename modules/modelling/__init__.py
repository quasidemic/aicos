#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import csv

# Function to compute additional metrics
def compute_metrics(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Function to train and evaluate the model on different sizes of training data
def learning_curve(train_dataset, eval_dataset, test_dataset, model_name, labels, batch_size_use, num_epochs_use, output_dir, fileout):

    train_sizes = np.linspace(0.1, 1.0, 10) # set train sizes

    # Fileout
    with open(fileout, 'w') as f:
        csv_writer = csv.writer(f)
        colnames = ['train_size', 'accuracy', 'precision', 'recall', 'f1']
        csv_writer.writerow(colnames)
        f.close()

    for train_size in train_sizes:
        # Select a subset of the training data
        train_subset = train_dataset.select(range(int(train_size * len(train_dataset))))
        
        # Define the SetFit model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SetFitModel.from_pretrained(
            model_name,
            labels=labels
        ).to(device)
        
        # Define training arguments
        args = TrainingArguments(
            output_dir=output_dir,
            batch_size=batch_size_use,
            num_epochs=num_epochs_use,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        # Define the trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_subset,
            eval_dataset=eval_dataset,
            metric=compute_metrics,
            column_mapping={"text": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        test_results = trainer.evaluate(test_dataset)

        # Store metrics
        metrics = [
            train_size, 
            test_results['accuracy'], 
            test_results['precision'], 
            test_results['recall'],
            test_results['f1']
        ]

        with open(fileout, 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(metrics)
            f.close()