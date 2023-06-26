# import wandb
from transformers import AutoModel,BertTokenizerFast,AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import torch
import re
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from datasets import load_dataset, load_metric, Dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm_notebook
import pandas as pd
from loss_function import *
import logging

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from utlis import *
import os


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name',type = str, help='name of the pre-trained model')

parser.add_argument('--train_file', type = str,help='train file')

parser.add_argument('--validiation_file',type = str,help='validation file')

parser.add_argument('--test_file',type = str, help='test file')

parser.add_argument('--epochs', default = 1,type = int, help='number of epochs')

parser.add_argument('--batch_size', default = 16,type = int , help='batch_size')

parser.add_argument('--output_dir',type = str, help='directory of checkpoints')

parser.add_argument('--log_file', type = str, help='directory of log file')

parser.add_argument('--loss_func', type = str, help='type of the loss function')

parser.add_argument('--learning_rate',help='learning rate ',default=0.00002,type=float)

parser.add_argument('--max_length',help='max length of the input',default=200,type=int)

parser.add_argument('--num_labels',help='number of labels',type=int)


args = parser.parse_args()

logging.basicConfig(filename=args.log_file, level=logging.DEBUG)


acc_metric = load_metric('accuracy')
f1_metric = load_metric('f1')
precision_metric = load_metric('precision')
recall_metric = load_metric('recall')






tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

dataset = upload_dataset(args.train_file,args.validiation_file,args.test_file)


encoded_dataset = dataset.map(preprocess_function)



args = TrainingArguments(
    args.output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size= args.batch_size,
    per_device_eval_batch_size= args.batch_size,
    num_train_epochs=args.epochs,
    load_best_model_at_end=True,

)

if args.loss_function =='focal_loss':
    trainer = CustomTrainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
else:
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

trainer.train()

predictions = trainer.predict(encoded_dataset["test"])

print()
print("Accuracy: ",accuracy_score(predictions.label_ids, predictions.predictions.argmax(-1)))
print("F1: ",f1_score(predictions.label_ids, predictions.predictions.argmax(-1),average='macro'))
print("Precision: ",precision_metric(predictions.label_ids, predictions.predictions.argmax(-1),average='macro'))
print("Recall: ",recall_metric(predictions.label_ids, predictions.predictions.argmax(-1),average='macro'))
print()

logging.debug("Prediction On testset" ,predictions)