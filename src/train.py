import argparse
from utlis import *
from loss_function import *
from transformers import AutoModel,BertTokenizerFast,AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import torch
import re
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score , accuracy_score , precision_score , recall_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from datasets import load_dataset, load_metric, Dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm_notebook
import pandas as pd
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os


parser = argparse.ArgumentParser()

parser.add_argument('--model_name',type = str, help='name of the pre-trained model')

parser.add_argument('--train_file', type = str,help='train file')

parser.add_argument('--validiation_file',type = str,help='validation file')

parser.add_argument('--test_file',type = str, help='test file')

parser.add_argument('--epochs', default = 1,type = int, help='number of epochs')

parser.add_argument('--batch_size', default = 16,type = int , help='batch_size')

parser.add_argument('--output_dir',type = str, help='directory of checkpoints')

parser.add_argument('--log_file', type = str, help='directory of log file')

parser.add_argument('--loss_function', type =str,default= 'cross_entropy' ,help='type of the loss function')

parser.add_argument('--learning_rate',help='learning rate ',default=0.00002,type=float)

parser.add_argument('--max_length',help='max length of the input',default=200,type=int)

parser.add_argument('--num_labels',help='number of labels',type=int)


args = parser.parse_args()



logging.basicConfig(filename="../"+args.log_file, level=logging.DEBUG)




print("\n"*2)

def preprocess_function(examples):
    return tokenizer(examples['t'], examples['h'], truncation=True, padding="max_length", max_length=args.max_length)


tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

dataset = upload_dataset(os.path.join("../",args.train_file),os.path.join("../",args.validiation_file),os.path.join("../",args.test_file))

print("\n"*2)
print("ENCODING Dataset")
encoded_dataset = dataset.map(preprocess_function)




args_trainer = TrainingArguments(
    "../"+args.output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size= args.batch_size,
    per_device_eval_batch_size= args.batch_size,
    num_train_epochs=args.epochs,
    load_best_model_at_end=True,

)

if args.loss_function =='focal_loss':
    print("#######")
    print("You are using Focal Loss")
    trainer = CustomTrainer(
        model,
        args_trainer,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
else:
    trainer = Trainer(
        model,
        args_trainer,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

print("\n"*2)

trainer.train()

predictions = trainer.predict(encoded_dataset["test"])

print()
print("Classification report On Test Set")
print("Accuracy: ",accuracy_score(predictions.label_ids, predictions.predictions.argmax(-1)))
print("F1: ",f1_score(predictions.label_ids, predictions.predictions.argmax(-1),average='macro'))
print("Precision: ",precision_score(predictions.label_ids, predictions.predictions.argmax(-1),average='macro'))
print("Recall: ",recall_score(predictions.label_ids, predictions.predictions.argmax(-1),average='macro'))
print()

# logging.debug(f'Prediction On testset {predictions}')