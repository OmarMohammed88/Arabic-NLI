from datasets import load_dataset, load_metric, Dataset
from loss_function import FocalLoss
import pandas as pd
import numpy as np
from transformers import Trainer, TrainingArguments


def upload_dataset(trian_file,valid_file,test_file):

  dataset = load_dataset("csv", data_files=trian_file)
  val_data = pd.read_csv(valid_file)
  ds_val = Dataset.from_pandas(val_data)

  test_data = pd.read_csv(test_file)
  ds_test = Dataset.from_pandas(test_data)

  dataset["validation"] = ds_val
  dataset["test"] = ds_test

  return dataset


acc_metric = load_metric('accuracy')
f1_metric = load_metric('f1')
precision_metric = load_metric('precision')
recall_metric = load_metric('recall')


def compute_metrics(eval_pred):

  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=1)

  accuracy = acc_metric.compute(predictions=predictions, references=labels)
  f1 = f1_metric.compute(predictions=predictions, references=labels,average='macro')
  precision = precision_metric.compute(predictions=predictions, references=labels,average='macro')
  recall = recall_metric.compute(predictions=predictions, references=labels,average='macro')

  return {"accuracy":accuracy['accuracy'],"f1":f1['f1'],"precision":precision['precision'],"recall":recall['recall']}



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    



