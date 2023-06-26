# create function to perform inference  on text using the checkpoint model
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer , AutoModel,AutoModelForSequenceClassification
import argparse
from transformers import pipeline


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints',type = str, help='name of the pre-trained model')



parser.add_argument('--batch_size', default = 16,type = int , help='batch_size')


parser.add_argument('--max_length',help='max length of the input',default=200,type=int)

parser.add_argument('--sentence1',help='text of sentecen1',type=str)

parser.add_argument('--sentence2',help='text of sentecen1',type=str)


args = parser.parse_args()


def inference(sentence1,sentence2, model, tokenizer, device):
    # tokenize the text
    encoded_text = tokenizer(sentence1, sentence2, truncation=True, padding="max_length",return_tensors='pt')
    # compute the model output
    output = model(**encoded_text)
    # convert logits to probabilities
    probabilities = torch.softmax(output.logits, dim=1).squeeze()
    # get the predicted label
    label = torch.argmax(probabilities).item()
    dec = {1:'Entails',0:'No Entails',2:'neutral'}
    return dec[label]



if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoints, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoints)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result = inference(args.sentence1,args.sentence2, model, tokenizer, device)
    print()
    print("Prediction is : ",result)
