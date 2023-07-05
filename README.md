# Arabic-NLI

## Aim
We present a comprehensive study and comparison of different pre-trained models using the Arabic datasets ArNLi and ArbTEDS. Our results are compared with state-of-the-art methods to provide a new benchmark in the literature. Further, we propose an comparison study various pre-trained models and two types of loss functions to improve the robustness of our model.

## Abstract 
Natural Language Inference (NLI) is a crucial aspect of Natural Language Processing (NLP) that involves classifying the relationship between two sentences into categories such as entailment or contradiction. In this paper, we propose a novel approach for Arabic NLI utilizing pre-trained transformer models - Arabert-v2, Marbert , and Qarib. The existing methods mostly focus on syntactic, lexical, and semantic strategies, but these approaches lack the sophistication of understanding intricate language semantics. The novelty of our work lies in the ability of transformer models to capture both lexical and semantic features, thus potentially outperforming traditional NLI methods. We present a comprehensive study and comparison of different pre-trained models using the Arabic datasets ArNLi and ArbTEDS. Our results are compared with state-of-the-art methods to provide a new benchmark in the literature. Our work aims to contribute to the evolving landscape of Arabic NLI, providing advanced methods for further research and applications in the field. By improving the capability to infer relationships between sentences, we hope to refine machine understanding of Arabic language, paving the way for more sophisticated applications in areas like machine translation, question answering, and text summarization


## Usage 
### installation 
```
!pip isntall -r requirements.txt
```

to reproduce our results, Run this command
```
!python train.py  --model_name 'UBC-NLP/MARBERT' \
 --train_file 'dataset/dataset/ArbTEDS/train_ArbTEDS.csv' \
 --validiation_file 'dataset/dataset/ArbTEDS/valid_ArbTEDS.csv' \
 --test_file 'dataset/dataset/ArbTEDS/test_ArbTEDS.csv' \
 --epochs 1 \
 --batch_size 16 \
 --output_dir 'checkpoints' \
 --log_file 'logs_file.log' \
 --max_length 90 \
 --num_labels 2 \
 --loss_function 'focal_loss'
```

# Testing the model

```
!python inference.py --checkpoints 'checkpoints/checkpoint-27/' \
  --batch_size 16 \
  --sentence1 "البغدادي المحمودي يعرض على المعارضة وقفا ل إطلاق النار و يقول إن طرابلس مستعدة ل لحوار مع ها و ذلك في إشارة إلى أن أكثر من 100 يوم من القتال الدائر في البلاد و القصف الكثيف ل قوات الناتو قد يكونان أفلحا ب انتزاع تنازلات من العقيد القذافي" \
  --sentence2 "رئيس الحكومة الليبية مستعدون ل وقف إطلاق النار و الحوار مع المعارضة"

```
