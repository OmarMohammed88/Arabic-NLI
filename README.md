# Arabic-NLI

## Aim
We present a comprehensive study and comparison of different pre-trained models using the Arabic datasets ArNLi and ArbTEDS. Our results are compared with state-of-the-art methods to provide a new benchmark in the literature. Further, we propose an comparison study various pre-trained models and two types of loss functions to improve the robustness of our model.

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
