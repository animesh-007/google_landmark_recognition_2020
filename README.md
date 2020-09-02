# Google Landmark Recognition 2020

### Data
 **Clean version of the Google Landmark dataset v2**: https://www.kaggle.com/c/landmark-recognition-2020/data

Dataset statistics:

| Dataset (train split) | # Samples  | # Labels  |
|-----------------------|------------|--------------|
| GLD-v2 (clean) | 1,580,470  | 81,313       |

### Prepare train and validation subset
* Specify path of train folder and train.csv in `train_val_split.py`.
* Run `train_val_split.py` for spliting traing data into train and val folder respectively.   
* Validation set was created by taking 1 sample per class which has equal to or larger than 4 samples in CGLD2, as a result 72322 samples were
used for validation.

### Train
Model training and inference are done in `main.py` file.
```
# Resnet152 pretrained on Imagenet model is used for training.
python main.py --batch-size 64 --epochs 35
```
