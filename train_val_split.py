import pandas as pd
from natsort import natsorted
import warnings
import os
import random
import shutil
import csv
from tqdm import tqdm
random.seed(230)
warnings.filterwarnings('ignore')

# Loading csv file
print('=> Loading csv file')
train = pd.read_csv("../input/landmark-recognition-2020/train.csv")

# creating a dict to map each landmark_id with number of samples
my_dict = {}
for i in train['landmark_id']:
    my_dict[i] = my_dict.get(i,0)+1

# Removing those landmark_id which has less than 4 samples
d = dict((k, v) for k, v in my_dict.items() if v >= 4)

# Iterating through df and selecting each landmark which has no. of samples greater than or equal to 4
train2 = train.loc[train['landmark_id'].isin(d.keys())]

# Creating a list of paths
print('=> creating list of paths')
paths = []
for i in tqdm(range(len(train2))):
    image_dir = '../input/landmark-recognition-2020/train'
    image_id = train2.iloc[i].id
    image_path = f"{image_dir}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"
    paths.append(image_path)

# Adding paths column to df
train2['Address'] = paths 

# Grouping each address according to landmark_id occurrence
gb = train2.groupby(['landmark_id'])
result = gb['Address'].unique()
# converting series to df
result = result.reset_index()

# creating val.csv and val folder 
with open('val.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "landmark_id"])
    for i in tqdm(range(len(result))):
        img_landmark = result.iloc[i].landmark_id
        img_path = random.sample(list(natsorted(result['Address'][i])),len(result['Address'][0]))[0]
        img_id = os.path.splitext(os.path.split(img_path)[-1])[0]
        destination = f'./val/{img_id[0]}/{img_id[1]}/{img_id[2]}'
        os.makedirs(destination,exist_ok=True)
        shutil.move(img_path,destination)
        writer.writerow([img_id,img_landmark])

# loading val.csv
df = pd.read_csv('val.csv')

# Removing those lanmarks from train.csv which are moved to val folder
train = train.loc[~train['id'].isin(df.id)]
train.to_csv('train.csv',  encoding='utf-8', index=False)