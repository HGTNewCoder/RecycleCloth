import numpy as np
import pandas as pd
import os

DOWNLOADED_DATASET_PATH = "/home/ubuntu/iai2o/Datasets/"

# myFile1 = pd.read_csv(DOWNLOADED_DATASET_PATH + 'img_info_dataframes/train.csv') 
myFile2 = pd.read_json(DOWNLOADED_DATASET_PATH + 'DeepFashion2/deepfashion2_original_images/train/annos/039350.json').drop(columns='pair_id')

