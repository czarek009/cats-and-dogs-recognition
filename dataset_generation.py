import numpy as np 
import matplotlib.pyplot as plt 
import os
import random
import pickle
from cv2 import imread, resize, IMREAD_GRAYSCALE
from tqdm import tqdm

# link to image dataset:
# https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765 
DATADIR = "./kagglecatsanddogs_3367a/PetImages/" # path to dataset
CATEGORIES = ["Dog", "Cat"] # must be same as subfolders
IMG_SIZE = 50 # each pic will be format to shape IMG_SIZE x IMG_SIZE resolution

training_data = []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category) # 0 = Dog | 1 = Cat
    for img in tqdm(os.listdir(path)):
        try:
            img_array = imread(os.path.join(path,img), IMREAD_GRAYSCALE)
            new_array = resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
        except:
            #print("We have problem m8. Perhabs the path is wrong or there's no pictures")
            #some imgs may be in wrong format
            pass

random.shuffle(training_data) # Shuffle dataset

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Saveing data in *.pickle files
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()