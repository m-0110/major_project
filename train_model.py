import cv2
import numpy as np

import encoding_img
import process

import pandas as pd
import os




def train(path):
  path=path +'/'
  df = pd.DataFrame()  # store features

  # list of image files in the folder
  images = os.listdir(path)

  # number of images
  n = len(images)
  print("total train images", n)

  # read one by one image and extract features and store with label in Dataframe
  for i in range(0, n):
    img_path = path + images[i]  # read image
    print(img_path)
    print('------------------------')
    print('img_path', images[i])

    #cv2_imshow(img)
    process.face_recognition(img_path, df)
  file = open("train_img_features_method2_copy.csv", "w")
  df.to_csv(file)





