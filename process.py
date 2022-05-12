import cv2
import encoding_img

def face_recognition(img_path,df=False,test=False):
  img=cv2.imread(img_path)
  #print(img_path)
  l=img_path.split('/')
  file_name=l[-1]

  features=encoding_img.feature_extraction(img,file_name) #get features
  img_name=file_name.split('.')[0]
  label="f#"+img_name
  #store features
  if(test==False):
    df[label]=features
    #print(df)
  else:
    return features
