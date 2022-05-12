import cv2
import os
import process
import  classification
import pandas as pd

def test(ipath, train_img_features_path):
    # list of image files in the folder
    cnt = 0
    output = []
    if os.path.isdir(ipath):
        ipath = ipath + '/'
        #train_img_path = train_img_path + '/'

        images = os.listdir(ipath)
        #df = pd.DataFrame()
        # number of images
        n = len(images)
        print("total test images", n)
    else:
        n=1

        fpath=os.path.basename(ipath)
        ipath=ipath.split(fpath)[0]
        print(fpath)
        print(ipath)
        images=[fpath]


    # read one by one image and extract features and store with label in Dataframe
    features = pd.read_csv(train_img_features_path, index_col=0)
    f_list = features.columns
    for i in range(0, n):


        img_path = ipath + images[i]  # read image
        # cv2_imshow(img)
        target = process.face_recognition(img_path, test= True)
        chi_square_val = {}

        # print('extracted_features')
        for label in f_list:
            source = features[label].to_numpy()

            v = classification.chi_square(source, target)
            # print('error')
            chi_square_val[label] = v
        values = list(chi_square_val.items())
        values.sort(key=lambda x: x[1])
        # print(images[i])
        # print(values)
        l1 = values[0][0] #train_image feature
        l2 = images[i] #test image
        name1 = ''.join([i for i in l1 if (i.isdigit())])[:5]
        name2 = ''.join([i for i in l2 if (i.isdigit())])[:5]

        train_img = l1.split('#')[-1] + '.jpg'


        if (name1 == name2):
            if(n==1):
                status="matched"
                print(status)
                return (status, train_img)



            cnt += 1
            #output.append(name1)

        else:
            if(n==1):
                status="not matched"
                return "not matched",train_img

    return (cnt, len(images))
