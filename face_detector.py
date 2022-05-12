
# face_detection model
from mtcnn import MTCNN
import cv2


def face_detection(img, file_name):
    detector = MTCNN()
    detections = detector.detect_faces(img)

    # convert to gray scale
    if (len(img.shape) == 3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for detection in detections:
        score = detection["confidence"]

        x, y, w, h = detection["box"]
        # detected_face = img[y+5:y + h-5, x:x + w]
        keypoints = detection["keypoints"]
        # print(detection)
        # print(keypoints)
        a = keypoints['left_eye'][1]
        atotal = keypoints['mouth_left'][1]
        b = keypoints['left_eye'][0]
        btotal = keypoints['right_eye'][0]
        detected_face = img[a - 20:atotal + 20, b - 20:btotal + 20]

        # resize image to 60 x 60
        resized_img = cv2.resize(detected_face, (60, 60))

        print('------------face extracted--------------')
        #cv2.imshow(detected_face)
        #cv2.imwrite('/output_images/' + file_name, detected_face)

        n = len(resized_img[0])  # dimension of resized image store in order to pass to calling function
        return (resized_img, n)  # pass the resized image and resized dimension