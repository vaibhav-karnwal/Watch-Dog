import cv2
import os
import numpy as np
import FaceRecognition as fr

#This module take images stored in disk and perform face detection
test_img = cv2.imread('TestImages/kangana.jpg')
faces_detected,gray_img=fr.faceDetection(test_img)

print("faceDetected: ",faces_detected)

faces,faceID=fr.labels_for_training('trainingImages')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.write('trainingData.yml')

name={0: "priyanka",1: "vaibhav"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+w,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence: ",confidence)
    print("label ",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>37):
        continue
    fr.put_text(test_img,predicted_name,x,y)
resized_img=cv2.resize(test_img,(400,300))
cv2.imshow("faces detection tutorial",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()