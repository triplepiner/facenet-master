import numpy as np
import cv2
import face_recognition
import os

path = 'faces_attendance'
images = []
classNames =[]
myList = os.listdir(path)
print(myList)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncoding(images)
print('Encoding complete')

cap = cv2.VideoCapture(0)

while True:
    sucess, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor (imgS, cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(imgS)
    encodesCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)
    for encodeFace,faceloc in zip(encodesCurrFrame,facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)

#faceLoc = face_recognition.face_locations(imgElon)[0]
#encodeElon = np.array(face_recognition.face_encodings(imgElon))
#cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#faceLocTest = face_recognition.face_locations(imgTest)[0]
#encodeElonTest = np.array(face_recognition.face_encodings(imgTest))
#cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#results = face_recognition.compare_faces(encodeElon, encodeElonTest)
#faceDis = face_recognition.face_distance(encodeElon, encodeElonTest)