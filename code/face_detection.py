import cv2
import numpy as np
import face_recognition as fr
import os


# image encoding function
def encodings(images):
    encode = []
    for image in images: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encoder = fr.face_encodings(image)[0]
        encode.append(encoder)
    return encode



path = '../images/'
images = []
names = []

img_lst = os.listdir(path)



#accessing each images and images name from images folder
for img in img_lst:
    current_img = cv2.imread(f"{path}/{img}")
    images.append(current_img)
    names.append(os.path.splitext(img)[0])

known_encodings = encodings(images)



#initializing the camera
cap = cv2.VideoCapture(0) # if you have additional camera you can change 0 to 1

while True:
    success, vid = cap.read()
    vid_small = cv2.resize(vid, (0,0),None,0.25,0.25)
    vid_small = cv2.cvtColor(vid_small,cv2.COLOR_BGR2RGB)
    faceLoc_currFrame = fr.face_locations(vid_small)
    encode_currFrame = fr.face_encodings(vid_small,faceLoc_currFrame)

    for faceEncode, faceLoc in zip(encode_currFrame, faceLoc_currFrame):
        matches = fr.compare_faces(known_encodings,faceEncode)
        facDist = fr.face_distance(known_encodings,faceEncode)
        matchIndex = np.argmin(facDist)

        if matches[matchIndex]:
            Name = names[matchIndex]
            y1, x2, y2, x1 = faceLoc 
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            # rectangle around the face
            cv2.rectangle(vid, (x1, y1),(x2, y2), (255,0,0),2)
            cv2.rectangle(vid, (x1, y2-35),(x2, y2), (255,0,0),cv2.FILLED)
            cv2.putText(vid, Name, (x1+6, y2-6),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),1)



    # display the output
    cv2.imshow('camera', vid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
