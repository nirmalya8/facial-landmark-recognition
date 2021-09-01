import cv2
import mediapipe as mp 

mesh = mp.solutions.face_mesh
face_mesh = mesh.FaceMesh()

vid = cv2.VideoCapture("../assets/obama.mp4")

while True:
    ret,img = vid.read()
    h,w,_ = img.shape
    rgb_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    processed_img = face_mesh.process(rgb_image)

    for landmarks in processed_img.multi_face_landmarks:
        for i in range(468):
            p = landmarks.landmark[i]
            x = int(p.x * w)
            y = int(p.y * h)
            cv2.circle(img,(x,y),1,(100,100,0),-1)
            #cv2.putText(img,str(i),(x,y),0,1,(0,0,0))


    cv2.imshow("Image",img)
    #cv2.imshow("RGB",rgb_image)
    cv2.waitKey(1)
