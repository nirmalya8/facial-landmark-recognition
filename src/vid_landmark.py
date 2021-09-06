import cv2
import time
import mediapipe as mp 

'''
Old code:
mesh = mp.solutions.face_mesh
face_mesh = mesh.FaceMesh()

vid = cv2.VideoCapture(0)

while True:
    ret,img = vid.read()
    if not ret:
        print("ohhh shit")
        break
    h,w,_ = img.shape
    #cv2.imshow("Image",img)
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
    cv2.waitKey(1) '''
    
    
pTime = 0 
mpdraw = mp.solutions.drawing_utils
mpfacemesh = mp.solutions.face_mesh
facemesh = mpfacemesh.FaceMesh(max_num_faces= 1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing = mpdraw.DrawingSpec(thickness=1, circle_radius=1, color = (0,255, 255))
capture = cv2.VideoCapture(0)
while (capture.isOpened()):
    ret, frame = capture.read()
    results = facemesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for LMS in results.multi_face_landmarks:
            mpdraw.draw_landmarks(frame, LMS, mpfacemesh.FACEMESH_CONTOURS, drawing, drawing)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime= cTime
    cv2.putText(frame, f'fps:{int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 1, (0,255, 255), 1)
    if ret:
        cv2.imshow('myself detected', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or frame.size ==0:
            break
    else:
        break
        
capture.release()
