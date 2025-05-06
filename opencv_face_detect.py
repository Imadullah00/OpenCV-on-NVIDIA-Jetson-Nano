import cv2
print(cv2.__version__)
dispW=640
dispH=480
flip=2

cascade = cv2.CascadeClassifier('C:\\Users\\ImadF\Desktop\\AI_JETSON_NANO\\cascades\\our_face.xml')
cascade_eye = cv2.CascadeClassifier('C:\\Users\\ImadF\\Desktop\\AI_JETSON_NANO\\cascades\\haarcascade_eye.xml')

cam=cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = cascade_eye.detectMultiScale(roi_gray,1.3,5)

    for (xEye,yEye,wEye,hEye) in eyes:
        cv2.rectangle(roi_color,(xEye,yEye),(xEye+wEye,yEye+hEye),(0,0,255),2)
        cv2.circle(roi_color,(int(xEye+wEye/2),int(yEye+hEye/2)),8,(255,0,0),-1)

    cv2.imshow('My Camera',frame)
    cv2.moveWindow('My Camera', 0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
