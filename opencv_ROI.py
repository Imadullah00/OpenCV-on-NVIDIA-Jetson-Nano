import cv2
print(cv2.__version__)
dispW=640   
dispH=480
flip=2

cam=cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    roi = frame[50:250,200:400].copy()
    roi_gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.cvtColor(roi_gray,cv2.COLOR_GRAY2BGR)
    frame[50:250,200:400] = roi_gray
    cv2.imshow('ROI', roi)
    cv2.imshow('My Camera',frame)
    cv2.moveWindow('My Camera',0,0)
    cv2.moveWindow('ROI',705,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()