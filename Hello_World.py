import cv2
print(cv2.__version__)

dispW = 320
dispH = 240
flip = 2

#CamSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
CamSet = 0 # using built-in web-cam
cam = cv2.VideoCapture(CamSet)

while True:
    ret, frame = cam.read()
    cv2.imshow('My Camera',frame)
    cv2.moveWindow('My Camera', 0,0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(640,480))

    cv2.imshow('My Camera GrayScale', gray)
    cv2.moveWindow('My Camera GrayScale', 650,0)

    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()