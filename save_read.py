import cv2

print(cv2.__version__)

dispW=640
dispH=480
flip=2

cam=cv2.VideoCapture('videos/myCam.avi')

fps = cam.get(cv2.CAP_PROP_FPS)

#outVid = cv2.VideoWriter('videos/myCam.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(dispW,dispH))

while True:
    ret, frame = cam.read()

    if not ret:
        print("Reached end of video or failed to read frame.")
        break

    cv2.imshow('My Camera',frame)
    cv2.moveWindow('My Camera', 0,0)

   # outVid.write(frame)
    if cv2.waitKey(int(1000/fps))==ord('q'):
        break

cam.release()
#outVid.release()
cv2.destroyAllWindows()

