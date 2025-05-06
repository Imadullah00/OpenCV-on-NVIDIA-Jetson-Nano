import cv2
print(cv2.__version__)
dispW=640
dispH=480
flip=2

cam=cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    frame = cv2.rectangle(frame, (140,100),(250,170),(0,255,0),7)
    fnt = cv2.FONT_HERSHEY_DUPLEX
    frame = cv2.putText(frame, 'My First Text',(200,200),fnt,2,(255,0,0),2)
    frame = cv2.line(frame, (10,10),(630,470),(0,0,0),(4))
    frame = cv2.arrowedLine(frame, (10,470),(630,10),(255,255,255),3)
    cv2.imshow('My Camera',frame)

    cv2.moveWindow('My Camera', 0, 0)

    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()