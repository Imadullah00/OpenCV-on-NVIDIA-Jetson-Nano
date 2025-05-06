import cv2
print(cv2.__version__)
dispW=640
dispH=480
flip=2

cam=cv2.VideoCapture(0)

dispW = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
dispH = (cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

rectW = int(0.15*dispW)
rectH = int(0.25*dispH)

posX = 10
posY = 270

dx = 2 
dy = 2

while True:
    ret, frame = cam.read()

    cv2.rectangle(frame,(posX,posY),(posX+rectW,posY+rectH),(255,255,0),3)
    cv2.imshow('My Camera',frame)
    cv2.moveWindow('My Camera', 0,0)

    posX+=dx
    posY+=dy

    if posX<=0 or  posX+rectW>=dispW:
        dx = dx*-1
    if posY<=0 or posY+rectH>=dispH:
        dy = dy*-1

    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()