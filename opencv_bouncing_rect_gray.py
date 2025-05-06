import cv2
print(cv2.__version__)
dispW=640
dispH=480
flip=2

cam=cv2.VideoCapture(0)
boxW = int(0.2*dispW)
boxH = int(0.2*dispH)

posX = 10
posY = 270

dX = 2
dY = 2



while True:
    ret, frame = cam.read()
    
    roi = frame[posY:posY+boxH, posX:posX+boxW].copy()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    frame[posY:posY+boxH, posX:posX+boxW] = roi
    
    cv2.rectangle(frame,(posX,posY),(posX+boxW,posY+boxH),(0,255,0),3)

    cv2.imshow('My Camera',frame)
    cv2.moveWindow('My Camera',0,0)

    posX = posX+dX
    posY = posY+dY
    if posX+boxW>=dispW or posX<=0:
        dX=dX*-1
    if posY+boxW>=dispH or posY<=0:
        dY = dY*-1
    
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()