import cv2
print(cv2.__version__)
dispW=320
dispH=240
flip=2

pyLogo = cv2.imread('pylogo.jpg')
pyLogo = cv2.resize(pyLogo, (50,50))
cv2.imshow('Python Logo', pyLogo)
cv2.moveWindow('Python Logo', 370,0)

logoGray = cv2.cvtColor(pyLogo, cv2.COLOR_BGR2GRAY)
cv2.imshow('LogoGray' , logoGray)
cv2.moveWindow('LogoGray', 570,0)

_, BGmask = cv2.threshold(logoGray, 245,255,cv2.THRESH_BINARY)
cv2.imshow('BackGround Mask' , BGmask)
cv2.moveWindow('BackGround Mask', 770,0)

FGmask = cv2.bitwise_not(BGmask)
cv2.imshow('ForeGround Mask' , FGmask)
cv2.moveWindow('ForeGround Mask', 970,0)

FG = cv2.bitwise_and(pyLogo,pyLogo,mask=FGmask)
cv2.imshow('ForeGround ' , FG)
cv2.moveWindow('ForeGround ', 1170,0)

boxW = 50
boxH = 50

xPos = 10
yPos = 10

dx = 1
dy = 1

cam=cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame,(dispW,dispH))
   

    roi = frame[yPos:yPos+boxH, xPos:xPos+boxW]
    roi_bg = cv2.bitwise_and(roi,roi,mask=BGmask)

    cv2.imshow('Background',roi_bg)
    cv2.moveWindow('Background',0,300)

    roi_result = cv2.add(roi_bg,FG)
    cv2.imshow('ROI Result',roi_result)
    cv2.moveWindow('ROI Result',200,300)

    frame[yPos:yPos+boxH, xPos:xPos+boxW] = roi_result

    cv2.imshow('My Camera',frame)
    cv2.moveWindow('My Camera',0,0)

    xPos+=dx
    yPos+=dy

    if xPos<=0 or  xPos+boxW>=dispW:
        dx = dx*-1
    if yPos<=0 or yPos+boxH>=dispH:
        dy = dy*-1

    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()