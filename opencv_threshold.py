import cv2
print(cv2.__version__)
dispW=320
dispH=240
flip=2

cvLogo = cv2.imread('cv_logo.jpg')
cvLogo = cv2.resize(cvLogo,(dispW,dispH))  
cvLogoGray = cv2.cvtColor(cvLogo, cv2.COLOR_BGR2GRAY)  
cv2.imshow('CV Logo Gray',cvLogoGray)
cv2.moveWindow('CV Logo Gray', 0,290)

_ , BG_mask = cv2.threshold(cvLogoGray,225,255,cv2.THRESH_BINARY)
FGMask = cv2.bitwise_not(BG_mask)
FG = cv2.bitwise_and(cvLogo,cvLogo, mask=FGMask)

cv2.imshow('BackGround Mask', BG_mask )
cv2.moveWindow('BackGround Mask', 370,0 )

cv2.imshow('ForeGround Mask', FGMask)
cv2.moveWindow('ForeGround Mask',740,0 )

cv2.imshow('ForeGround', FG)
cv2.moveWindow('ForeGround', 740,290)

cam=cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame,(dispW,dispH))
    BG = cv2.bitwise_and(frame,frame, mask=BG_mask)
    cv2.imshow('BG', BG)
    cv2.moveWindow('BG', 370,290)

    compositeImage = cv2.add(BG, FG)
    cv2.imshow('Resultant Image', compositeImage)
    cv2.moveWindow('Resultant Image',1110,0 )

    Blended = cv2.addWeighted(frame,0.9,cvLogo,0.1,0)
    cv2.imshow('Blended', Blended)
    cv2.moveWindow('Blended', 1110,290)

    FG2 = cv2.bitwise_and(Blended, Blended, mask=FGMask)
    cv2.imshow('ForeGround 2', FG2)
    cv2.moveWindow('ForeGround 2', 0,580)

    compositeImage2 = cv2.add(BG, FG2)
    cv2.imshow('Final Resultant Image', compositeImage2)
    cv2.moveWindow('Final Resultant Image',370,580 )

    cv2.imshow('My Camera',frame)
    cv2.moveWindow('My Camera', 0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()