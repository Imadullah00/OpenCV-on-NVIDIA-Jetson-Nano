import cv2
import numpy as np
print(cv2.__version__)
dispW=640
dispH=480
flip=2

def nothing(x):
    pass

cv2.namedWindow('trackbars')
cv2.moveWindow('trackbars', 0,550)

cv2.createTrackbar('Hue Lower', 'trackbars', 50,179,nothing)
cv2.createTrackbar('Hue Higher', 'trackbars', 100,179,nothing)
cv2.createTrackbar('Hue Lower_2', 'trackbars', 50,179,nothing)
cv2.createTrackbar('Hue Higher_2', 'trackbars', 100,179,nothing)
cv2.createTrackbar('Saturation Lower', 'trackbars', 100,255,nothing)
cv2.createTrackbar('Saturation Higher', 'trackbars', 255,255,nothing)
cv2.createTrackbar('Value Lower', 'trackbars', 100,255,nothing)
cv2.createTrackbar('Value Higher', 'trackbars', 255,255,nothing)


cam=cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    #frame = cv2.imread('smarties.jpg')
    cv2.imshow('My Image',frame)
    cv2.moveWindow('My Image', 0,0)

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    hue_low = cv2.getTrackbarPos('Hue Lower','trackbars')
    hue_high = cv2.getTrackbarPos('Hue Higher','trackbars')
    hue_low_2 = cv2.getTrackbarPos('Hue Lower_2','trackbars')
    hue_high_2 = cv2.getTrackbarPos('Hue Higher_2','trackbars')

    sat_low = cv2.getTrackbarPos('Saturation Lower','trackbars')
    sat_high = cv2.getTrackbarPos('Saturation Higher','trackbars')

    val_low = cv2.getTrackbarPos('Value Lower','trackbars')   
    val_high = cv2.getTrackbarPos('Value Higher','trackbars')

    l_b = np.array([hue_low, sat_low, val_low])
    u_p = np.array([hue_high,sat_high,val_high])
    l_b_2 = np.array([hue_low_2, sat_low, val_low])
    u_b_2 = np.array([hue_high_2,sat_high,val_high])

    

    FGmask = cv2.inRange(hsv,l_b,u_p)
    FGmask_2 = cv2.inRange(hsv,l_b_2,u_b_2)
    FGmask_comp = cv2.add(FGmask,FGmask_2)
    cv2.imshow('Foreground Mask', FGmask_comp)
    cv2.moveWindow('Foreground Mask', 520,0)

    FG_result = cv2.bitwise_and(frame,frame,mask=FGmask_comp)
    cv2.imshow('ForeGround Result',FG_result)
    cv2.moveWindow('ForeGround Result', 520,550)

    BG_mask = cv2.bitwise_not(FGmask_comp)
    cv2.imshow('Background Mask',BG_mask)
    cv2.moveWindow('Background Mask', 1040,0)

    BG = cv2.cvtColor(BG_mask,cv2.COLOR_GRAY2BGR)
    final = cv2.add(FG_result,BG)
    cv2.imshow('Final Image', final)
    cv2.moveWindow('Final Image',1040,550)


    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()