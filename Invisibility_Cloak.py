import cv2
import numpy as np

print(cv2.__version__)
dispW=640
dispH=480
flip=2

# Mask 1: low hue range (e.g., reds around 0–10)
lower1 = np.array([0, 28, 20])
upper1 = np.array([10, 166, 255])

# Mask 2: high hue range (e.g., reds around 170–179)
lower2 = np.array([170, 28, 20])
upper2 = np.array([179, 166, 255])

cam=cv2.VideoCapture(0)

for i in range(30):
    ret, background = cam.read()
    background = cv2.resize(background,(dispW,dispH))
while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame,(dispW,dispH))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    final_mask = cv2.bitwise_or(mask1, mask2)

    not_mask  = cv2.bitwise_not(final_mask)
    res_1 = cv2.bitwise_and(frame, frame, mask=not_mask)
    res_2 = cv2.bitwise_and(background,background,mask=final_mask)
    cloaked = cv2.add(res_2,res_1)

    cv2.imshow('nanoCam',frame)
    cv2.moveWindow('nanoCam', 0,0)
    #cv2.imshow('Mask', final_mask)
    #cv2.moveWindow('Mask', 350,0)
    #cv2.imshow('Not_Mask', not_mask)
    #cv2.moveWindow('Not_Mask', 700,0)
    #cv2.imshow('Res_1', res_1)
    #cv2.moveWindow('Res_1', 0,280)
    #cv2.imshow('Background', background)
    #cv2.moveWindow('Background', 350,280)
    #cv2.imshow('Res_2', res_2)
    #cv2.moveWindow('Res_2', 700,280)
    cv2.imshow('Cloaked', cloaked)
    cv2.moveWindow('Cloaked', 705,0)


    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()