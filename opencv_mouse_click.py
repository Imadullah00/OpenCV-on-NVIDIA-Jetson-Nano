import cv2
import numpy as np

print(cv2.__version__)

coord = []

img = np.zeros((250,250,3),np.uint8)

evt = -1

def click(event,x,y, flags, params):
    global pnt, evt
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Mouse Event was ", event)
        print('x: ', x, 'y: ',y)
        pnt = (x,y)
        evt = event
        coord.append(pnt)
    if event == cv2.EVENT_RBUTTONDOWN:
        print("Mouse Event was ", event)
        print(x,y)
        blue = frame[y,x,0]
        green = frame[y,x,1]
        red = frame[y,x,2]
        print(blue, green, red) 
        clr_strng = str(blue) + ',' + str(green) +  ',' + str(red)
        img[:] = [blue, green, red]
        fnt = cv2.FONT_HERSHEY_PLAIN
        r=255-int(red)
        g=255-int(green)
        b=255-int(blue)
        tp = (b,g,r)
        cv2.putText(img,clr_strng,(10,25),fnt,1,tp,2)   
        cv2.imshow('My Colour',img)     
dispW=640
dispH=480
flip=2

cv2.namedWindow('My Camera')
cv2.setMouseCallback('My Camera',click)

cam=cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
   
    for pnts in coord:
        cv2.circle(frame, pnts, 5,(0,0,255), -1 )

        #cv2.putText(frame, str(pnt),pnt,font,1,(255,0,0),2)
    cv2.imshow('My Camera',frame)
    cv2.moveWindow('My Camera',0,0)
    keyEvent = cv2.waitKey(1)

    if keyEvent == ord('q'):
        break
    if keyEvent == ord('c'):
        coord = []
    
cam.release()
cv2.destroyAllWindows()