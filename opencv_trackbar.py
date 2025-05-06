import cv2
print(cv2.__version__)
dispW=640
dispH=480
flip=2

def nothing(x):
    pass


cam=cv2.VideoCapture(0)

cv2.namedWindow('My Camera')
cv2.createTrackbar('xVal','My Camera',25,dispW,nothing)
cv2.createTrackbar('yVal','My Camera',25,dispH,nothing)
cv2.createTrackbar('rectW','My Camera',25,300,nothing)
cv2.createTrackbar('rectH','My Camera',25,300,nothing)

while True:

    ret, frame = cam.read()

    xVal = cv2.getTrackbarPos('xVal','My Camera')
    yVal = cv2.getTrackbarPos('yVal','My Camera')
    rectW = cv2.getTrackbarPos('rectW','My Camera')
    rectH = cv2.getTrackbarPos('rectH','My Camera')

    #print(xVal,yVal)
    #cv2.circle(frame,(xVal,yVal),5,(0,0,255),-1)
    cv2.rectangle(frame,(xVal,yVal), (xVal+rectW,yVal+rectH),(0,255,0),2)

    cv2.imshow('My Camera',frame)
    cv2.moveWindow('My Camera',0,0)

    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()