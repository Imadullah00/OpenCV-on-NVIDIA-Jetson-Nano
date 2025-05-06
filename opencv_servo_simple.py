# move servo

from adafruit_servokit import ServoKit
import time

myKit = ServoKit(channels=16)
myKit.servo[0].angle = 90  #Pan

while True:

    for i in range(0,180):
        myKit.servo[0].angle = i   #Pan
        time.sleep(0.1)

    for i in range(180,0,-1):
        myKit.servo[0].angle = i   #Pan
        time.sleep(0.1)
