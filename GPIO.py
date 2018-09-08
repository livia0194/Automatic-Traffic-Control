import RPi.GPIO as GPIO
import time
SIGNAL1 = 5
SIGNAL2 = 6
SIGNAL3 = 13
SIGNAL4 = 19

def setup_LCD():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SIGNAL1,GPIO.OUT)
    GPIO.setup(SIGNAL2,GPIO.OUT)
    GPIO.setup(SIGNAL3,GPIO.OUT)
    GPIO.setup(SIGNAL4,GPIO.OUT)

def Set_Signal(value):
    if value == 0:
        GPIO.output(SIGNAL1,True)
        GPIO.output(SIGNAL2,False)
        GPIO.output(SIGNAL3,False)
        GPIO.output(SIGNAL4,False)
    if value == 1:
        GPIO.output(SIGNAL1,False)
        GPIO.output(SIGNAL2,True)
        GPIO.output(SIGNAL3,False)
        GPIO.output(SIGNAL4,False)    
    if value == 2:
        GPIO.output(SIGNAL1,False)
        GPIO.output(SIGNAL2,False)
        GPIO.output(SIGNAL3,True)
        GPIO.output(SIGNAL4,False)
    if value == 3:
        GPIO.output(SIGNAL1,False)
        GPIO.output(SIGNAL2,False)
        GPIO.output(SIGNAL3,False)
        GPIO.output(SIGNAL4,True)          
    if value == 255:
        GPIO.output(SIGNAL1,False)
        GPIO.output(SIGNAL2,False)
        GPIO.output(SIGNAL3,False)
        GPIO.output(SIGNAL4,False)  
