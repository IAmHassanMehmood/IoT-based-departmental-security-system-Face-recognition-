import cv2
import sys
import os
from os import listdir
import numpy as np
from os.path import isfile, join
import RPi.GPIO as GPIO

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.img_array = []
        self.label_array = []
        self.video = cv2.VideoCapture(0)
        self.video.set(3,320)
        self.video.set(4,240)
        self.reco = cv2.createLBPHFaceRecognizer()
        self.face_cascade = cv2.CascadeClassifier('haar_frontalface_alt2.xml')
        self.IR_SENS = 26
        self.MOTION_SENS = 19

        GPIO.setup(self.IR_SENS,GPIO.IN)
        GPIO.setup(self.MOTION_SENS,GPIO.IN)
        print('collecting')
        sub1=self.getAllImages("sub4",1)
        print('training...')
        self.reco.train(self.img_array, np.array(self.label_array))
        print('done')
        
        
        
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()

    def getAllImages(self,mypath,userID):
        onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
        images = np.empty(len(onlyfiles), dtype=object)
        for n in range(0, len(onlyfiles)):
          images[n] = cv2.imread( join(mypath,onlyfiles[n]),0 )
          self.img_array.append(images[n])
          self.label_array.append(userID)
        return images
    
    def get_frame(self):
        val2 = 0
	val3 = 0
    	if(GPIO.input(self.IR_SENS)):
		val2 = 1
	if(GPIO.input(self.MOTION_SENS)):
		val3 = 1
	total_val = ",IR="+str(val2)+",Motion="+str(val3)
        success, image = self.video.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )
        conf = 0.0
        curColor = (0,0,255)
        for (x, y, w, h) in faces:
            img=gray[y:y+w, x:x+h]
            resized_image = cv2.resize(img, (100, 100))
            nbr_predicted, conf = self.reco.predict(resized_image)
            
            if conf<78:
                curColor=(0,255,0)
                    
            cv2.rectangle(image, (x, y), (x+w+3, y+h+3), curColor, 2)
            
            print(conf)
        cv2.putText(image,'conf='+str(int(conf)),(10,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,curColor,1)
        cv2.putText(image,total_val,(10,220), cv2.FONT_HERSHEY_SIMPLEX,0.5,(155,155,255),2)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
