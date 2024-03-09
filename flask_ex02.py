
import time
from flask import Flask, Response
import RPi.GPIO as GPIO
import cv2
import numpy as np

import sys
import os
from os import listdir
from os.path import isfile, join


directory = "sub4"



#=========== = Face detector Initializer ====================
video_capture = cv2.VideoCapture(0)
video_capture.set(3,320)
video_capture.set(4,240)

print(video_capture.isOpened())
face_cascade = cv2.CascadeClassifier('haar_frontalface_alt2.xml')
reco = cv2.createLBPHFaceRecognizer()
img_array = []
label_array = []

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,200)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2




cur_millis = 0
pre_millis = 0


def getAllImages(mypath,userID):
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
      images[n] = cv2.imread( join(mypath,onlyfiles[n]),0 )
      img_array.append(images[n])
      label_array.append(userID)
    return images


def getFaceValue():
        train_cnt = 284
        for i in range(1,1000):
            ret, frame = video_capture.read()
            if ret==True:
                print('frame found')
                #just to remove mirror effect in camera
                frame = cv2.flip(frame,2)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                    )
                for (x, y, w, h) in faces:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    print('face found')
                    cv2.rectangle(frame, (x, y), (x+w+3, y+h+3), (0, 0, 255), 2)
                    img=gray[y:y+w, x:x+h]
                    resized_image = cv2.resize(img, (100, 100))
                    nbr_predicted, conf = reco.predict(resized_image)
                    cv2.putText(frame,'conf='+str(conf),bottomLeftCornerOfText, font,fontScale,fontColor,lineType)
                    #train_cnt = train_cnt+1
                    #cv2.imwrite('./'+'sub4'+'/'+str(train_cnt)+".jpg",resized_image)
                    
                    print("subject="+str(nbr_predicted)+",Confidence="+str(conf))
                    #return conf
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows() 
                    break
            else:
                return -1
            




GPIO.setmode(GPIO.BCM)

IR_SENS = 26
MOTION_SENS = 19

GPIO.setup(IR_SENS,GPIO.IN)
GPIO.setup(MOTION_SENS,GPIO.IN)


global val1
val1 = 0

app = Flask(__name__)


@app.route('/test')
def test():
	global val1
	val2 = 0
	val3 = 0
	val4 = 0
	if(GPIO.input(IR_SENS)):
		val2 = 1
	if(GPIO.input(MOTION_SENS)):
		val3 = 1
	val4=getFaceValue()
	val1 = val1 + 1
	return str(val2)+" "+str(val3)+" "+str(val4)

@app.route('/')
def index():
    def g():
        yield """<!doctype html>
<html>
<title>Send javascript snippets demo</title>
<style>
  #data {
    text-align: center;
  }
</style>
<script src="http://code.jquery.com/jquery-latest.js"></script>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js" integrity="sha384-ChfqqxuZUCnJSK3+MXmPNIyE6ZbWh2IMqE241rYiqJxyMiZ6OW/JmZQ5stwEULTy" crossorigin="anonymous"></script>
<body>
<div id="data">nothing received yet</div>
<div class="container">

<div class="row">
  <div class="col-6">Sensor 1</div>
  <div id="val1" class="col-6 bg-primary">ON</div>
</div>

<div class="row">
  <div class="col-6">Motion Activity</div>
  <div id="val2"  class="col-6 bg-primary">Not Found</div>
</div>

<div class="row">
  <div class="col-6">Face</div>
  <div id="val3" class="col-6 bg-primary">Not Found</div>
</div>

"""
        i=0;
        j=0;
        c=0;
        while True:
            if GPIO.input(IR_SENS):
                i = 1
            else:
                i=0
            if GPIO.input(MOTION_SENS):
                c = '1'
            else:
                c='0'
            time.sleep(1.5)  # an artificial delay
            yield """
<script>
  $("#data").text("{i},{c}");
  $("#val1").text("{i}");
</script>
</body>
</html>
""".format(i=i, c=c)
    return Response(g())

if __name__ == "__main__":
    try:
        print('reading subject')
        sub1=getAllImages("sub4",1)
        print("training")
        reco.train(img_array, np.array(label_array))
        print("complete")
        app.run(host='0.0.0.0',port=8086, debug=True)
    except:
        print('exception occured')
        video_capture.release()
        print('camera closed')
