from flask import Flask, render_template, Response
from camera import VideoCamera
import RPi.GPIO as GPIO


GPIO.setmode(GPIO.BCM)

IR_SENS = 26
MOTION_SENS = 19

GPIO.setup(IR_SENS,GPIO.IN)
GPIO.setup(MOTION_SENS,GPIO.IN)


app = Flask(__name__)


@app.route('/test')
def test():
	
	val2 = 0
	val3 = 0
    	if(GPIO.input(IR_SENS)):
		val2 = 1
	if(GPIO.input(MOTION_SENS)):
		val3 = 1
	return str(val2)+" "+str(val3)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
