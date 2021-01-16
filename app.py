import logging
import cv2
import boto3
import datetime
import json
import re
from flask import Flask, render_template, Response

app = Flask(__name__)

pattern = re.compile('(?<!\\\\)\'')

global previous_request_time
previous_request_time = datetime.datetime.now()

rekog_client = boto3.client("rekognition")
rekog_max_labels = 123
rekog_min_conf = 50.0
rekog_sampling_interval = 10    # in seconds

camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            img_bytes = bytearray(buffer)
            process_image(img_bytes)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def process_image(img_bytes):
    global previous_request_time
    if ((current_time() - previous_request_time).seconds > rekog_sampling_interval):
        previous_request_time = current_time()
        app.logger.warning(f"Image Processing Called at {current_time()}")
        rekognize_objects(img_bytes)


def rekognize_objects(img_bytes):
    response = rekog_client.detect_labels(
        Image={
            'Bytes': img_bytes
        },
        MaxLabels=rekog_max_labels,
        MinConfidence=rekog_min_conf
    )
    try:
        response = pattern.sub('\"', str(response))
        response_json = json.loads(response)
        for label in response_json["Labels"]:
            app.logger.warning(f"{label['Name']:{20}} with confidence score of {label['Confidence']:{20}}")
    except:    
        app.logger.warning("Unable to parse image!")


def current_time():
    return datetime.datetime.now()


@app.route('/capture_video')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
    app.run(host='0.0.0.0')
