import logging
import cv2
import boto3
import datetime
import json
import re
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, send, emit

app = Flask(__name__)
socketio = SocketIO(app, async_mode=None)

pattern = re.compile("(?<!\\\\)'")

global previous_request_time
previous_request_time = datetime.datetime.now()

rekog_client = boto3.client("rekognition")  # use aws cli to configure api keys
rekog_max_labels = 123
rekog_min_conf = 50.0
rekog_sampling_interval = 30  # in seconds

camera = cv2.VideoCapture(0)


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            img_bytes = bytearray(buffer)
            process_image(img_bytes)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


def process_image(img_bytes):
    global previous_request_time
    if (current_time() - previous_request_time).seconds > rekog_sampling_interval:
        previous_request_time = current_time()
        app.logger.warning(f"Image Processing Called at {current_time()}")
        rekognize_objects(img_bytes)
        rekognize_celebs(img_bytes)


def rekognize_objects(img_bytes):
    response = rekog_client.detect_labels(
        Image={"Bytes": img_bytes},
        MaxLabels=rekog_max_labels,
        MinConfidence=rekog_min_conf,
    )
    try:
        response = pattern.sub('"', str(response))
        response_json = json.loads(response)
        labels, confidence = [], []

        for label in response_json["Labels"]:
            labels.append(label["Name"])
            confidence.append(label["Confidence"])

        labels_confidence_array = [
            {"Label": l, "Confidence": c} for l, c in zip(labels, confidence)
        ]
        object_rekognition_event(json.dumps(labels_confidence_array))
    except:
        on_error_event("Unable to parse image!")


def rekognize_celebs(img_bytes):
    response = rekog_client.recognize_celebrities(Image={"Bytes": img_bytes})
    try:
        response = pattern.sub('"', str(response))
        response_json = json.loads(response)
        celebrities = []

        for celebrity in response_json["CelebrityFaces"]:
            celebrities.append(celebrity["Name"])

        celebrity_rekognition_event(json.dumps(celebrities))
    except:
        on_error_event("Unable to parse image!")


def current_time():
    return datetime.datetime.now()


@app.route("/capture_video")
def capture_video():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    return render_template("index.html", async_mode=socketio.async_mode)


def object_rekognition_event(data):
    socketio.emit("object_rekognition_event", {"data": data}, callback=ack, namespace="/test")


def celebrity_rekognition_event(data):
    socketio.emit("celebrity_rekognition_event", {"data": data}, callback=ack, namespace="/test")


def on_error_event(message):
    socketio.emit("on_error_event", {"error": message}, callback=ack, namespace="/test")


@socketio.on("connected_event", namespace="/test")
def connected_event(data):
    app.logger.warning("Connected To Server: " + str(data))


def ack():
    app.logger.warning("Message Received!")


if __name__ == "__main__":
    app.run(debug=True)
    app.run(host="0.0.0.0")
    socketio.run(app)
