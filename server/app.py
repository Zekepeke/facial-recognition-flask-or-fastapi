from flask import Flask, request, jsonify
import cv2, numpy as np, json, pathlib, datetime
import mediapipe as mp
from keras_facenet import FaceNet


app = Flask(__name__)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)