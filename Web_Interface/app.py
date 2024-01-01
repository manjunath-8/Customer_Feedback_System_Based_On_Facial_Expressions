from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import datetime

app = Flask(__name__)
socketio = SocketIO(app)

def load_emotion_model(model_path='my_model.h5'):
    return load_model(model_path)

def detect_faces(frame, face_detector):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))

    return faces

# Define emotion_dict globally
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def get_emotion_text(frame, face_detector, emotion_model):
    faces = detect_faces(frame, face_detector)

    if faces:
        (x, y, w, h) = faces[0]
        roi_color_frame = frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(cv2.resize(roi_color_frame, (48, 48)), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion_text = emotion_dict[maxindex]
    else:
        emotion_text = "No face detected"
        
    return emotion_text

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    emotion_model = load_emotion_model()

    deploy_prototxt_path = 'face/deploy.prototxt'
    caffemodel_path = 'face/res10_300x300_ssd_iter_140000.caffemodel'
    face_detector = cv2.dnn.readNetFromCaffe(deploy_prototxt_path, caffemodel_path)

    while True:
        success, frame = cap.read()
        if not success:
            break

        try:
            emotion_text = get_emotion_text(frame, face_detector, emotion_model)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Emit emotion updates through WebSocket
            socketio.emit('emotion_update', emotion_text)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error processing frame: {e}")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()

    emotion = data.get('emotion', '')
    time = data.get('time', '')

    feedback_info = f'Emotion: {emotion}, Time: {time}\n'

    file_path = 'feedback/feedback.txt'
    with open(file_path, 'a') as file:
        file.write(feedback_info)

    return jsonify({'message': 'Feedback submitted successfully'})
    
if __name__ == "__main__":
    socketio.run(app, debug=True)

