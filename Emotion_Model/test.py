import cv2
import numpy as np
from tensorflow.keras.models import load_model

def load_emotion_model(model_path='model/my_model.h5'):
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

def main():
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Load the model
    emotion_model = load_emotion_model()
    print("Loaded emotion model from disk")

    # Adjust the path based on your project structure
    deploy_prototxt_path = 'face/deploy.prototxt'
    caffemodel_path = 'face/res10_300x300_ssd_iter_140000.caffemodel'

    #    Load the face detector
    face_detector = cv2.dnn.readNetFromCaffe(deploy_prototxt_path, caffemodel_path)


    # Start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))

        if not ret:
            break

        # Detect faces
        faces = detect_faces(frame, face_detector)

        # Process the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_color_frame = frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(cv2.resize(roi_color_frame, (48, 48)), 0)

            # Predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Emotion Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

