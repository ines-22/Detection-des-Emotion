import cv2
import sys
import time
import numpy as np
from decimal import Decimal
from model_utils import define_model, model_weights
import tensorflow as tf
from tensorflow import keras
import pyrebase


config = {
    "apiKey": "yourApiKey",
    "authDomain": "yourAuthDomain",
    "databaseURL": "yourDatabaseURL",
    "projectId": "yourProjectId",
    "storageBucket": "yourStorageBucket",
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()


def main():
    model = define_model()
    model = model_weights(model)
    print('Model loaded')

    result = np.array((1, 7))
    faceCascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_frontalface_default.xml')
    EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 640)
    video_capture.set(4, 480)

    emotion_statistics = {emotion: 0 for emotion in EMOTIONS}

    time_interval = 10  
    start_time = time.time()

    captured_emotion_statistics = None

    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x - 10, y - 70), (x + w + 20, y + h + 40), (15, 175, 61), 4)

            color_img = frame[y - 90:y + h + 70, x - 50:x + w + 50]
            if color_img.size != 0:
                color_img_resized = cv2.resize(color_img, (48, 48))
                color_img_resized = cv2.cvtColor(color_img_resized, cv2.COLOR_BGR2GRAY)
                color_img_resized = np.expand_dims(color_img_resized, axis=0)
                color_img_resized = np.expand_dims(color_img_resized, axis=-1)

                cv2.putText(frame, f"Next save in: {time_interval - int(time.time() - start_time)} seconds", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if time.time() - start_time >= time_interval:
                    result = model.predict(color_img_resized)
                    start_time = time.time()
                    total_sum = np.sum(result[0])

                    figure = frame[y - 90:y + h + 70, x - 50:x + w + 50]
                    cv2.imwrite('detected_figure.jpg', figure)

                    emotion_statistics = {emotion : float(round(Decimal(result[0][index] / total_sum * 100), 2))
                                          for index, emotion in enumerate(EMOTIONS)}
                    
                    captured_emotion_statistics = emotion_statistics  

                    for index, (emotion, percentage) in enumerate(emotion_statistics.items()):
                        text = f"{emotion}:  {emotion}    :    {percentage}%"
                        cv2.putText(frame, text, (10, 80 + index * 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (7, 109, 16), 2)
                    db.child("emotion_data").set(emotion_statistics)

                    return captured_emotion_statistics  
            break

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    captured_stats = main()
    print("Captured Emotion Statistics:", captured_stats)

