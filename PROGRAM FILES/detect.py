# detect.py
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = 'mask_detector.h5'
IMG_SIZE = (128, 128)

def preprocess_face(face_bgr):
    # Convert BGR to RGB, resize, scale
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, IMG_SIZE)
    face_norm = face_resized.astype('float32') / 255.0
    return np.expand_dims(face_norm, axis=0)

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue
            inp = preprocess_face(face)
            prob = float(model.predict(inp, verbose=0)[0][0])

            # Adjust threshold
            if prob >= 0.9:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)

            text = f"{label} ({prob:.2f})"
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow('Mask Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
