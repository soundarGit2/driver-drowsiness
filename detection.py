import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

model = load_model('D:/driver drowisness detection/model/drowsiness_model_combined.h5', compile=False)

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_drowsiness(video_path, seq_len=10, img_size=64):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return None, None, None

    # Create output folder if not exist
    output_folder = os.path.join('static', 'output')
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, 'output.mp4')

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (frame_width, frame_height))

    frame_buffer = []
    preds = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (img_size, img_size))
        norm = resized / 255.0
        frame_buffer.append(norm)

        if len(frame_buffer) < seq_len:
            continue
        elif len(frame_buffer) > seq_len:
            frame_buffer.pop(0)

        input_seq = np.expand_dims(frame_buffer, axis=0)
        prediction = model.predict(input_seq, verbose=0)[0][0]
        preds.append(prediction)

        if prediction < 0.5:
            label = "Awake"
            color = (0, 0, 255)
        else:
            label = "Drowsy"
            color = (0, 255, 0)

        display_frame = frame.copy()
        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(display_frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            cv2.putText(display_frame, f"{label} (Confidence: {prediction:.2f})", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)



        out.write(display_frame)

    cap.release()
    out.release()

    # Final classification and confidence score
    avg_pred = np.mean(preds)
    final_status = "Awake" if avg_pred < 0.5 else "Drowsy"
    confidence = round(avg_pred, 2)  # Score between 0 and 1, rounded to 2 decimal places

    return out_path, final_status, confidence

