import os
import time
from collections import deque
import cv2
import numpy as np
import onnxruntime as ort

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def run_realtime_inference():
    onnx_path = "weights/mobilenet_v3.onnx"
    if not os.path.exists(onnx_path):
        print(f"[!] Error: ONNX model not found at {onnx_path}. Run export_onnx.py first.")
        return

    print("[-] Initializing ONNX Runtime Inference Session...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    video_path = "assets/dog.mp4"
    print(f"[-] Opening video stream from file: {video_path}...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[!] Error: Could not open video file at {video_path}.")
        return

    print("[+] Stream active! Press 'q' in the video window to quit.")
    prev_time = time.time()

    pred_history = deque(maxlen=20)
    last_stable_prediction = 0
    confidence_threshold = 0.20  # Model must be at least 20% confident to switch

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 1. Preprocess frame
        resized = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_normalized = rgb_frame.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        input_tensor = np.expand_dims(img_transposed, axis=0)

        # 2. Run Inference & Compute Probabilities
        outputs = session.run(None, {input_name: input_tensor})
        logits = outputs[0]
        probabilities = softmax(logits)
        
        top_pred = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0][top_pred]

        # 3. Smart Filtering: Only accept prediction if confidence is high enough
        if confidence >= confidence_threshold:
            pred_history.append(top_pred)
            # Require a strong consensus in our history buffer
            if len(pred_history) >= 10:
                most_common = max(set(pred_history), key=pred_history.count)
                if pred_history.count(most_common) >= 5:  # At least 5 frames agreed
                    last_stable_prediction = most_common

        # 4. Calculate Real-Time FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        prev_time = current_time

        # 5. Draw Overlays (Showing both ID and confidence percentage)
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Pred ID: {last_stable_prediction} ({confidence*100:.1f}%)", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("EdgeAI-Vision Clean Stream", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[+] Stream closed successfully.")

if __name__ == "__main__":
    run_realtime_inference()