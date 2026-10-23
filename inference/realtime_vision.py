import os
import time
from collections import deque
import urllib.request
import json
import cv2
import numpy as np
import onnxruntime as ort

def load_imagenet_labels():
    """Downloads or loads the official ImageNet class labels mapping."""
    labels_path = "weights/imagenet_classes.json"
    if not os.path.exists(labels_path):
        os.makedirs("weights", exist_ok=True)
        print("[-] Downloading ImageNet class labels mapping...")
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        try:
            urllib.request.urlretrieve(url, labels_path)
        except Exception as e:
            print(f"[!] Could not download labels automatically: {e}")
            return {i: f"Class {i}" for i in range(1000)}
            
    with open(labels_path, "r") as f:
        labels = json.load(f)
    return {i: label for i, label in enumerate(labels)}

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def get_available_videos():
    """Scans the assets folder and returns a list of valid video paths."""
    assets_dir = "assets"
    if not os.path.exists(assets_dir):
        return []
    valid_extensions = (".mp4", ".avi", ".mov", ".mkv")
    return [os.path.join(assets_dir, f) for f in os.listdir(assets_dir) if f.lower().endswith(valid_extensions)]

def run_realtime_inference():
    onnx_path = "weights/mobilenet_v3.onnx"
    if not os.path.exists(onnx_path):
        print(f"[!] Error: ONNX model not found at {onnx_path}. Run export_onnx.py first.")
        return

    # Load readable class names
    class_labels = load_imagenet_labels()

    print("[-] Initializing ONNX Runtime Inference Session...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Get list of videos in assets folder
    videos = get_available_videos()
    if not videos:
        print("[!] Error: No video files found in the 'assets' directory.")
        return

    # Print out the mapping so you know which number matches which video
    print("\n========================================================")
    print("🎬 AVAILABLE VIDEOS:")
    for idx, vpath in enumerate(videos, start=1):
        print(f"  [{idx}] {os.path.basename(vpath)}")
    print("========================================================\n")

    current_video_idx = 0
    video_path = videos[current_video_idx]

    print(f"[+] Opening video stream: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[!] Error: Could not open video file at {video_path}.")
        return

    print("========================================================")
    print("🎮 LIVE CONTROLS:")
    print("  • Press '1' to '9' : Jump DIRECTLY to that video")
    print("  • Press 'n'        : Cycle to the NEXT video")
    print("  • Press 'p'        : Cycle to the PREVIOUS video")
    print("  • Press 'q'        : Quit the application")
    print("========================================================\n")

    prev_time = time.time()
    pred_history = deque(maxlen=20)
    last_stable_prediction = 0
    confidence_threshold = 0.15

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

        # 3. Smart Filtering
        if confidence >= confidence_threshold:
            pred_history.append(top_pred)
            if len(pred_history) >= 10:
                most_common = max(set(pred_history), key=pred_history.count)
                if pred_history.count(most_common) >= 4:
                    last_stable_prediction = most_common

        object_name = class_labels.get(last_stable_prediction, f"Class {last_stable_prediction}")

        # 4. Calculate Real-Time FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        prev_time = current_time

        # 5. Draw Clean Overlays
        video_filename = os.path.basename(video_path)
        cv2.putText(frame, f"[{current_video_idx+1}] {video_filename} (n: next, p: prev, 1-{len(videos)}: jump)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {last_stable_prediction} | {object_name}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("EdgeAI-Vision Clean Stream", frame)

        # Handle Keyboard Inputs
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            # Sequential next video
            cap.release()
            current_video_idx = (current_video_idx + 1) % len(videos)
            video_path = videos[current_video_idx]
            print(f"\n[+] Switching live to next video [{current_video_idx+1}]: {video_path}")
            cap = cv2.VideoCapture(video_path)
            pred_history.clear()
        elif key == ord('p'):
            # Sequential previous video
            cap.release()
            current_video_idx = (current_video_idx - 1) % len(videos)
            video_path = videos[current_video_idx]
            print(f"\n[+] Switching live to previous video [{current_video_idx+1}]: {video_path}")
            cap = cv2.VideoCapture(video_path)
            pred_history.clear()
        elif ord('1') <= key <= ord('9'):
            # Direct jump based on number key pressed
            target_idx = key - ord('1')
            if target_idx < len(videos):
                cap.release()
                current_video_idx = target_idx
                video_path = videos[current_video_idx]
                print(f"\n[+] Directly switching to video [{current_video_idx+1}]: {video_path}")
                cap = cv2.VideoCapture(video_path)
                pred_history.clear()

    cap.release()
    cv2.destroyAllWindows()
    print("[+] Stream closed successfully.")

if __name__ == "__main__":
    run_realtime_inference()