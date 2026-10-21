import os
import time
import cv2
import numpy as np
import onnxruntime as ort


class cap:
    """Small webcam abstraction that exposes a stable capture interface."""

    def __init__(self, camera_index=0, backend=None):
        self._capture = None
        self._camera_index = camera_index
        self._backend = backend
        self._open(camera_index, backend)

    def _open(self, camera_index, backend=None):
        if backend is not None:
            self._capture = cv2.VideoCapture(camera_index, backend)
            if not self._capture.isOpened() and backend != cv2.CAP_ANY:
                self._capture = cv2.VideoCapture(camera_index)
        else:
            self._capture = cv2.VideoCapture(camera_index)

    def isOpened(self):
        return self._capture is not None and self._capture.isOpened()

    def read(self):
        if self._capture is None:
            return False, None
        return self._capture.read()

    def release(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def set(self, prop, value):
        if self._capture is None:
            return False
        return self._capture.set(prop, value)

    def get(self, prop):
        if self._capture is None:
            return 0
        return self._capture.get(prop)

    def close(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def load_imagenet_labels():
    """
    Returns a dummy list or downloads ImageNet labels. 
    For production, we map index to human-readable labels.
    """
    # Simple placeholder categories; or you can load standard ImageNet 1000 labels.
    return {i: f"Class {i}" for i in range(1000)}

def run_realtime_inference():
    onnx_path = "weights/mobilenet_v3.onnx"
    if not os.path.exists(onnx_path):
        print(f"[!] Error: ONNX model not found at {onnx_path}. Run export_onnx.py first.")
        return

    print("[-] Initializing ONNX Runtime Inference Session...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Point this to any MP4/AVI video file in your project workspace
    video_path = "assets/test_video.mp4"
    print(f"[-] Opening video stream from file: {video_path}...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[!] Error: Could not open video file at {video_path}.")
        return

    print("[+] Stream active! Press 'q' in the video window to quit.")
    prev_time = time.time()
    
    # Initialize a clean canvas for a moving shape test
    x_pos = 50
    while True:
        # Create a clean white canvas [480, 640, 3]
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 240
        
        # Draw a moving blue box to simulate a real object moving in frame
        x_pos = (x_pos + 5) % 580
        cv2.rectangle(frame, (x_pos, 200), (x_pos + 60, 260), (255, 100, 50), -1)
        cv2.putText(frame, "Tracking Test Object", (x_pos - 10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

        # 1. Preprocess frame for MobileNetV3 [1, 3, 224, 224]
        resized = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_normalized = rgb_frame.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        input_tensor = np.expand_dims(img_transposed, axis=0)

        # 2. Run Inference via ONNX Runtime
        outputs = session.run(None, {input_name: input_tensor})
        logits = outputs[0]
        predicted_class = np.argmax(logits, axis=1)[0]

        # 3. Calculate Real-Time FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        prev_time = current_time

        # 4. Draw Overlays
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 0), 2)
        cv2.putText(frame, f"Pred ID: {predicted_class}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)

        cv2.imshow("EdgeAI-Vision Clean Stream", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("[+] Stream closed successfully.")

if __name__ == "__main__":
    run_realtime_inference()