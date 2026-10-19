import cv2
import time
import numpy as np
import onnxruntime as ort

def preprocess_frame(frame, input_shape=(224, 224)):
    """
    Preprocesses an OpenCV BGR frame into a normalized tensor matching 
    the model's expected input format.
    """
    # Resize frame to model input dimensions
    resized = cv2.resize(frame, input_shape)
    
    # Convert BGR (OpenCV default) to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] range and transpose to CHW format (Channels, Height, Width)
    img_normalized = rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    
    # Add batch dimension: [1, 3, H, W]
    input_tensor = np.expand_dims(img_chw, axis=0)
    return input_tensor

def main():
    # Path to your exported ONNX model
    model_path = "../weights/mobilenet_v3.onnx"
    
    print(f"[-] Loading ONNX runtime session for {model_path}...")
    # Initialize ONNX Runtime execution provider (CPU / CUDA / CoreML)
    providers = ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(model_path, providers=providers)
    
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # Initialize video capture (0 for default webcam, or pass a video path like 'sample.mp4')
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[!] Error: Could not open video stream or webcam.")
        return

    print("[+] Starting video stream loop. Press 'q' to exit.")
    
    prev_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame.")
            break

        # 1. Preprocess the live frame
        input_tensor = preprocess_frame(frame)

        # 2. Run Inference
        start_inference = time.time()
        outputs = ort_session.run([output_name], {input_name: input_tensor})
        inference_time = (time.time() - start_inference) * 1000.0 # in ms

        # Get predicted class index (assuming ImageNet classification output for MobileNet)
        probabilities = outputs[0][0]
        predicted_class_id = np.argmax(probabilities)

        # 3. Calculate Live FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time

        # 4. Overlay metrics onto the video frame for visualization
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Inference: {inference_time:.1f} ms", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Class ID: {predicted_class_id}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the live stream window
        cv2.imshow("EdgeAI-Vision Real-Time Stream", frame)

        # Break loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("[+] Video stream closed successfully.")

if __name__ == "__main__":
    main()