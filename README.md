# EdgeAI-Vision

EdgeAI-Vision is an experimental edge-vision pipeline built around a
MobileNetV3-Small image classifier. It converts a pretrained PyTorch model into
ONNX, optionally runs INT8 static quantization, measures performance, and
streams predictions over local video files.

## Pipeline

1. [models/base_model.py](/Users/abhik/Downloads/EdgeAI-Vision/models/base_model.py)
   loads MobileNetV3-Small pretrained on ImageNet.
2. [quantization/export_onnx.py](/Users/abhik/Downloads/EdgeAI-Vision/quantization/export_onnx.py)
   exports the model to a portable ONNX graph.
3. [quantization/quantize.py](/Users/abhik/Downloads/EdgeAI-Vision/quantization/quantize.py)
   prepares the model for INT8 static quantization.
4. [inference/realtime_vision.py](/Users/abhik/Downloads/EdgeAI-Vision/inference/realtime_vision.py)
   preprocesses frames, runs ONNX Runtime, stabilizes predictions, and displays
   FPS and labels.
5. [benchmarks/benchmark.py](/Users/abhik/Downloads/EdgeAI-Vision/benchmarks/benchmark.py)
   compares PyTorch and ONNX Runtime latency, throughput, and file size.
6. [deployment/deploy_aihub.py](/Users/abhik/Downloads/EdgeAI-Vision/deployment/deploy_aihub.py)
   compiles and profiles the model for a Qualcomm device through AI Hub.

## Simple explanation

### Why MobileNetV3-Small?

MobileNetV3-Small is a convolutional neural network trained to recognize 1,000
common ImageNet object categories. It is designed to use fewer operations and
less memory than large models, making it a reasonable starting point for phones
and embedded devices. The project uses this model as an image classifier, not as
a detector that draws a separate box around each object.

### What PyTorch and ONNX mean

PyTorch is used to download the pretrained architecture and weights, test it,
and export it. ONNX is an exchange format for neural-network graphs. ONNX
Runtime reads that graph and performs inference without requiring the full
PyTorch training stack.

### Model tensor shape

The input shape is `[1, 3, 224, 224]`:

- `1`: batch size, the number of images processed together
- `3`: red, green, and blue channels
- `224 x 224`: resized frame height and width

### Frame preprocessing

OpenCV reads BGR video frames and resizes them to `224 x 224`. The project
changes BGR to RGB, converts pixel values from `0-255` to approximately `0-1`,
and rearranges channels from `[height, width, channel]` to
`[channel, height, width]`, which is the layout MobileNet expects.

### Softmax, logits, and confidence

The model first produces logits, which are raw scores for 1,000 classes. Softmax
turns those scores into probabilities that generally add up to 1. The class with
the largest probability is the top prediction, and that probability is presented
as confidence.

### FPS and latency

Latency is time per inference, usually measured in milliseconds. FPS (frames
per second) measures throughput. It is possible for FPS and latency to disagree
because video decoding, frame rendering, preprocessing, and batching also affect
end-to-end FPS.

### Quantization

Static INT8 quantization stores weights using 8-bit integers instead of 32-bit
floating-point numbers. Calibration passes representative input through the
model so PyTorch can record activation ranges and choose scale/zero-point
values. The QNNPACK backend is selected because it targets ARM CPUs commonly
found in mobile devices.

INT8 can reduce memory and speed up supported CPU workloads, but it can also
reduce accuracy. It should always be validated against representative real data.
The current random calibration input is only a functional demonstration; it is
not production calibration.

### Benchmarking

A warm-up is important because caches and runtime initialization can make the
first runs unusually slow. The benchmark then averages many runs. In its current
form it compares the same model under PyTorch and ONNX Runtime, while model
accuracy should be checked separately.

### Qualcomm AI Hub deployment

TorchScript tracing converts the dynamic Python model operations into a serial,
saved graph. Qualcomm AI Hub compiles that graph for the chosen Snapdragon
device, targeting the Hexagon NPU when supported. A separate profiling job
returns execution information such as latency and hardware utilization.

## AddressSanitizer

AddressSanitizer (ASan) is a compiler-injected memory-error detector for native
C and C++ programs. It detects problems such as out-of-bounds reads/writes, use
after free, double free, and certain stack issues.

Most application code here is Python, and ONNX Runtime, OpenCV, NumPy, and
PyTorch are prebuilt native libraries. Running ordinary Python under a normal
compiler does **not** rebuild those dependencies with ASan, so it would not
provide meaningful ASan coverage. This project therefore includes a small native
C++ target as an ASan demonstration and a place to add instrumented native code.

Run it on macOS with Apple Clang:

```bash
./native/run_asan.sh
```

The script compiles `native/asan_smoke.cpp` with `-fsanitize=address` and runs it
under ASan. A clean run means ASan detected no instrumented memory errors in the
smoke target. For production native work, the entire native executable or library
that you want to protect, plus its native dependencies, should be built with the
same sanitizer configuration.

## Known limitations and interview caveats

- The SSL certificate verification override in `models/base_model.py` is for a
  local download environment and weakens HTTPS security. It should be removed
  once the certificate setup is fixed.
- Missing label downloads currently fall back to generic class names.
- Quantization calibration uses random tensors and should use a representative
  real-image dataset before production deployment.
- `quantization/quantize.py` calculates output paths relative to the current
  working directory. Run the project scripts from the repository root.
- Quantized state is saved with `torch.save(...)`; loading INT8 model weights
  requires the matching quantized model structure.
- On-demand outputs and camera input are not the same as actual edge-hardware
  validation. Final claims should include the target device and measured build.
