#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/stderr_reporter.h>

using namespace cv;
using namespace tflite;

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the webcam." << std::endl;
        return -1;
    }

    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("selfie_segmentation_landscape.tflite");
    if (!model) {
        std::cerr << "Failed to load model." << std::endl;
        return -1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to create interpreter." << std::endl;
        return -1;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return -1;
    }

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Failed to capture frame from webcam." << std::endl;
            break;
        }

        Mat resized_frame;
        resize(frame, resized_frame, Size(256, 144));

        resized_frame.convertTo(resized_frame, CV_32FC3, 1.0 / 255.0);

        // Input dimensions (float32[1,144,256,3])
        float* input_tensor = interpreter->typed_input_tensor<float>(0);
        memcpy(input_tensor, resized_frame.data, resized_frame.total() * resized_frame.elemSize());

        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Failed to invoke tflite interpreter." << std::endl;
            break;
        }

        // Get output (float32[1,144,256,1])
        const float* output_tensor = interpreter->typed_output_tensor<float>(0);
        Mat segmentation_mask(144, 256, CV_32FC1, (void*)output_tensor);

        Mat segmentation_mask_resized;
        resize(segmentation_mask, segmentation_mask_resized, frame.size());
        
        imshow("Webcam Frame", frame);
        imshow("Segmentation Mask", segmentation_mask_resized);
        
        // Exit on 'q' key press
        if (waitKey(1) == 'q') {
            break;
        }
    }

    // Release the webcam and close windows
    cap.release();
    destroyAllWindows();

    return 0;
}
