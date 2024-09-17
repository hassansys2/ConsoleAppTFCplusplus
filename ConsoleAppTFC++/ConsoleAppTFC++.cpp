#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/stderr_reporter.h>
#include <deque>

using namespace cv;
using namespace tflite;

int main() {
    VideoCapture cap(0);
    std::unique_ptr<tflite::Interpreter> interpreter;

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
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to create interpreter." << std::endl;
        return -1;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return -1;
    }

    // Load a new background image
    Mat background = imread("background_image.jpg");
    if (background.empty()) {
        std::cerr << "Error: Could not load background image." << std::endl;
        return -1;
    }

    int n = 5;              // Number of previous masks to average
    float threshold_value = 0.5;  // Initial threshold value
    std::deque<Mat> mask_buffer;  // Buffer to store previous masks

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Failed to capture frame from webcam." << std::endl;
            break;
        }

        resize(background, background, frame.size());
        
        // input size (256x144 for selfie segmentation landscape)
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

        // Add the current mask to the buffer
        mask_buffer.push_back(segmentation_mask_resized);

        // Ensure the buffer does not exceed the size `n`
        if (mask_buffer.size() > n) {
            mask_buffer.pop_front();
        }

        // Calculate the average mask
        Mat avg_mask = Mat::zeros(segmentation_mask_resized.size(), CV_32FC1);
        for (const Mat& m : mask_buffer) {
            avg_mask += m;
        }
        avg_mask /= static_cast<float>(mask_buffer.size());

        // Threshold the averaged mask to create a binary mask
        Mat binary_mask;
        threshold(avg_mask, binary_mask, threshold_value, 1, THRESH_BINARY);

        // Apply Gaussian blur to smooth the edges of the mask
        Mat smooth_mask;
        GaussianBlur(binary_mask, smooth_mask, Size(15, 15), 0);  // Adjust kernel size for more or less smoothing

        // Convert mask to 3 channels and the same type as the frame
        Mat smooth_mask_3ch;
        Mat mask_channels[] = { smooth_mask, smooth_mask, smooth_mask };
        merge(mask_channels, 3, smooth_mask_3ch);
        smooth_mask_3ch.convertTo(smooth_mask_3ch, CV_32FC3);

        // Convert frame and background to float
        Mat frame_float, background_float;
        frame.convertTo(frame_float, CV_32FC3, 1.0 / 255.0);
        background.convertTo(background_float, CV_32FC3, 1.0 / 255.0);

        // Replace background using the smoothed mask
        Mat foreground, background_overlay;
        multiply(frame_float, smooth_mask_3ch, foreground);
        multiply(background_float, Scalar(1, 1, 1) - smooth_mask_3ch, background_overlay);

        // Combine foreground and background
        Mat output;
        add(foreground, background_overlay, output);

        // Convert the output to 8-bit for display
        output.convertTo(output, CV_8UC3, 255.0);

        // Display the results
        imshow("Webcam Frame", frame);
        imshow("Background Replaced", output);

        // Handle key inputs
        char key = waitKey(1);
        if (key == 'q') {
            break;
        }
        else if (key == '+') {
            n = std::min(n + 1, 20);  // Increase `n`, cap at 20
            std::cout << "Increased number of masks to average: " << n << std::endl;
        }
        else if (key == '-') {
            n = std::max(n - 1, 1);  // Decrease `n`, minimum is 1
            std::cout << "Decreased number of masks to average: " << n << std::endl;
        }
        else if (key == 'u') {
            threshold_value = std::min(threshold_value + 0.05f, 1.0f);  // Increase threshold, cap at 1.0
            std::cout << "Increased threshold value: " << threshold_value << std::endl;
        }
        else if (key == 'd') {
            threshold_value = std::max(threshold_value - 0.05f, 0.0f);  // Decrease threshold, minimum is 0.0
            std::cout << "Decreased threshold value: " << threshold_value << std::endl;
        }
    }

    // Release the webcam and close windows
    cap.release();
    destroyAllWindows();

    return 0;
}
