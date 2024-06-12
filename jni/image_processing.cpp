#include <iostream>
#include <fstream>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <vector>
#include <cmath>
#include <thread>
#include <sched.h>
#include <filesystem>
#include <cstring>
#include <unistd.h>
#include <sys/syscall.h>
#include <future>

namespace fs = std::filesystem;

void readDataFromFile(const std::string& filename, float* data, int dataSize) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
        return;
    }
    for (int i = 0; i < dataSize; ++i) {
        if (!(file >> data[i])) {
            std::cerr << "Error reading data from file at index " << i << std::endl;
            return;
        }
    }
    file.close();
}

void maxpool2d(float* input_data, int input_channels, int input_height, int input_width,
               int pool_size, int stride, float* output_data) {
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;
    int total_iterations = input_channels * output_height * output_width;

    for (int index = 0; index < total_iterations; index++) {
        int c = index / (output_height * output_width);
        int h = (index / output_width) % output_height;
        int w = index % output_width;

        float max_val = -FLT_MAX;
        for (int p = 0; p < pool_size * pool_size; p++) {
            int ph = p / pool_size;
            int pw = p % pool_size;

            int input_h = h * stride + ph;
            int input_w = w * stride + pw;
            if (input_h < input_height && input_w < input_width) {
                int input_index = c * (input_height * input_width) + input_h * input_width + input_w;
                max_val = std::max(max_val, input_data[input_index]);
            }
        }
        int output_index = c * (output_height * output_width) + h * output_width + w;
        output_data[output_index] = max_val;
    }
}

void conv2d(float* input_data, int image_input_channels, int input_height, int input_width,
            float* weight_data, int weight_output_channels, int weight_input_channels, int weight_height, int weight_width,
            float* bias_data, int kernel_size, int stride, int padding, 
            bool relu, float* output_data) {
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    for (int index = 0; index < weight_output_channels * output_height * output_width; index++) {
        int out_channel = index / (output_height * output_width);
        int y = (index / output_width) % output_height;
        int x = index % output_width;

        float sum = 0.0f;
        for (int in_channel = 0; in_channel < weight_input_channels; ++in_channel) {
            for (int ky = 0; ky < weight_height; ++ky) {
                int image_y = y * stride + ky - padding;
                for (int kx = 0; kx < weight_width; ++kx) {
                    int image_x = x * stride + kx - padding;
                    if (image_y >= 0 && image_y < input_height && image_x >= 0 && image_x < input_width) {
                        int input_index = (in_channel * input_height + image_y) * input_width + image_x;
                        int weight_index = (((out_channel * weight_input_channels + in_channel) * weight_height + ky) * weight_width + kx);
                        sum += input_data[input_index] * weight_data[weight_index];
                    }
                }
            }
        }
        if (bias_data) {
            sum += bias_data[out_channel];
        }
        if (relu && sum < 0) {
            sum = 0.0f;
        }
        output_data[(out_channel * output_height + y) * output_width + x] = sum;
    }
}

void linearLayer(float* input_data, float* weights, float* bias, float* output_data, int input_size, int output_size) {
    for (int i = 0; i < output_size; ++i) {
        float sum = 0;
        for (int j = 0; j < input_size; ++j) {
            sum += input_data[j] * weights[i * input_size + j];
        }
        sum += bias[i];
        output_data[i] = sum;
    }
}

void setCpuAffinity(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pid_t tid = syscall(SYS_gettid); // Get the thread ID
    int result = sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        std::cerr << "Error setting CPU affinity: " << strerror(errno) << std::endl;
    }
}

double benchmarkLayer(const std::string& layer_name, std::function<void()> layer_func, const std::vector<int>& threads, bool print_layer_timings) {
    std::vector<std::thread> thread_pool;
    auto start = std::chrono::high_resolution_clock::now();
    for (int t : threads) {
        thread_pool.emplace_back([&, t]() {
            setCpuAffinity(t);
            layer_func();
        });
    }
    for (auto& t : thread_pool) {
        t.join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    if (print_layer_timings) {
        std::cout << layer_name << " on threads ";
        for (int t : threads) {
            std::cout << t << (t != threads.back() ? ", " : "");
        }
        std::cout << " took " << duration.count() << " seconds" << std::endl;
    }
    return duration.count();
}

void processInput(const std::string& inputFile, const std::vector<std::vector<int>>& thread_groups, bool print_layer_timings, bool print_linear_layer) {
    // Network parameters
    const int input_channels = 3;
    const int input_height = 32;
    const int input_width = 32;
    const int num_classes = 10;
    const std::vector<std::string> class_labels = {
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    };

    // Load input data
    int input_size = input_channels * input_height * input_width;
    std::vector<float> input_data(input_size);
    readDataFromFile(inputFile, input_data.data(), input_size);

    // Conv1
    const int conv1_out_channels = 64;
    const int conv1_kernel_size = 3;
    const int conv1_stride = 1;
    const int conv1_padding = 1;
    std::vector<float> conv1_weight(conv1_out_channels * input_channels * conv1_kernel_size * conv1_kernel_size);
    std::vector<float> conv1_bias(conv1_out_channels);
    readDataFromFile("data/conv1_weight.txt", conv1_weight.data(), conv1_weight.size());
    readDataFromFile("data/conv1_bias.txt", conv1_bias.data(), conv1_bias.size());
    std::vector<float> conv1_output(conv1_out_channels * input_height * input_width);

    // MaxPool1
    const int pool1_size = 2;
    const int pool1_stride = 2;
    const int pool1_output_height = (input_height - pool1_size) / pool1_stride + 1;
    const int pool1_output_width = (input_width - pool1_size) / pool1_stride + 1;
    std::vector<float> pool1_output(conv1_out_channels * pool1_output_height * pool1_output_width);

    // Conv2
    const int conv2_out_channels = 192;
    const int conv2_kernel_size = 3;
    const int conv2_stride = 1;
    const int conv2_padding = 1;
    std::vector<float> conv2_weight(conv2_out_channels * conv1_out_channels * conv2_kernel_size * conv2_kernel_size);
    std::vector<float> conv2_bias(conv2_out_channels);
    readDataFromFile("data/conv2_weight.txt", conv2_weight.data(), conv2_weight.size());
    readDataFromFile("data/conv2_bias.txt", conv2_bias.data(), conv2_bias.size());
    const int conv2_output_height = pool1_output_height;
    const int conv2_output_width = pool1_output_width;
    std::vector<float> conv2_output(conv2_out_channels * conv2_output_height * conv2_output_width);

    // MaxPool2
    const int pool2_size = 2;
    const int pool2_stride = 2;
    const int pool2_output_height = (conv2_output_height - pool2_size) / pool2_stride + 1;
    const int pool2_output_width = (conv2_output_width - pool2_size) / pool2_stride + 1;
    std::vector<float> pool2_output(conv2_out_channels * pool2_output_height * pool2_output_width);

    // Conv3
    const int conv3_out_channels = 384;
    const int conv3_kernel_size = 3;
    const int conv3_stride = 1;
    const int conv3_padding = 1;
    std::vector<float> conv3_weight(conv3_out_channels * conv2_out_channels * conv3_kernel_size * conv3_kernel_size);
    std::vector<float> conv3_bias(conv3_out_channels);
    readDataFromFile("data/conv3_weight.txt", conv3_weight.data(), conv3_weight.size());
    readDataFromFile("data/conv3_bias.txt", conv3_bias.data(), conv3_bias.size());
    const int conv3_output_height = pool2_output_height;
    const int conv3_output_width = pool2_output_width;
    std::vector<float> conv3_output(conv3_out_channels * conv3_output_height * conv3_output_width);

    // Conv4
    const int conv4_out_channels = 256;
    const int conv4_kernel_size = 3;
    const int conv4_stride = 1;
    const int conv4_padding = 1;
    std::vector<float> conv4_weight(conv4_out_channels * conv3_out_channels * conv4_kernel_size * conv4_kernel_size);
    std::vector<float> conv4_bias(conv4_out_channels);
    readDataFromFile("data/conv4_weight.txt", conv4_weight.data(), conv4_weight.size());
    readDataFromFile("data/conv4_bias.txt", conv4_bias.data(), conv4_bias.size());
    const int conv4_output_height = conv3_output_height;
    const int conv4_output_width = conv3_output_width;
    std::vector<float> conv4_output(conv4_out_channels * conv4_output_height * conv4_output_width);

    // Conv5
    const int conv5_out_channels = 256;
    const int conv5_kernel_size = 3;
    const int conv5_stride = 1;
    const int conv5_padding = 1;
    std::vector<float> conv5_weight(conv5_out_channels * conv4_out_channels * conv5_kernel_size * conv5_kernel_size);
    std::vector<float> conv5_bias(conv5_out_channels);
    readDataFromFile("data/conv5_weight.txt", conv5_weight.data(), conv5_weight.size());
    readDataFromFile("data/conv5_bias.txt", conv5_bias.data(), conv5_bias.size());
    const int conv5_output_height = conv4_output_height;
    const int conv5_output_width = conv4_output_width;
    std::vector<float> conv5_output(conv5_out_channels * conv5_output_height * conv5_output_width);

    // MaxPool3
    const int pool3_size = 2;
    const int pool3_stride = 2;
    const int pool3_output_height = (conv5_output_height - pool3_size) / pool3_stride + 1;
    const int pool3_output_width = (conv5_output_width - pool3_size) / pool3_stride + 1;
    std::vector<float> pool3_output(conv5_out_channels * pool3_output_height * pool3_output_width);

    // Linear layer
    const int linear_input_size = conv5_out_channels * pool3_output_height * pool3_output_width;
    const int linear_output_size = num_classes;
    std::vector<float> linear_weight(linear_output_size * linear_input_size);
    std::vector<float> linear_bias(linear_output_size);
    readDataFromFile("data/linear_weight.txt", linear_weight.data(), linear_weight.size());
    readDataFromFile("data/linear_bias.txt", linear_bias.data(), linear_bias.size());
    std::vector<float> linear_output(linear_output_size);

    // Benchmark each layer for each thread configuration
    for (const auto& threads : thread_groups) {
        double total_time = 0.0;

        total_time += benchmarkLayer("Conv1", [&]() {
            conv2d(input_data.data(), input_channels, input_height, input_width,
                   conv1_weight.data(), conv1_out_channels, input_channels, conv1_kernel_size, conv1_kernel_size,
                   conv1_bias.data(), conv1_kernel_size, conv1_stride, conv1_padding, true, conv1_output.data());
        }, threads, print_layer_timings);

        total_time += benchmarkLayer("MaxPool1", [&]() {
            maxpool2d(conv1_output.data(), conv1_out_channels, input_height, input_width, pool1_size, pool1_stride, pool1_output.data());
        }, threads, print_layer_timings);

        total_time += benchmarkLayer("Conv2", [&]() {
            conv2d(pool1_output.data(), conv1_out_channels, conv2_output_height, conv2_output_width,
                   conv2_weight.data(), conv2_out_channels, conv1_out_channels, conv2_kernel_size, conv2_kernel_size,
                   conv2_bias.data(), conv2_kernel_size, conv2_stride, conv2_padding, true, conv2_output.data());
        }, threads, print_layer_timings);

        total_time += benchmarkLayer("MaxPool2", [&]() {
            maxpool2d(conv2_output.data(), conv2_out_channels, conv2_output_height, conv2_output_width, pool2_size, pool2_stride, pool2_output.data());
        }, threads, print_layer_timings);

        total_time += benchmarkLayer("Conv3", [&]() {
            conv2d(pool2_output.data(), conv2_out_channels, conv3_output_height, conv3_output_width,
                   conv3_weight.data(), conv3_out_channels, conv2_out_channels, conv3_kernel_size, conv3_kernel_size,
                   conv3_bias.data(), conv3_kernel_size, conv3_stride, conv3_padding, true, conv3_output.data());
        }, threads, print_layer_timings);

        total_time += benchmarkLayer("Conv4", [&]() {
            conv2d(conv3_output.data(), conv3_out_channels, conv4_output_height, conv4_output_width,
                   conv4_weight.data(), conv4_out_channels, conv3_out_channels, conv4_kernel_size, conv4_kernel_size,
                   conv4_bias.data(), conv4_kernel_size, conv4_stride, conv4_padding, true, conv4_output.data());
        }, threads, print_layer_timings);

        total_time += benchmarkLayer("Conv5", [&]() {
            conv2d(conv4_output.data(), conv4_out_channels, conv5_output_height, conv5_output_width,
                   conv5_weight.data(), conv5_out_channels, conv4_out_channels, conv5_kernel_size, conv5_kernel_size,
                   conv5_bias.data(), conv5_kernel_size, conv5_stride, conv5_padding, true, conv5_output.data());
        }, threads, print_layer_timings);

        total_time += benchmarkLayer("MaxPool3", [&]() {
            maxpool2d(conv5_output.data(), conv5_out_channels, conv5_output_height, conv5_output_width, pool3_size, pool3_stride, pool3_output.data());
        }, threads, print_layer_timings);

        total_time += benchmarkLayer("LinearLayer", [&]() {
            linearLayer(pool3_output.data(), linear_weight.data(), linear_bias.data(), linear_output.data(), linear_input_size, linear_output_size);
        }, threads, print_layer_timings);

        // Output the total time for this thread configuration
        std::cout << "Total time for threads ";
        for (int t : threads) {
            std::cout << t << (t != threads.back() ? ", " : "");
        }
        std::cout << " is " << total_time << " seconds" << std::endl;

        // Output the classified result
        float max_value = linear_output[0];
        int max_index = 0;
        for (int i = 0; i < linear_output_size; ++i) {
            if (linear_output[i] > max_value) {
                max_value = linear_output[i];
                max_index = i;
            }
        }
        std::cout << "Classified image: " << class_labels[max_index] << std::endl;

        // If the -linearlayer flag is set, print the class scores and percentages
        if (print_linear_layer) {
            std::cout << "Class scores:" << std::endl;
            float sum = 0;
            for (int i = 0; i < linear_output_size; ++i) {
                std::cout << "Class " << i << ": " << linear_output[i] << std::endl;
                sum += exp(linear_output[i]);
            }
            std::cout << "Class percentages:" << std::endl;
            for (int i = 0; i < linear_output_size; ++i) {
                float percentage = exp(linear_output[i]) / sum * 100;
                std::cout << "Class " << i << ": " << percentage << "%" << std::endl;
            }
        }
    }
}

int main(int argc, char** argv) {
    std::string input_dir = "input/";
    std::vector<std::vector<int>> thread_groups = {
        {0}, {0, 1}, {0, 1, 2}, {0, 1, 2, 3}, // Core 0 threads
        {4}, {4, 5},                         // Core 1 threads
        {6}, {6, 7}                          // Core 2 threads
    };

    bool print_layer_timings = false;
    bool print_linear_layer = false;
    bool use_full_cores = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-layers") == 0) {
            print_layer_timings = true;
        }
        if (std::strcmp(argv[i], "-linearlayer") == 0) {
            print_linear_layer = true;
        }
        if (std::strcmp(argv[i], "-fullcores") == 0) {
            use_full_cores = true;
        }
        // Add more flags here if needed in the future
    }

    if (use_full_cores) {
        thread_groups = {
            {0, 1, 2, 3},  // Core 0 full threads
            {4, 5},        // Core 1 full threads
            {6, 7}         // Core 2 full threads
        };
    }

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        std::cout << "Processing file: " << entry.path() << std::endl;
        processInput(entry.path().string(), thread_groups, print_layer_timings, print_linear_layer);
    }

    return 0;
}

