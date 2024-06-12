#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <sched.h>
#include <unistd.h>
#include <cstring>
#include <numeric> // Include this header for std::accumulate

using namespace std;
using namespace chrono;

// Function to perform matrix multiplication and addition using a specified range of rows
void matrixMultiplyAddPartial(const vector<vector<float>>& A, const vector<vector<float>>& B, const vector<vector<float>>& C, vector<vector<float>>& D, int startRow, int endRow, double& mul_time, double& add_time) {
    int n = A.size();
    auto mul_start = high_resolution_clock::now();
    vector<vector<float>> temp(n, vector<float>(n, 0.0));
    
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < n; ++j) {
            temp[i][j] = 0;
            for (int k = 0; k < n; ++k) {
                temp[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    auto mul_end = high_resolution_clock::now();
    mul_time = duration<double>(mul_end - mul_start).count();

    auto add_start = high_resolution_clock::now();
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < n; ++j) {
            D[i][j] = temp[i][j] + C[i][j];
        }
    }
    auto add_end = high_resolution_clock::now();
    add_time = duration<double>(add_end - add_start).count();
}

// Function to perform matrix multiplication and addition using a thread pool
void matrixMultiplyAddThreadPool(const vector<vector<float>>& A, const vector<vector<float>>& B, const vector<vector<float>>& C, vector<vector<float>>& D, int numThreads, double& mul_time, double& add_time) {
    int n = A.size();
    vector<thread> threads;
    vector<double> mul_times(numThreads, 0.0);
    vector<double> add_times(numThreads, 0.0);
    int rowsPerThread = n / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i == numThreads - 1) ? n : startRow + rowsPerThread;
        threads.emplace_back(matrixMultiplyAddPartial, cref(A), cref(B), cref(C), ref(D), startRow, endRow, ref(mul_times[i]), ref(add_times[i]));
    }

    for (auto& th : threads) {
        th.join();
    }

    mul_time = accumulate(mul_times.begin(), mul_times.end(), 0.0);
    add_time = accumulate(add_times.begin(), add_times.end(), 0.0);
}

// Function to benchmark matrix multiplication and addition using a thread pool
void benchmark(int size, int numThreads, double& mul_time, double& add_time, double& total_time) {
    vector<vector<float>> A(size, vector<float>(size, 1.0));
    vector<vector<float>> B(size, vector<float>(size, 1.0));
    vector<vector<float>> C(size, vector<float>(size, 1.0));
    vector<vector<float>> D(size, vector<float>(size, 0.0));

    auto start = high_resolution_clock::now();
    matrixMultiplyAddThreadPool(A, B, C, D, numThreads, mul_time, add_time);
    auto end = high_resolution_clock::now();

    total_time = duration<double>(end - start).count();
}

// Function to set CPU affinity
void setCpuAffinity(const vector<int>& cores) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int core : cores) {
        CPU_SET(core, &cpuset);
    }

    int result = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        cerr << "Error setting CPU affinity: " << strerror(errno) << endl;
    }
}

void benchmarkCoreType(const vector<int>& cores, const string& coreType) {
    vector<int> sizes = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};

    for (int numThreads = 1; numThreads <= cores.size(); ++numThreads) {
        cout << "Benchmarking " << coreType << " cores with " << numThreads << " threads" << endl;
        vector<int> activeCores(cores.begin(), cores.begin() + numThreads);
        setCpuAffinity(activeCores);
        for (int size : sizes) {
            double mul_time = 0.0, add_time = 0.0, total_time = 0.0;
            benchmark(size, numThreads, mul_time, add_time, total_time);
            cout << "Cores: ";
            for (int core : activeCores) {
                cout << core << " ";
            }
            cout << ", Size: " << size << "x" << size << ", Threads: " << numThreads << ", Mul Time: " << mul_time << " seconds, Add Time: " << add_time << " seconds, Total Time: " << total_time << " seconds" << endl;
        }
    }
}

int main() {
    vector<int> efficiency_cores = {0, 1, 2, 3}; // Cortex-A55
    vector<int> performance_cores = {4, 5}; // Cortex-A78
    vector<int> prime_cores = {6, 7}; // Cortex-X1

    benchmarkCoreType(efficiency_cores, "high-efficiency (Cortex-A55)");
    benchmarkCoreType(performance_cores, "high-performance (Cortex-A78)");
    benchmarkCoreType(prime_cores, "prime (Cortex-X1)");

    return 0;
}

