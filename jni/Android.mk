LOCAL_PATH := $(call my-dir)

# Build matrix_benchmark module
include $(CLEAR_VARS)
LOCAL_MODULE := matrix_benchmark
LOCAL_SRC_FILES := matrix_multiplication.cpp
LOCAL_SHARED_LIBRARIES := libc++_shared
include $(BUILD_EXECUTABLE)

# Prebuilt shared library for matrix_benchmark
include $(CLEAR_VARS)
LOCAL_MODULE := libc++_shared
LOCAL_SRC_FILES := $(NDK_HOME)/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/arm-linux-androideabi/libc++_shared.so
include $(PREBUILT_SHARED_LIBRARY)

# Build image_processing module
include $(CLEAR_VARS)
LOCAL_MODULE := image_processing
LOCAL_SRC_FILES := image_processing.cpp
LOCAL_SHARED_LIBRARIES := libc++_shared
include $(BUILD_EXECUTABLE)

# Build task_pipeline module
include $(CLEAR_VARS)
LOCAL_MODULE := task_pipeline
LOCAL_SRC_FILES := task_pipeline.cpp
LOCAL_SHARED_LIBRARIES := libc++_shared
include $(BUILD_EXECUTABLE)

