#!/bin/bash

# Transfer executables
adb push libs/armeabi-v7a/matrix_benchmark /data/local/tmp/
adb push libs/armeabi-v7a/image_processing /data/local/tmp/

# Transfer data files
adb push data/conv1_weight.txt /data/local/tmp/data/
adb push data/conv1_bias.txt /data/local/tmp/data/
adb push data/conv2_weight.txt /data/local/tmp/data/
adb push data/conv2_bias.txt /data/local/tmp/data/
adb push data/conv3_weight.txt /data/local/tmp/data/
adb push data/conv3_bias.txt /data/local/tmp/data/
adb push data/conv4_weight.txt /data/local/tmp/data/
adb push data/conv4_bias.txt /data/local/tmp/data/
adb push data/conv5_weight.txt /data/local/tmp/data/
adb push data/conv5_bias.txt /data/local/tmp/data/
adb push data/linear_weight.txt /data/local/tmp/data/
adb push data/linear_bias.txt /data/local/tmp/data/

# Create input directory
adb shell mkdir /data/local/tmp/input

# Transfer input files
for file in input/*; do
  adb push "$file" /data/local/tmp/input/
done

