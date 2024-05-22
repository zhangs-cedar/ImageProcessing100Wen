#!/bin/bash


rm -rf out.jpg

# 检查是否传递了源文件名作为参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 <source_file>"
    exit 1
fi

# 获取源文件名和输出文件名
source_file="$1"
output_file="build/a.out" #"${source_file%.*}"
rm -rf output_file

# 编译源文件
echo "Compiling $source_file..."
#g++ "$source_file" -g -o "$output_file" `pkg-config --cflags --libs opencv4`
g++ "$source_file" -g -O2 -o "$output_file" -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc


# 检查编译是否成功
if [ $? -eq 0 ]; then
    # 如果编译成功，运行生成的可执行文件
    echo "Running $output_file..."
    ./"$output_file"
else
    echo "Compilation failed."
fi


# "g++ /workspace/c++_whl_learning/l02/temp02/01_imwrite.cpp -o my_program `pkg-config --cflags --libs opencv4`" 