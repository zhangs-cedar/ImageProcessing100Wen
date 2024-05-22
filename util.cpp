#include <algorithm> // 算法库，包含了各种常用算法，如排序、查找等
#include <cmath>            // 数学库，提供了数学函数和常量
#include <ctime>            // 时间库，用于时间的获取和处理
#include <filesystem>       // 文件系统库，用于文件和目录的操作
#include <fstream>          // 文件流，用于文件的读写操作
#include <iostream>         // 输入输出流，用于标准输入输出操作
#include <list>             // 链表容器，用于存储链式结构数据
#include <map>              // 映射容器，用于存储键值对
#include <opencv2/core.hpp> // OpenCV核心功能模块
#include <opencv2/highgui.hpp> // OpenCV图形用户界面模块，用于图像显示和用户交互
#include <set>       // 集合容器，用于存储不重复的元素集合
#include <string>    // 字符串类，用于字符串操作
#include <vector>    // 向量容器，用于存储动态数组
using namespace std; // 使用标准命名空间，避免频繁使用std::

void printMatProperties(const cv::Mat &mat) {
  std::cout << "Rows: " << mat.rows << std::endl;
  std::cout << "Cols: " << mat.cols << std::endl;
  std::cout << "Data type: " << mat.type() << std::endl;
  std::cout << "Channels: " << mat.channels() << std::endl;
  std::cout << "Strides: " << mat.step << std::endl;
  std::cout << "Size: " << mat.size() << std::endl;
  std::cout << "Depth: " << mat.depth() << std::endl;
  std::cout << "Channel order: " << mat.channels() << std::endl;
  std::cout << "Continuous: " << (mat.isContinuous() ? "Yes" : "No")
            << std::endl;
  std::cout << "Data pointer: " << reinterpret_cast<void *>(mat.data)
            << std::endl;
  std::cout << "ROI: " << mat.size() << std::endl;
}