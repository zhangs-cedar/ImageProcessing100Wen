#include "cedar/image.hpp"
#include <iostream>
// #include <opencv2/opencv.hpp>
// #include <xtensor/xadapt.hpp>
// #include <xtensor/xarray.hpp>
// #include <xtensor/xmath.hpp>
// #include <xtensor/xview.hpp>

// using namespace cv;
// using namespace std;

// 加载图像并检查是否成功加载
// Mat loadAndCheckImage(const string &filename)
// {
//     Mat image = imread(filename);
//     if (image.empty())
//     {
//         cerr << "无法加载图像: " << filename << endl;
//         exit(EXIT_FAILURE);
//     }
//     return image;
// }

// // 交换图像通道
// Mat swapChannels(const Mat &image)
// {
//     Mat swapped_image;
//     cvtColor(image, swapped_image, COLOR_BGR2RGB);
//     return swapped_image;
// }

// // 灰度化
// Mat BGR2GRAY(const cv::Mat &image)
// {
//     cv::Mat gray;
//     cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
//     return gray;
// }

// // 二值化
// Mat thresholding(const Mat &image, int threshold_value)
// {
//     Mat binary_image;
//     threshold(image, binary_image, threshold_value, 255, THRESH_BINARY);
//     return binary_image;
// }

// // 二值化（Otsu 方法）
// Mat thresholdingOtsu(const Mat &image)
// {
//     Mat binary_image;
//     threshold(image, binary_image, 0, 255, THRESH_BINARY | THRESH_OTSU);
//     return binary_image;
// }

// // 使用OpenCV函数实现平均池化
// Mat averagePooling(const cv::Mat &img, int pool_size)
// {
//     int height = img.rows;         // 图像高度
//     int width = img.cols;          // 图像宽度
//     int channels = img.channels(); // 图像通道数

//     // 准备输出图像
//     cv::Mat out = cv::Mat::zeros(height / pool_size, width / pool_size,
//     img.type());

//     // 对输入图像进行遍历，以池化窗口大小为步长进行池化操作
//     for (int y = 0; y < height; y += pool_size)
//     {
//         for (int x = 0; x < width; x += pool_size)
//         {
//             for (int c = 0; c < channels; c++)
//             {
//                 // 计算池化窗口内像素值的平均值
//                 cv::Rect roi(x, y, pool_size, pool_size);
//                 cv::Scalar mean = cv::mean(img(roi));
//                 // 将平均值赋值给输出图像中相应位置的像素
//                 out.at<cv::Vec3b>(y / pool_size, x / pool_size)[c] =
//                 mean.val[c];
//             }
//         }
//     }

//     return out;
// }

// // 最大池化
// void maxPooling(const Mat &mat)
// {
//     // auto img_arr = xt::adapt(img.data, img.total() * img.channels(),
//     xt::no_ownership()); xt::xarray<float> ximage = xt::adapt((float
//     *)mat.data, mat.cols * mat.rows, xt::no_ownership(),
//     std::vector<std::size_t>{mat.rows, mat.cols});

//     // 打印转换后的 XTensor 的形状
//     cout << "Shape of adapted XTensor: " << ximage.shape()[0] << "x" <<
//     ximage.shape()[1] << "x" << ximage.shape()[2] << endl;
// }

// // 显示图像
// void showImage(const string &window_name, const Mat &image)
// {
//     imshow(window_name, image);
//     waitKey(0);
// }

// // 保存图像
// void saveImage(const string &filename, const Mat &image)
// {
//     imwrite(filename, image);
// }

int main() {
  // 读取图像
  // cedar;
  Mat image = loadAndCheckImage("imori.jpg");

  // 灰度化
  // Mat gray_image = BGR2GRAY(image);

  // // 二值化
  // int threshold_value = 128; // 设置阈值
  // Mat binary_image = thresholding(gray_image, threshold_value);

  // 二值化（Otsu 方法）
  // Mat otsu_binary_image = thresholdingOtsu(gray_image);

  // 池化
  // int pool_size = 2;
  // Mat averagePoolimage = averagePooling(image, pool_size);

  // 最大池化

  // 显示并保存二值化结果
  // showImage("Binary Image", binary_image);
  // maxPooling(image);
  saveImage("out.jpg", image);

  return 0;
}
