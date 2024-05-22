#include <cedar/image.hpp>
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>  // 包含 xtensor 库的随机数头文件

/**
 * @brief 打印 xarray 对象
 *
 * 将给定的 xarray 对象打印到标准输出流中，包括其形状和内容。
 *
 * @param arr xarray 对象
 */
void print_xarray(const xt::xarray<double>& arr) {
    std::cout << "Shape: ";
    for (size_t i = 0; i < arr.shape().size(); ++i) {
        std::cout << arr.shape()[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Content: " << std::endl << arr << std::endl;
}

/**
 * @brief 计算两个矩形的 IoU 值
 *
 * 根据给定的两个矩形坐标数组 a 和 b，计算它们之间的 IoU 值。
 *
 * @param a 矩形 a 的坐标数组
 * @param b 矩形 b 的坐标数组
 *
 * @return 两个矩形的 IoU 值
 */
double iou(const xt::xarray<double>& a, const xt::xarray<double>& b) {
    // 计算 a 的面积
    double area_a = (a(2) - a(0)) * (a(3) - a(1));
    // 计算 b 的面积
    double area_b = (b(2) - b(0)) * (b(3) - b(1));

    // 获取 IoU 的左上角 x 坐标
    double iou_x1 = max(a(0), b(0));
    // 获取 IoU 的左上角 y 坐标
    double iou_y1 = max(a(1), b(1));
    // 获取 IoU 的右下角 x 坐标
    double iou_x2 = min(a(2), b(2));
    // 获取 IoU 的右下角 y 坐标
    double iou_y2 = min(a(3), b(3));

    // 计算 IoU 的宽度
    double iou_w = iou_x2 - iou_x1;
    // 计算 IoU 的高度
    double iou_h = iou_y2 - iou_y1;

    // 检查交集面积是否非正
    if (iou_w < 0 || iou_h < 0) {
        return 0.0;
    }

    // 计算 IoU 的面积
    double area_iou = iou_w * iou_h;
    // 计算 IoU 与总面积的重叠比率
    double iou_ratio = area_iou / (area_a + area_b - area_iou);

    return iou_ratio;
}

/**
 * @brief 裁剪图像中的边界框
 *
 * 根据给定的 ground truth 边界框和参数，对图像进行裁剪，并在图像上绘制裁剪框。
 *
 * @param img 输入图像
 * @param gt ground truth 边界框
 * @param Crop_N 裁剪数量，默认为 200
 * @param L 裁剪框的边长，默认为 60
 * @param th IoU 阈值，默认为 0.5
 *
 * @return 裁剪后的图像
 */
cv::Mat crop_bbox(cv::Mat img, const xt::xarray<double>& gt, int Crop_N = 200, int L = 60, double th = 0.5) {
    // 获取图像尺寸
    int H = img.rows;
    int W = img.cols;

    // 每个裁剪
    for (int i = 0; i < Crop_N; ++i) {
        // 获取裁剪 bounding box 的左上角 x 坐标
        int x1 = xt::random::randint<int>({1}, 0, W - L)(0);
        // 获取裁剪 bounding box 的左上角 y 坐标
        int y1 = xt::random::randint<int>({1}, 0, H - L)(0);
        // 获取裁剪 bounding box 的右下角 x 坐标
        int x2 = x1 + L;
        // 获取裁剪 bounding box 的右下角 y 坐标
        int y2 = y1 + L;

        // 创建裁剪 bounding box
        xt::xarray<double> crop = {static_cast<double>(x1), static_cast<double>(y1), static_cast<double>(x2), static_cast<double>(y2)};

        // 计算裁剪框和 gt 之间的 IoU
        double _iou = iou(gt, crop);

        // 分配标签
        if (_iou >= th) {
            cout << "生成红色: " << _iou << endl;
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 1);
        } else {
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 1);
        }
    }

    return img;
}

int main() {
    // 读取图片
    Mat image = loadAndCheckImage("./imori_1.jpg");

    // [x1, y1, x2, y2]
    xt::xarray<double> a = {{50, 50, 150, 150}};
    xt::xarray<double> b = {{60, 60, 170, 160}};

    cout << iou(a, b) << endl;

    // gt = np.array((47, 41, 129, 103), dtype=np.float32)
    xt::xarray<double> gt = {{47, 41, 129, 103}};

    Mat img = crop_bbox(image, gt, 200);

    saveImage("./out.jpg", img);

    return 0;
}
