#include <cedar/image.hpp>

// 定义高斯滤波函数
Mat gaussianBlur(const Mat& src, int kernel_size = 5, double sigma = 1.0) {
    // 对图像进行高斯滤波
    Mat dst;
    GaussianBlur(src, dst, Size(kernel_size, kernel_size), sigma);
    return dst;
}

Mat drawHistogram(const Mat& image) {
    // 计算直方图
    Mat hist;
    int histSize = 256;        // 直方图的 bin 数量
    float range[] = {0, 256};  // 像素值范围
    const float* histRange = {range};
    calcHist(&image, 1, nullptr, Mat(), hist, 1, &histSize, &histRange, true, false);

    // 创建直方图画布
    int histWidth = 512;
    int histHeight = 400;
    int binWidth = cvRound(static_cast<double>(histWidth) / histSize);
    Mat histImage(histHeight, histWidth, CV_8UC3, Scalar(255, 255, 255));

    // 归一化直方图
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    // 绘制直方图
    for (int i = 1; i < histSize; ++i) {
        line(histImage, Point(binWidth * (i - 1), histHeight - cvRound(hist.at<float>(i - 1))),
             Point(binWidth * (i), histHeight - cvRound(hist.at<float>(i))), Scalar(0, 0, 0), 2, 8, 0);
    }

    return histImage;
}

int main() {
    // 读取图像
    Mat image = loadAndCheckImage("imori.jpg");

    Mat gray = BGR2GRAY(image);

    Mat dst = gaussianBlur(image);

    Mat hist = drawHistogram(gray);
    saveImage("out.jpg", hist);

    return 0;
}
