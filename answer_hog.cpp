#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>
#include <xtensor/xarray.hpp>

cv::Mat hog(const cv::Mat& gray) {
    int h = gray.rows;
    int w = gray.cols;
    int HH = h / 8;
    int HW = w / 8;

    // Magnitude and gradient
    cv::Mat padded;
    cv::copyMakeBorder(gray, padded, 1, 1, 1, 1, cv::BORDER_REFLECT);

    xt::xarray<float> gx = padded(cv::Rect(1, 0, w, h)).clone() - padded(cv::Rect(0, 0, w, h)).clone();
    xt::xarray<float> gy = padded(cv::Rect(0, 1, w, h)).clone() - padded(cv::Rect(0, 0, w, h)).clone();

    gx = xt::where(xt::equal(gx, 0), 0.000001, gx);

    xt::xarray<float> mag = xt::sqrt(gx * gx + gy * gy);
    xt::xarray<float> gra = xt::arctan2(gy, gx);

    gra = xt::where(xt::less(gra, 0), xt::pi<float>() / 2 + gra + xt::pi<float>() / 2, gra);

    // Gradient histogram
    xt::xarray<int> gra_n = xt::zeros<int>({h, w});

    float d = xt::pi<float>() / 9;
    for (int i = 0; i < 9; ++i) {
        auto idx = xt::where((gra >= d * i) && (gra <= d * (i + 1)));
        gra_n[idx] = i;
    }

    xt::xarray<float> Hist = xt::zeros<float>({HH, HW, 9});
    const int N = 8;
    for (int y = 0; y < HH; ++y) {
        for (int x = 0; x < HW; ++x) {
            for (int j = 0; j < N; ++j) {
                for (int i = 0; i < N; ++i) {
                    Hist(y, x, gra_n(y * 4 + j, x * 4 + i)) += mag(y * 4 + j, x * 4 + i);
                }
            }
        }
    }

    // Normalization
    const int C = 3;
    const float eps = 1;
    for (int y = 0; y < HH; ++y) {
        for (int x = 0; x < HW; ++x) {
            float norm = std::sqrt(xt::sum(xt::view(Hist, xt::range(std::max(y - 1, 0), std::min(y + 2, HH)),
                                                    xt::range(std::max(x - 1, 0), std::min(x + 2, HW)), xt::all()) *
                                           xt::view(Hist, xt::range(std::max(y - 1, 0), std::min(y + 2, HH)),
                                                    xt::range(std::max(x - 1, 0), std::min(x + 2, HW)), xt::all()))()[0] +
                                   eps);
            Hist(y, x, xt::all()) /= norm;
        }
    }

    return xt::cast<float>(Hist);
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
