#include <iostream>

#include <opencv2/opencv.hpp>

#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

cv::Mat xarray_to_mat_elementwise(xt::xarray<float> xarr) {
    int ndims = xarr.dimension();
    assert(ndims == 2 && "can only convert 2d xarrays");
    int nrows = xarr.shape()[0];
    int ncols = xarr.shape()[1];
    cv::Mat mat(nrows, ncols, CV_32FC1);
    for (int rr = 0; rr < nrows; rr++) {
        for (int cc = 0; cc < ncols; cc++) {
            mat.at<float>(rr, cc) = xarr(rr, cc);
        }
    }
    return mat;
}

xt::xarray<float> mat_to_xarray_elementwise(cv::Mat mat) {
    int ndims = mat.dims;
    assert(ndims == 2 && "can only convert 2d xarrays");
    int nrows = mat.rows;
    int ncols = mat.cols;
    xt::xarray<float> xarr = xt::empty<float>({nrows, ncols});
    for (int rr = 0; rr < nrows; rr++) {
        for (int cc = 0; cc < ncols; cc++) {
            xarr(rr, cc) = mat.at<float>(rr, cc);
        }
    }
    return xarr;
}

int main() {
    int nrows = 2, ncols = 3;
    float data[150];
    for (int i = 0; i < nrows * ncols; i++) {
        data[i] = .1 * i;
    }

    cv::Mat mat(nrows, ncols, CV_32FC1, data, 0);
    std::cout << "mat:\n" << mat << std::endl;

    // 将指向数据的指针转换为 xarray 对象
    // 数据类型为 float，数据指针为 data
    xt::xarray<float> xarr =
        // 通过 xt::adapt() 函数创建 xarray 对象，将指向数据的指针转换为 xarray
        // 对象
        xt::adapt(
            // 数据指针，指向需要转换为 xarray 对象的数据
            (float*)data,
            // 数据的总大小，即数据的行数乘以列数
            // 使用 static_cast<long unsigned int> 进行类型转换，以避免警告
            static_cast<long unsigned int>(nrows) * static_cast<long unsigned int>(ncols),
            // 指定数据的所有权
            // 在这里使用 xt::no_ownership() 表示不需要释放指针指向的数据
            xt::no_ownership(),
            // 指定数据的形状
            // 使用 std::vector<std::size_t> 表示数据的形状，其中包括行数和列数
            std::vector<std::size_t>{static_cast<long unsigned int>(nrows), static_cast<long unsigned int>(ncols)});

    cv::Mat mat2_ew = xarray_to_mat_elementwise(xarr);
    std::cout << "mat2_ew (from xt::xarray):\n" << mat2_ew << std::endl;

    xt::xarray<float> xarr2_ew = mat_to_xarray_elementwise(mat);
    std::cout << "xarr2_ew (from cv::mat):\n" << xarr2_ew << std::endl;

    return 0;
}