#include <cedar/image.hpp>
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

/**
 * @brief 打印 xarray 数组
 *
 * 将传入的 xt::xarray<double> 类型的数组 arr 打印输出其形状和内容。
 *
 * @param arr 要打印的数组
 */
void print_xarray(const xt::xarray<double>& arr) {
    std::cout << "Shape: ";
    for (size_t i = 0; i < arr.shape().size(); ++i) {
        std::cout << arr.shape()[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Content: " << std::endl << arr << std::endl;
}

int main() {
    xt::xarray<double> x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    xt::xarray<double> t = {0, 1, 1, 0};
    xt::xarray<double> t_reshaped = t.reshape({4, 1});

    print_xarray(x);
    print_xarray(t);
    print_xarray(t_reshaped);
}