#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

class NN {
   private:
    xt::xarray<double> w1, b1, w2, b2, wout, bout;
    xt::xarray<double> z2, z3, out;
    double learning_rate;
    // Sigmoid函数作为类内方法
    xt::xarray<double> sigmoid(const xt::xarray<double>& x) { return 1 / (1 + xt::exp(-x)); }

   public:
    /**
     * @brief 构造函数，初始化神经网络
     *
     * 初始化神经网络的权重和偏置，包括输入层到隐藏层、隐藏层到隐藏层以及隐藏层到输出层的权重和偏置。
     *
     * @param input_dim 输入层维度，默认为2
     * @param hidden_dim 第一个隐藏层维度，默认为64
     * @param hidden_dim2 第二个隐藏层维度，默认为64
     * @param output_dim 输出层维度，默认为1
     * @param lr 学习率，默认为0.1
     */
    NN(int input_dim = 2, int hidden_dim = 64, int hidden_dim2 = 64, int output_dim = 1, double lr = 0.1) {
        // 初始化权重和偏置
        w1 = xt::random::randn<double>({input_dim, hidden_dim});
        b1 = xt::random::randn<double>({hidden_dim});
        w2 = xt::random::randn<double>({hidden_dim, hidden_dim2});
        b2 = xt::random::randn<double>({hidden_dim2});
        wout = xt::random::randn<double>({hidden_dim2, output_dim});
        bout = xt::random::randn<double>({output_dim});
        learning_rate = lr;
    }

    /**
     * @brief 前向传播
     *
     * 根据给定的输入数组 x，执行前向传播计算并返回输出结果。
     *
     * @param x 输入数组
     *
     * @return 输出结果数组
     */
    xt::xarray<double> forward(const xt::xarray<double>& x) {
        // 前向传播
        z2 = sigmoid(xt::linalg::dot(x, w1) + b1);
        z3 = sigmoid(xt::linalg::dot(z2, w2) + b2);
        out = sigmoid(xt::linalg::dot(z3, wout) + bout);

        return out;
    }

    /**
     * @brief 训练函数
     *
     * 使用给定的输入和目标进行训练，更新权重和偏置。
     *
     * @param x 输入数据
     * @param t 目标数据
     */
    void train(const xt::xarray<double>& x, const xt::xarray<double>& t) {
        // 训练

        out = forward(x);
        xt::xarray<double> En = (out - t) * out * (1 - out);

        // 更新输出层权重和偏置
        wout -= learning_rate * xt::linalg::dot(xt::transpose(z3), En);
        bout -= learning_rate * xt::sum(En, {0});

        // 计算误差传播到第二层的梯度
        xt::xarray<double> grad_u2 = xt::linalg::dot(En, xt::transpose(wout)) * z3 * (1 - z3);

        // 更新第二层权重和偏置
        w2 -= learning_rate * xt::linalg::dot(xt::transpose(z2), grad_u2);
        b2 -= learning_rate * xt::sum(grad_u2, {0});

        // 计算误差传播到第一层的梯度
        xt::xarray<double> grad_u1 = xt::linalg::dot(grad_u2, xt::transpose(w2)) * z2 * (1 - z2);

        // 更新第一层权重和偏置
        w1 -= learning_rate * xt::linalg::dot(xt::transpose(x), grad_u1);
        b1 -= learning_rate * xt::sum(grad_u1, {0});
    }
};

/**
 * @brief 打印 xt::xarray<double> 类型的数组
 *
 * 打印给定 xt::xarray<double> 类型数组的形状和内容。
 *
 * @param arr xt::xarray<double> 类型的数组
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
 * @brief 主函数
 *
 * 主函数用于训练神经网络模型并测试其预测结果。
 *
 * @return 返回值为0，表示程序正常结束。
 */
int main() {
    NN nn;
    xt::xarray<double> x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    xt::xarray<double> _t = {0, 1, 1, 0};
    xt::xarray<double> t = _t.reshape({4, 1});

    // 训练模型
    for (int i = 0; i < 1000; ++i) {
        nn.train(x, t);
    }

    // 测试模型
    xt::xarray<double> x_test = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    xt::xarray<double> pred = nn.forward(x_test);
    std::cout << "Predictions:" << std::endl;
    std::cout << pred << std::endl;

    return 0;
}