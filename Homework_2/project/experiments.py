import optimization
import oracles
import os
from libsvmdata import fetch_libsvm
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from collections import Counter


def experiment_1():
    np.random.seed(7412)
    figure = plt.figure(figsize=(7, 5))
    ax_1 = figure.add_subplot(121)
    ax_2 = figure.add_subplot(122)
    grid = np.array(range(3, 120))

    def get_A(n, k):
        data = np.concatenate((np.random.uniform(1, k, n - 2), np.array([1, k])), axis=0)
        return sp.diags(data, 0)

    def T(k, n):
        A = get_A(n, k)
        b = np.random.uniform(1, 20, n)
        x_0 = np.random.uniform(1, 20, n)
        matvec = lambda v: A @ v
        x_k, message, history = optimization.conjugate_gradients(matvec, b, x_0, trace=True)
        return len(history['residual_norm']), message

    for idx, color in enumerate(['green', 'red', 'blue', 'purple']):
        n = 10 ** (idx + 1)
        all_results = []
        for i in range(10):
            result, message = np.vectorize(T)(grid, n)
            all_results.append(result)
            ax_1.plot(grid, result, color=color, label=f'n = 10^{idx + 1}')
            ax_1.title.set_text('Абсолютное количество итераций')

            print(f'{idx + 1}/4: {i + 1}/10 - {Counter(message)}')

        ax_2.plot(grid, np.mean(all_results, axis=0), color=color, label=f'n = 10^{idx + 1}')
        ax_2.title.set_text('Среднее количество итераций')

    ax_1.set_xlabel('k')
    ax_2.set_xlabel('k')
    ax_1.set_ylabel('T(n, k)')
    ax_2.set_ylabel('T(n, k)')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_1.legend(by_label.values(), by_label.keys())
    ax_2.legend()
    plt.savefig('experiment_1/Tkn.png')


def experiment_2():
    np.random.seed(7412)

    A, b = fetch_libsvm('news20.binary')
    lamda = 1 / b.shape[0]
    x_0 = np.zeros(A.shape[1])

    logistic_regression = oracles.create_log_reg_oracle(A, b, lamda)
    denominator = np.square(np.linalg.norm(logistic_regression.grad(x_0)))

    figure = plt.figure(figsize=(7, 5))
    ax_1 = figure.add_subplot(121)
    ax_2 = figure.add_subplot(122)
    figure.suptitle('news20.binary')
    color_map = plt.get_cmap("tab10").colors

    def get_ratio(grad_norm):
        return np.log(np.square(grad_norm) / denominator)

    for idx, memory_size in enumerate([0, 1, 5, 10, 50, 100]):
        x_k, message, history = optimization.lbfgs(logistic_regression, x_0, trace=True, memory_size=memory_size)
        print(f"Memory_size: {memory_size}, message: {message}")

        r_k = np.vectorize(get_ratio)(history['grad_norm'])
        ax_1.plot(range(len(history['time'])), r_k, color=color_map[idx], label=f'l={memory_size}')
        ax_2.plot(history['time'], r_k, color=color_map[idx], label=f'l={memory_size}')

    ax_1.set_xlabel('Iterations')
    ax_1.set_ylabel('ln(r_k)')
    ax_1.set_title('Зависимость относительного квадрата \n нормы градиента против номера итерации', fontsize=8)
    ax_1.legend()

    ax_2.set_xlabel('Seconds')
    ax_2.set_ylabel('ln(r_k)')
    ax_2.set_title('Зависимость относительного квадрата нормы \n градиента против реального времени работы', fontsize=8)
    ax_2.legend()

    plt.savefig(f'experiment_2/news20.binary.png')


def experiment_3():
    np.random.seed(7412)
    methods = [optimization.hessian_free_newton, optimization.lbfgs, optimization.gradient_descent]
    methods_name = ['Newton hessian-free', 'L-BFGS', 'Gradient descent']
    datasets = ['w8a', 'gisette', 'real-sim', 'news20.binary', 'rcv1.binary']
    color_map = plt.get_cmap("tab10").colors

    def get_ratio(grad_norm, denominator):
        return np.log(np.square(grad_norm) / denominator)

    for data in datasets:
        A, b = fetch_libsvm(data)
        lamda = 1 / b.shape[0]
        x_0 = np.zeros(A.shape[1])
        logistic_regression = oracles.create_log_reg_oracle(A, b, lamda)
        denominator = np.square(np.linalg.norm(logistic_regression.grad(x_0)))

        figure = plt.figure(figsize=(12, 5))
        ax_1 = figure.add_subplot(131)
        ax_2 = figure.add_subplot(132)
        ax_3 = figure.add_subplot(133)
        figure.suptitle(f'{data}')

        for idx, (method, method_name) in enumerate(zip(methods, methods_name)):
            x_k, message, history = method(logistic_regression, x_0, trace=True)
            print(f"Data: {data}, method: {method_name}, message: {message}")

            ax_1.plot(range(len(history['time'])), history['func'], color=color_map[idx], label=method_name)
            ax_2.plot(history['time'], history['func'], color=color_map[idx], label=method_name)
            r_k = np.vectorize(get_ratio)(history['grad_norm'], denominator)
            ax_3.plot(history['time'], r_k, color=color_map[idx], label=method_name)

        ax_1.set_xlabel('Iterations')
        ax_1.set_ylabel('F(x)')
        ax_1.set_title('Зависимость значения функции \n против номера итерации метода', fontsize=8)
        ax_1.legend()

        ax_2.set_xlabel('Seconds')
        ax_2.set_ylabel('F(x)')
        ax_2.set_title('Зависимость значения функции \n против реального времени работы', fontsize=8)
        ax_2.legend()

        ax_3.set_xlabel('Seconds')
        ax_3.set_ylabel('ln(r_k)')
        ax_3.set_title('Зависимость относительного квадрата нормы \n градиента против реального времени работы', fontsize=8)
        ax_3.legend()

        plt.savefig(f'experiment_3/{data}.png')


def main():
    for directory in ['experiment_1', 'experiment_2', 'experiment_3']:
        if not os.path.exists(directory):
            os.makedirs(directory)

    experiment_1()
    experiment_2()
    experiment_3()


if __name__ == '__main__':
    main()
