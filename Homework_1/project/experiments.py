import optimization
import oracles
import os
# from libsvmdata import fetch_libsvm
import math
import random
import matplotlib.pyplot as plt
import plot_trajectory_2d
import numpy as np
import scipy.sparse as sp
from collections import Counter


def experiment_1():
    data = [{'A': np.array([[5, 1], [1, 1]]),
             'b': np.zeros(2)},

            {'A': np.array([[25, 2], [2, 1]]),
             'b': np.array([1, 1])}]

    for idx, params in enumerate(data):
        for x_0 in [np.array([3, 4]), np.array([1, 25])]:
            for method in ['Wolfe', 'Armijo', 'Constant']:
                plt.clf()
                oracle = oracles.QuadraticOracle(params['A'], params['b'])
                plot_trajectory_2d.plot_levels(oracle.func)
                x_k, message, history = optimization.gradient_descent(oracle, x_0, trace=True,
                                                                      line_search_options={'method': method, 'c': 0.01})
                plot_trajectory_2d.plot_trajectory(oracle.func, history['x'])
                plt.savefig(f'experiment_1/{x_0}--{method}--{idx + 1}.png')


def experiment_2():
    np.random.seed(7412)
    figure = plt.figure(figsize=(7, 5))
    ax_1 = figure.add_subplot(121)
    ax_2 = figure.add_subplot(122)

    def get_A(n, k):
        data = np.concatenate((np.random.uniform(1, k, n - 2), np.array([1, k])), axis=0)
        return sp.diags(data, 0)

    def T(k, n):
        oracle = oracles.QuadraticOracle(get_A(n, k), np.random.uniform(1, 10, n))
        x_k, message, history = optimization.gradient_descent(oracle, np.random.uniform(10, 30, n), trace=True)
        return len(history['grad_norm']), message

    for idx, color in enumerate(['green', 'red', 'blue', 'purple']):
        n = 10 ** (idx + 1)
        all_results = []
        for i in range(10):
            grid = np.array(range(3, 120))
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
    plt.savefig('experiment_2/Tkn.png')


def experiment_3():
    np.random.seed(7412)

    for data in ['w8a', 'gisette', 'real-sim']:
        A, b = fetch_libsvm(data)
        lamda = 1 / b.shape[0]
        histories = dict()
        logistic_regression = oracles.create_log_reg_oracle(A, b, lamda)
        figure = plt.figure(figsize=(7, 5))
        ax_1 = figure.add_subplot(121)
        ax_2 = figure.add_subplot(122)
        figure.suptitle(f'{data}')
        for method, method_name, color in zip([optimization.gradient_descent, optimization.newton],
                                              ['GD', 'Newton'],
                                              ['green', 'blue']):
            x_k, message, history = method(logistic_regression, np.zeros(A.shape[1]), trace=True, display=True)
            print(f"Method: {method_name}, message: {message}")
            histories[method_name] = history

        ax_1.plot(histories['GD']['time'], histories['GD']['func'], color='green', label='Gradient descent')
        ax_1.plot(histories['Newton']['time'], histories['Newton']['func'], color='blue', label='Newton method')
        ax_1.set_xlabel('Seconds')
        ax_1.set_ylabel('F(x)')
        ax_1.legend()

        denominator = np.square(np.linalg.norm(logistic_regression.grad(np.zeros(A.shape[1]))))
        r_k_gd = np.vectorize(lambda x: math.log(np.linalg.norm(x ** 2) / denominator))(histories['GD']['grad_norm'])
        r_k_new = np.vectorize(lambda x: math.log(np.linalg.norm(x ** 2) / denominator))(histories['Newton']['grad_norm'])
        ax_2.plot(histories['GD']['time'], r_k_gd, color='green', label='Gradient descent')
        ax_2.plot(histories['Newton']['time'], r_k_new, color='blue', label='Newton method')
        ax_2.set_xlabel('Seconds')
        ax_2.set_ylabel('ln(r_k)')
        ax_2.legend()
        plt.savefig(f'experiment_3/{data}.png')


def experiment_4():
    np.random.seed(237654)
    color_map = plt.get_cmap("tab10").colors
    n, m, k = 20, 20, 50

    A = np.concatenate((np.random.uniform(1, k, (n - 2, m)), np.array([[1 for _ in range(m)], [k for _ in range(m)]])))
    random.shuffle(A)
    b = np.sign(np.random.randn(n))
    lamda = 1 / b.shape[0]

    # Experiment with c.
    for method in ['Wolfe', 'Armijo', 'Constant']:
        figure = plt.figure(figsize=(7, 5))
        ax_1 = figure.add_subplot(121)
        ax_2 = figure.add_subplot(122)
        figure.suptitle(f'{method}')
        for c, color in zip(np.linspace(0.001, 0.999, num=5), color_map):
            logistic_regression = oracles.create_log_reg_oracle(A, b, lamda)
            x_k, message, history = optimization.gradient_descent(logistic_regression, np.zeros(A.shape[1]), trace=True, display=True,
                                                                  line_search_options={'method': method, 'c': c})
            ax_1.plot(range(len(history['time'])), history['func'], color=color, label=f'c = {round(c, 3)}')
            denominator = np.linalg.norm(logistic_regression.grad(np.zeros(A.shape[1]))) ** 2
            r_k_gd = np.vectorize(lambda x: math.log(np.linalg.norm(x ** 2) / denominator))(history['grad_norm'])
            ax_2.plot(range(len(history['time'])), r_k_gd, color=color, label=f'c = {round(c, 3)}')

        ax_1.set_xlabel('Iteration')
        ax_2.set_xlabel('Iteration')
        ax_1.set_ylabel('F(x)')
        ax_2.set_ylabel('ln(r_k)')
        ax_1.legend()
        ax_2.legend()
        plt.savefig(f'experiment_4/{method}-c.png')

    # Experiment with starting points.
    start_points = [random.randint(1, 10) * np.random.randn(m) + random.randint(1, 100) for _ in range(5)]
    for method in ['Wolfe', 'Armijo', 'Constant']:
        figure = plt.figure(figsize=(7, 5))
        ax_1 = figure.add_subplot(121)
        ax_2 = figure.add_subplot(122)
        figure.suptitle(f'{method}')
        for x_0, color in zip(start_points, color_map):
            logistic_regression = oracles.create_log_reg_oracle(A, b, lamda)
            x_k, message, history = optimization.gradient_descent(logistic_regression, x_0, trace=True, display=True,
                                                                  line_search_options={'method': method, 'c': 0.001})
            ax_1.plot(range(len(history['time'])), history['func'], color=color)
            denominator = np.linalg.norm(logistic_regression.grad(np.zeros(A.shape[1]))) ** 2
            r_k_gd = np.vectorize(lambda x: math.log(np.linalg.norm(x ** 2) / denominator))(history['grad_norm'])
            ax_2.plot(range(len(history['time'])), r_k_gd, color=color)

        ax_1.set_xlabel('Iteration')
        ax_2.set_xlabel('Iteration')
        ax_1.set_ylabel('F(x)')
        ax_2.set_ylabel('ln(r_k)')
        plt.savefig(f'experiment_4/{method}-start-point.png')

    # Plot all methods in one figure.
    figure = plt.figure(figsize=(7, 5))
    ax_1 = figure.add_subplot(121)
    ax_2 = figure.add_subplot(122)
    for method, color in zip(['Wolfe', 'Armijo', 'Constant'], color_map):
        logistic_regression = oracles.create_log_reg_oracle(A, b, lamda)
        x_k, message, history = optimization.gradient_descent(logistic_regression, np.zeros(A.shape[1]), trace=True, display=True,
                                                              line_search_options={'method': method, 'c': 0.001})
        ax_1.plot(range(len(history['time'])), history['func'], color=color, label=method)
        denominator = np.linalg.norm(logistic_regression.grad(np.zeros(A.shape[1]))) ** 2
        r_k_gd = np.vectorize(lambda x: math.log(np.linalg.norm(x ** 2) / denominator))(history['grad_norm'])
        ax_2.plot(range(len(history['time'])), r_k_gd, color=color, label=method)

    ax_1.set_xlabel('Iteration')
    ax_2.set_xlabel('Iteration')
    ax_1.set_ylabel('F(x)')
    ax_2.set_ylabel('ln(r_k)')
    ax_1.legend()
    ax_2.legend()
    plt.savefig(f'experiment_4/all.png')


def experiment_5():
    np.random.seed(676454)
    color_map = plt.get_cmap("tab10").colors
    n, m, k = 20, 20, 50

    A = np.concatenate((np.random.uniform(1, k, (n - 2, m)), np.array([[1 for _ in range(m)], [k for _ in range(m)]])))
    random.shuffle(A)
    b = np.sign(np.random.randn(n))
    lamda = 1 / b.shape[0]

    # Experiment with c.
    for method in ['Wolfe', 'Armijo', 'Constant']:
        figure = plt.figure(figsize=(7, 5))
        ax_1 = figure.add_subplot(121)
        ax_2 = figure.add_subplot(122)
        figure.suptitle(f'{method}')
        for c, color in zip(np.linspace(0.001, 1, num=5), color_map):
            logistic_regression = oracles.create_log_reg_oracle(A, b, lamda)
            x_k, message, history = optimization.newton(logistic_regression, np.zeros(A.shape[1]), trace=True, display=True,
                                                        line_search_options={'method': method, 'c': c})
            ax_1.plot(range(len(history['time'])), history['func'], color=color, label=f'c = {round(c, 3)}')
            denominator = np.linalg.norm(logistic_regression.grad(np.zeros(A.shape[1]))) ** 2
            r_k_gd = np.vectorize(lambda x: math.log(np.linalg.norm(x ** 2) / denominator))(history['grad_norm'])
            ax_2.plot(range(len(history['time'])), r_k_gd, color=color, label=f'c = {round(c, 3)}')

        ax_1.set_xlabel('Iteration')
        ax_2.set_xlabel('Iteration')
        ax_1.set_ylabel('F(x)')
        ax_2.set_ylabel('ln(r_k)')
        ax_1.legend()
        ax_2.legend()
        plt.savefig(f'experiment_5/{method}-c.png')

    # Experiment with starting points.
    start_points = [random.randint(1, 10) * np.random.randn(m) + random.randint(1, 100) for _ in range(5)]
    for method in ['Wolfe', 'Armijo', 'Constant']:
        figure = plt.figure(figsize=(7, 5))
        ax_1 = figure.add_subplot(121)
        ax_2 = figure.add_subplot(122)
        figure.suptitle(f'{method}')
        for x_0, color in zip(start_points, color_map):
            logistic_regression = oracles.create_log_reg_oracle(A, b, lamda)
            x_k, message, history = optimization.newton(logistic_regression, x_0, trace=True, display=True,
                                                        line_search_options={'method': method, 'c': 1})
            ax_1.plot(range(len(history['time'])), history['func'], color=color)
            denominator = np.linalg.norm(logistic_regression.grad(np.zeros(A.shape[1]))) ** 2
            r_k_gd = np.vectorize(lambda x: math.log(np.linalg.norm(x ** 2) / denominator))(history['grad_norm'])
            ax_2.plot(range(len(history['time'])), r_k_gd, color=color)

        ax_1.set_xlabel('Iteration')
        ax_2.set_xlabel('Iteration')
        ax_1.set_ylabel('F(x)')
        ax_2.set_ylabel('ln(r_k)')
        plt.savefig(f'experiment_5/{method}-start-point.png')

    # Plot all methods in one figure.
    figure = plt.figure(figsize=(7, 5))
    ax_1 = figure.add_subplot(121)
    ax_2 = figure.add_subplot(122)
    for method, color in zip(['Wolfe', 'Armijo', 'Constant'], color_map):
        logistic_regression = oracles.create_log_reg_oracle(A, b, lamda)
        x_k, message, history = optimization.newton(logistic_regression, np.zeros(A.shape[1]), trace=True, display=True,
                                                    line_search_options={'method': method, 'c': 1})
        ax_1.plot(range(len(history['time'])), history['func'], color=color, label=method)
        denominator = np.linalg.norm(logistic_regression.grad(np.zeros(A.shape[1]))) ** 2
        r_k_gd = np.vectorize(lambda x: math.log(np.linalg.norm(x ** 2) / denominator))(history['grad_norm'])
        ax_2.plot(range(len(history['time'])), r_k_gd, color=color, label=method)

    ax_1.set_xlabel('Iteration')
    ax_2.set_xlabel('Iteration')
    ax_1.set_ylabel('F(x)')
    ax_2.set_ylabel('ln(r_k)')
    ax_1.legend()
    ax_2.legend()
    plt.savefig(f'experiment_5/all.png')


def main():
    for directory in ['experiment_1', 'experiment_2', 'experiment_3', 'experiment_4', 'experiment_5']:
        if not os.path.exists(directory):
            os.makedirs(directory)

    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
    experiment_5()


if __name__ == '__main__':
    main()
