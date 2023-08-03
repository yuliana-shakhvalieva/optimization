import random
import optimization
import oracles
import os
import matplotlib.pyplot as plt
import numpy as np


def experiment_1():
    np.random.seed(7412)

    def compose(values, indexes):
        return [values[idx] for idx in indexes]

    list_a_0 = np.linspace(0.1, 10, 50)
    list_k = [0, 1, 10, 100]
    list_x_0 = [k * np.random.rand(5) for k in list_k]

    result = []
    for k, x_0 in zip(list_k, list_x_0):
        count_iter = []
        for a_0 in list_a_0:
            A = np.random.rand(5, 5)
            b = np.random.rand(5)
            lamda = 1 / 5
            oracle = oracles.create_lasso_nonsmooth_oracle(A, b, lamda)
            x_k, message, history = optimization.subgradient_method(oracle, x_0, alpha_0=a_0, trace=True,
                                                                    max_iter=int(4e4))
            print(f'k: {k}, a_0: {a_0} - {message}')

            len_history = len(history['func'])
            count_iter.append(len_history)

        result.append(count_iter)
        plt.plot(list_a_0, count_iter, label=f'k = {k}')

    plt.xlabel("Alpha")
    plt.ylabel("Iterations")
    plt.legend()
    plt.savefig(f'experiment_1/img_1.png')

    result = np.array(result)

    figure = plt.figure(figsize=(7, 5))
    ax_1 = figure.add_subplot(121)
    ax_2 = figure.add_subplot(122)

    ax_1.plot(list_k, compose(list_a_0, np.argmin(result, axis=1)))
    ax_2.plot(list_a_0, compose(list_k, np.argmin(result, axis=0)))

    ax_1.set_xlabel('k')
    ax_1.set_ylabel('Alpha')
    ax_1.set_title('Зависимость лучшего a_0 от x_0', fontsize=10)

    ax_2.set_xlabel('Alpha')
    ax_2.set_ylabel('k')
    ax_2.set_title('Зависимость лучшей x_0 от a_0', fontsize=10)

    plt.savefig(f'experiment_1/img_2.png')


def experiment_2():
    np.random.seed(7412)

    A = np.random.rand(5, 5)
    b = np.random.rand(5)
    lamda = 1 / 5

    for method, method_name in zip([optimization.proximal_gradient_method, optimization.proximal_fast_gradient_method],
                                   ['Gradient method', 'Fast gradient method']):
        oracle = oracles.create_lasso_prox_oracle(A, b, lamda)
        x_k, message, history = method(oracle, np.zeros(5), trace=True)
        print(f'{method_name}: {message}')

        plt.plot(range(len(history['cycle_count'])), history['cycle_count'], label=method_name)

    plt.xlabel("Iteration")
    plt.ylabel("Line search counter")
    plt.legend()
    plt.savefig(f'experiment_2/img.png')


def experiment_3():
    np.random.seed(7412)
    methods = [optimization.subgradient_method, optimization.proximal_gradient_method,
               optimization.proximal_fast_gradient_method]
    methods_name = ['Subgradient method', 'Gradient method', 'Fast gradient method']
    oracles_func = [oracles.create_lasso_nonsmooth_oracle, oracles.create_lasso_prox_oracle,
                    oracles.create_lasso_prox_oracle]
    color_map = plt.get_cmap("tab10").colors

    for i in range(3):
        m = random.randint(10, 100)
        n = random.randint(2, 10)
        A = np.random.rand(m, n)
        b = np.random.rand(m)
        lamda = 1 / m
        x_0 = np.zeros(n)

        figure = plt.figure(figsize=(7, 5))
        ax_1 = figure.add_subplot(121)
        ax_2 = figure.add_subplot(122)
        figure.suptitle(f'm={m}, n={n}')

        for idx, (method, method_name, oracle_func) in enumerate(zip(methods, methods_name, oracles_func)):
            oracle = oracle_func(A, b, lamda)
            x_k, message, history = method(oracle, x_0, max_iter=int(2e5), trace=True)
            print(f"{i}, m={m}, n={n}, method: {method_name}, message: {message}")

            ax_1.plot(range(len(history['duality_gap'])), np.log(history['duality_gap']), color=color_map[idx],
                      label=method_name)
            ax_2.plot(history['time'], history['func'], color=color_map[idx], label=method_name)

        ax_1.set_xlabel('Iterations')
        ax_1.set_ylabel('ln(duality_gap)')
        ax_1.set_title('Гарантируемая точность по зазору \n двойственности против числа итераций', fontsize=8)
        ax_1.legend()

        ax_2.set_xlabel('Seconds')
        ax_1.set_ylabel('ln(duality_gap)')
        ax_2.set_title('Гарантируемая точность по зазору \n двойственности против реального времени работы', fontsize=8)
        ax_2.legend()

        plt.savefig(f'experiment_3/{i}.png')


def main():
    for directory in ['experiment_1', 'experiment_2', 'experiment_3']:
        if not os.path.exists(directory):
            os.makedirs(directory)

    experiment_1()
    experiment_2()
    experiment_3()


if __name__ == '__main__':
    main()
