from collections import defaultdict
import numpy as np
from numpy.linalg import norm
from time import time


def subgradient_method(oracle, x_0, tolerance=1e-2, max_iter=1000, alpha_0=1,
                       display=False, trace=False):
    """
    Subgradient descent method for nonsmooth convex optimization.

    Parameters
    ----------
    oracle : BaseNonsmoothConvexOracle-descendant object
        Oracle with .func() and .subgrad() methods implemented for computing
        function value and its one (arbitrary) subgradient respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    alpha_0 : float
        Initial value for the sequence of step-sizes.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)

    if trace:
        history['func'] = []
        history['time'] = []
        history['duality_gap'] = []
        if len(x_k) <= 2:
            history['x'] = []

    def get_alpha(k):
        return float(alpha_0 / (k + 1) ** 0.5)

    f_k = oracle.func(x_k)
    min_x_k = np.copy(x_k)
    min_f_k = np.copy(f_k)

    start_time = time()
    for k in range(max_iter):
        sub_grad_k = oracle.subgrad(x_k)
        d_k = sub_grad_k / norm(sub_grad_k)
        dual_gap_k = oracle.duality_gap(x_k)

        if trace:
            history['func'].append(f_k)
            history['time'].append(time() - start_time)
            history['duality_gap'].append(dual_gap_k)
            if len(x_k) <= 2:
                history['x'].append(x_k)

        if dual_gap_k <= tolerance:
            return min_x_k, 'success', history

        if display:
            print(f'subgradient_method: {k}')

        x_k = np.float64(x_k)
        x_k -= get_alpha(k) * d_k
        f_k = oracle.func(x_k)

        if f_k < min_f_k:
            min_x_k = np.copy(x_k)
            min_f_k = np.copy(f_k)

    dual_gap_k = oracle.duality_gap(x_k)
    if dual_gap_k <= tolerance:
        return min_x_k, 'success', history
    else:
        return min_x_k, 'iterations_exceeded', history


def proximal_gradient_method(oracle, x_0, L_0=1, tolerance=1e-5,
                             max_iter=1000, trace=False, display=False):
    """
    Gradient method for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented
        for computing function value, its gradient and proximal mapping
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of objective function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    L_k = np.copy(L_0)

    if trace:
        history['func'] = []
        history['time'] = []
        history['duality_gap'] = []
        history['cycle_count'] = []
        if len(x_k) <= 2:
            history['x'] = []

    start_time = time()
    for k in range(max_iter):
        f_k = oracle._f.func(x_k)
        d_k = oracle.grad(x_k)
        dual_gap_k = oracle.duality_gap(x_k)

        if dual_gap_k <= tolerance:
            return x_k, 'success', history

        if display:
            print(f'proximal_gradient_method: {k}')

        cycle_counter = 0
        while True:
            cycle_counter += 1
            x_k_next = oracle.prox(x_k - 1 / L_k * d_k, 1 / L_k)
            if oracle._f.func(x_k_next) > f_k + d_k @ (x_k_next - x_k) + L_k / 2 * np.square(norm(x_k_next - x_k)):
                L_k *= 2
            else:
                break

        if trace:
            history['func'].append(f_k)
            history['time'].append(time() - start_time)
            history['duality_gap'].append(dual_gap_k)
            history['cycle_count'].append(cycle_counter)
            if len(x_k) <= 2:
                history['x'].append(x_k)

        L_k = np.float64(L_k)
        L_k /= 2
        x_k = np.copy(x_k_next)

    dual_gap_k = oracle.duality_gap(x_k)
    if dual_gap_k <= tolerance:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history


def proximal_fast_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                                  max_iter=1000, trace=False, display=False):
    """
    Fast gradient method for composite minimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented 
        for computing function value, its gradient and proximal mapping 
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of objective function values phi(best_point) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    L_k = np.copy(L_0)
    A_k = 0
    v_k = 0

    if trace:
        history['func'] = []
        history['time'] = []
        history['duality_gap'] = []
        history['cycle_count'] = []

    min_x_k = np.copy(x_k)
    min_f_k = np.copy(oracle._f.func(x_k))
    sum_v_k = 0
    start_time = time()
    for k in range(max_iter):

        dual_gap_k = oracle.duality_gap(x_k)

        if dual_gap_k <= tolerance:
            return min_x_k, 'success', history

        if display:
            print(f'proximal_fast_gradient_method: {k}')

        cycle_counter = 0
        while True:
            cycle_counter += 1
            a_k = (1 + np.sqrt(1 + 4 * L_k * A_k)) / (2 * L_k)
            A_k_next = A_k + a_k
            y_k = (A_k * x_k + a_k * v_k) / A_k_next

            d_k = oracle.grad(y_k)
            sum_v_k_next = sum_v_k + a_k * d_k
            v_k_next = oracle.prox(x_0 - sum_v_k_next, A_k_next)

            x_k_next = (A_k * x_k + a_k * v_k_next) / A_k_next
            f_k = oracle._f.func(y_k)

            if oracle._f.func(x_k_next) > f_k + d_k @ (x_k_next - y_k) + L_k / 2 * np.square(norm(x_k_next - y_k)):
                L_k *= 2
            else:
                break

        if trace:
            history['func'].append(f_k)
            history['time'].append(time() - start_time)
            history['duality_gap'].append(dual_gap_k)
            history['cycle_count'].append(cycle_counter)

        L_k = np.float64(L_k)
        L_k /= 2
        x_k = np.copy(x_k_next)
        A_k = np.copy(A_k_next)
        v_k = np.copy(v_k_next)
        sum_v_k = np.copy(sum_v_k_next)

        if f_k < min_f_k:
            min_x_k = np.copy(x_k)
            min_f_k = np.copy(f_k)

    dual_gap_k = oracle.duality_gap(x_k)
    if dual_gap_k <= tolerance:
        return min_x_k, 'success', history
    else:
        return min_x_k, 'iterations_exceeded', history
