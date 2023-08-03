import numpy as np
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from utils import get_line_search_tool
from time import time
from itertools import islice


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

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
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)

    if trace:
        history['time'] = []
        history['residual_norm'] = []
        if len(x_k) <= 2:
            history['x'] = []

    g_k = matvec(x_k) - b
    d_k = - g_k

    if max_iter is None:
        max_iter = b.shape[0]

    start_time = time()
    for i in range(max_iter):
        const_1 = (g_k @ g_k) / (matvec(d_k) @ d_k)
        x_k = x_k + const_1 * d_k
        g_k_next = matvec(x_k) - b

        const_2 = (g_k_next @ g_k_next) / (g_k @ g_k)
        d_k = - g_k_next + const_2 * d_k
        g_k = g_k_next

        if trace:
            history['time'].append(time() - start_time)
            history['residual_norm'].append(np.linalg.norm(g_k))
            if len(x_k) <= 2:
                history['x'].append(x_k)

        if np.linalg.norm(matvec(x_k) - b) <= tolerance * np.linalg.norm(b):
            return x_k, 'success', history

        if display:
            print(f'conjugate_gradients: {i}')

    if np.linalg.norm(matvec(x_k) - b) <= tolerance * np.linalg.norm(b):
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history


def BFGS_multiply(v, H, mu):
    if len(H) == 0:
        return mu * v

    s, y = H[-1]
    H_1 = deque(islice(H, 0, len(H) - 1))
    v_1 = v - (s @ v) / (y @ s) * y

    z = BFGS_multiply(v_1, H_1, mu)

    return z + ((s @ v) - (y @ z)) / (y @ s) * s


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
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
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    stop_number = tolerance * np.square(np.linalg.norm(oracle.grad(x_0), 2))
    x_k = np.copy(x_0)
    H = deque(maxlen=memory_size)
    x_old = np.copy(x_0)
    grad_old = oracle.grad(x_0)

    if trace:
        history['func'] = []
        history['time'] = []
        history['grad_norm'] = []
        if len(x_k) <= 2:
            history['x'] = []

    start_time = time()
    for i in range(max_iter):
        grad = oracle.grad(x_k)
        s_k = x_k - x_old
        y_k = grad - grad_old

        if np.square(np.linalg.norm(grad, 2)) <= stop_number:
            return x_k, 'success', history

        if i > 0:
            H.append((np.copy(s_k), np.copy(y_k)))

        if len(H) == 0:
            d_k = BFGS_multiply(-grad, H, 1)
        else:
            s, y = H[-1]
            mu = (y @ s) / (y @ y)
            d_k = BFGS_multiply(-grad, H, mu)

        x_old = np.copy(x_k)
        grad_old = np.copy(grad)

        a_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1)
        x_k += d_k * a_k

        if trace:
            history['func'].append(oracle.func(x_k))
            history['time'].append(time() - start_time)
            history['grad_norm'].append(np.linalg.norm(grad))
            if len(x_k) <= 2:
                history['x'].append(x_k)

        if display:
            print(f'lbfgs: {i}')

    grad = oracle.grad(x_k)
    if np.square(np.linalg.norm(grad, 2)) <= stop_number:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history


def get_eta(grad):
    return min(0.5, np.sqrt(np.linalg.norm(grad)))


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
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
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    stop_number = tolerance * np.square(np.linalg.norm(oracle.grad(x_0), 2))

    if trace:
        history['func'] = []
        history['time'] = []
        history['grad_norm'] = []
        if len(x_k) <= 2:
            history['x'] = []

    start_time = time()
    for i in range(max_iter):
        grad = oracle.grad(x_k)

        if np.square(np.linalg.norm(grad, 2)) <= stop_number:
            return x_k, 'success', history

        eta = get_eta(grad)
        matvec = lambda v: oracle.hess_vec(x_k, v)

        d_k, message, _ = conjugate_gradients(matvec, -grad, -grad, tolerance=eta, trace=trace, display=display)
        while grad @ d_k >= 0:
            eta /= 10
            d_k, message, _ = conjugate_gradients(matvec, -grad, -grad, tolerance=eta, trace=trace, display=display)

        a_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1)
        x_k += d_k * a_k

        if trace:
            history['func'].append(oracle.func(x_k))
            history['time'].append(time() - start_time)
            history['grad_norm'].append(np.linalg.norm(grad))
            if len(x_k) <= 2:
                history['x'].append(x_k)

        if display:
            print(f'hessian_free_newton: {i}')

    grad = oracle.grad(x_k)
    if np.square(np.linalg.norm(grad, 2)) <= stop_number:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None

    if history is not None:
        history['time'] = []
        history['func'] = []
        history['grad_norm'] = []
        if x_0.size <= 2:
            history['x'] = []
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    stop_number = tolerance * np.square(np.linalg.norm(oracle.grad(x_0), 2))
    start = time()

    try:
        for it in range(max_iter):
            func, func_grad = oracle.func(x_k), oracle.grad(x_k)
            if history is not None:
                history['time'].append(time() - start)
                history['func'].append(func)
                history['grad_norm'].append(np.linalg.norm(func_grad, 2))
                if x_0.size <= 2:
                    history['x'].append(np.copy(x_k))
            if np.square(np.linalg.norm(func_grad, 2)) <= stop_number:
                return x_k, 'success', history
            d_k = -func_grad
            alpha = line_search_tool.line_search(oracle, x_k, d_k)
            x_k = x_k + alpha * d_k
            if display:
                print(f'Iteration: {it}, Error: {np.square(np.linalg.norm(func_grad, 2))}')
        return x_k, 'iterations_exceeded', history
    except Exception as e:
        return x_k, 'computational_error', history
