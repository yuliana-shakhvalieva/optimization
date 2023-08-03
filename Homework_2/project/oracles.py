import numpy as np
import scipy
import scipy.sparse as sp


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A

    def minimize_directional(self, x, d):
        """
        Minimizes the function with respect to a specific direction:
            Finds alpha = argmin f(x + alpha d)
        """
        # TODO: Implement for bonus part
        pass


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATy : function of y
            Computes matrix-vector product A^Ty, where y is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef
        self.diag_b = sp.diags(b, 0)
        self.matmat_diagbA = lambda x: self.diag_b.dot(x)

    def func(self, x):
        return np.mean(
            np.vectorize(lambda x: np.logaddexp(0, x))(self.matmat_diagbA(self.matvec_Ax(x)) * (-1))) + np.dot(x,
                                                                                                               x) * self.regcoef / 2

    def grad(self, x):
        m = self.b.shape[0]
        x = np.array(x)
        return self.matvec_ATx(
            self.matmat_diagbA(np.vectorize(scipy.special.expit)(self.matmat_diagbA(self.matvec_Ax(x)) * (-1)))) * (
                           -1 / m) + x * self.regcoef

    def hess(self, x):
        m = self.b.shape[0]
        n = x.shape[0]
        return self.matmat_ATsA(np.vectorize(lambda x: scipy.special.expit(x) * (1 - scipy.special.expit(x)))(
            self.matmat_diagbA(self.matvec_Ax(x)) * (-1))) * (1 / m) + np.eye(n) * self.regcoef

    def hess_vec(self, x, v):
        m = self.b.shape[0]
        s = sp.diags(np.vectorize(lambda x: scipy.special.expit(x)*(1-scipy.special.expit(x)))(self.matmat_diagbA(self.matvec_Ax(x))*(-1)), 0) * (1/m)
        return self.matvec_ATx(s.dot(self.matvec_Ax(v))) + v * self.regcoef


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None

    def grad_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A @ x
    matvec_ATx = lambda x: A.T @ x

    def matmat_ATsA(s):
        return A.T.dot(sp.diags(s, 0).dot(A))

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def hess_vec_finite_diff(func, x, v, eps=1e-5):
    """
    Returns approximation of the matrix product 'Hessian times vector'
    using finite differences.
    """
    eps = eps ** (1/3)
    n = x.shape[0]
    E = np.eye(n)
    return np.array([(func(x + v * eps + e_j * eps) - func(x + v * eps) - func(x + e_j * eps) + func(x)) / (eps ** 2) for e_j in E])


def testing_hess():
    A = np.random.rand(10, 4)
    b = np.random.rand(10)
    oracle = create_log_reg_oracle(A, b, 1)

    test = 1
    failed = False
    while not failed:
        x = np.random.rand(4)
        v = np.random.rand(4)
        func = oracle.hess_vec(x, v)
        finite_func = hess_vec_finite_diff(oracle.func, x, v)
        flag = np.allclose(func, finite_func, rtol=2e-1, atol=2e-1)
        if flag:
            print(f'Test: {test} passed.')
        else:
            failed = True
            print(f'Test: {test} failed.')
            print(f'Func: {func}.')
            print(f'Finite_func: {finite_func}.')
        test += 1

# testing_hess()
