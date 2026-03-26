import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from scipy.linalg import cho_solve, cholesky, solve_triangular

class ModelSelectorMOD:
    def __init__(self,bo, X_train, y_train, kernel,
                 normalize_y=True, normalize_X=True):
        self.bo = bo
        self.X = np.asarray(X_train)
        self.y = np.asarray(y_train)
        self.kernel = kernel

        if normalize_X:
            self.scaler_X = StandardScaler()
            self.X = self.scaler_X.fit_transform(self.X)
        else:
            self.scaler_X = None

        if normalize_y:
            self.scaler_y = StandardScaler()
            self.y = self.scaler_y.fit_transform(
                self.y.reshape(-1, 1)
            ).ravel()
        else:
            self.scaler_y = None

        self.n = len(self.X)

    def _loo_and_grad(self, theta):
        kernel_t = self.kernel.clone_with_theta(theta)
        K, K_grad = kernel_t(self.X, eval_gradient=True)

        jitter = 1e-6 * np.mean(np.diag(K))
        K_reg = K + jitter * np.eye(self.n)

        L = cholesky(K_reg, lower=True)

        K_inv = cho_solve((L, True), np.eye(self.n))
        alpha = K_inv @ self.y

        K_inv_diag = np.diag(K_inv)
        eps = 1e-12
        K_inv_diag = np.clip(K_inv_diag, eps, None)

        mu_loo = self.y - alpha / K_inv_diag
        s2_loo = 1.0 / K_inv_diag

        residual = self.y - mu_loo

        logp = (
            -0.5 * np.log(s2_loo)
            -0.5 * residual**2 / s2_loo
            -0.5 * np.log(2 * np.pi)
        )
        loo = np.sum(logp)
        n_params = len(theta)
        grad = np.zeros(n_params)

        for j in range(n_params):
            dK = K_grad[:, :, j]
            dK_inv = -K_inv @ dK @ K_inv
            dK_inv_diag = np.diag(dK_inv)
            dalpha = dK_inv @ self.y
            term1 = 0.5 * dK_inv_diag / K_inv_diag * (
                1 - residual**2 / s2_loo
            )

            term2 = residual * dalpha

            grad[j] = np.sum(term1 + term2)

        return -loo, -grad  # minimize

    def find_via_scipy(self,method='L-BFGS-B', n_restarts=5, options=None):
        bounds = self.kernel.bounds
        best_val = np.inf
        best_theta = None

        for i in range(n_restarts):
            if i == 0:
                x0 = self.kernel.theta
            else:
                x0 = np.array([
                    np.random.uniform(low, high)
                    for low, high in bounds
                ])

            def objective(th):
                val, grad = self._loo_and_grad(th)
                return val, grad

            res = minimize(
                fun=lambda th: objective(th)[0],
                jac=lambda th: objective(th)[1],
                x0=x0,
                method=method,
                bounds=bounds,
                options={'maxiter': 200}
            )

            if res.success and res.fun < best_val:
                best_val = res.fun
                best_theta = res.x

        if best_theta is None:
            best_theta = self.kernel.theta

        return self.kernel.clone_with_theta(best_theta)