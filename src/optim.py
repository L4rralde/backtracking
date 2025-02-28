"""
Optimization algorithms module
Author: Emmanuel Larralde
"""
import numpy as np


class SteepestDescent:
    """
    Steepest Gradient Descent algorithm
    """
    def __init__(self, model: object) -> None:
        self.model = model
        self.prev_model_w = None
        self.loss = self.model.loss()
        self.prev_loss = self.loss
        self.history = {
            'grad_norm': [],
            'loss': [],
        }

    def estimate_step_size(self) -> float:
        """Abstrach method. To be implemented by child"""
        raise NotImplementedError

    def met_stop_criteria(self, *args, **kwargs) -> bool:
        """
        Checks if the algorithm has plateaued
        """
        tf = kwargs.get('tf', 1e-6)
        tx = kwargs.get('tx', 1e-6)
        f_criterion = abs(self.loss - self.prev_loss)/max(1, abs(self.loss))
        if f_criterion <= tf:
            return True
        x_criterion = (
            np.linalg.norm(self.model.w - self.prev_model_w)/
            (max(1, np.linalg.norm(self.model.w)))
        )
        if x_criterion <= tx:
            return True
        return False

    def step(self) -> None:
        """
        Perfmors one cycle of the algorithm
        """
        self.prev_loss = self.model.loss()
        self.grad = self.model.gradient()

        alpha = self.estimate_step_size()

        self.prev_model_w = np.copy(self.model.w)
        self.model.w += -alpha * self.grad
        self.loss = self.model.loss()
        self.history['grad_norm'].append(np.linalg.norm(self.grad))
        self.history['loss'].append(self.loss)


    def solve(self, *args, **kwargs) -> None:
        """
        Finds a local minima of a function given initial guess
        tf, tx thresholds and a maximum number of allowed cycles.
        """
        num_its = kwargs.get('epochs', 1000)
        for _ in range(num_its):
            self.step()
            if self.met_stop_criteria(**kwargs):
                break
        return self.history

class FixedSteepestDescent(SteepestDescent):
    """
    Steepest descent with fixed step size.
    """
    def __init__(self, model: object, *args, **kwargs) -> None:
        super().__init__(model)
        self.alpha = kwargs.get('alpha', 3e-4)

    def estimate_step_size(self) -> float:
        """Selects the constant step size."""
        return self.alpha


class Backtracking(SteepestDescent):
    """
    Steepest descent with backtracking
    """
    def __init__(self, model: object, *args, **kwargs) -> None:
        super().__init__(model)
        self.alpha = kwargs.get('alpha_0', 3e-4)
        self.rho = kwargs.get('rho', 0.9)
        self.c1 = kwargs.get('c1', 0.1)

    def estimate_step_size(self) -> float:
        """
        Selects step size using backtracking
        """
        model_params = self.model.w.copy()
        grad_norm = np.linalg.norm(self.grad)
        k = self.c1 * grad_norm**2
        current_loss = self.model.loss()
        alpha = self.alpha
        while True:
            self.model.w = model_params - alpha*self.grad
            new_loss = self.model.loss()
            if new_loss <= current_loss - alpha*k:
                break
            alpha *= self.rho
        self.model.w = model_params
        return self.alpha

    def met_stop_criteria(self, *args, **kwargs) -> bool:
        """
        Stops when gradient is small enough
        """
        tau = kwargs.get('tau', 1e-3)
        return np.linalg.norm(self.grad) < tau
