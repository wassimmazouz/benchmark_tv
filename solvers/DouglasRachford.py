from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch


class Solver(BaseSolver):
    name = 'Douglas-Rachford'

    parameters = {
        'gamma': [0.1, 1, 10]
    }

    def skip(self, A, Anorm2, reg, data_fit, y, isotropy):
        if isotropy == 'anisotropic':
            return True, f"solver does not work with {isotropy} regularization"
        return False, None

    def set_objective(self, A, Anorm2, reg, data_fit, y, isotropy):
        self.A, self.reg, self.y = A, reg, y
        self.data_fit = data_fit
        self.isotropy = isotropy

    def run(self, n_iter):
        if torch.cuda.is_available():
            device = dinv.utils.get_freer_gpu()
        else:
            device = 'cpu'

        y = self.y
        xk = y.clone().to(device)

        data_fidelity = dinv.optim.L2()
        prior = dinv.optim.TVPrior()
        vk = xk.clone().to(device)

        for _ in range(n_iter):

            xk = data_fidelity.prox(vk, y, self.A.physics,
                                    gamma=self.gamma)

            vk = vk + prior.prox(2*xk - vk,
                                 gamma=self.reg*self.gamma) - xk

        self.out = xk.clone().to(device)
        self.obj = data_fidelity(xk, y, self.A.physics) + self.reg*prior.g(xk)

    def get_result(self):
        return dict(name=f'Douglas-Rachford[gamma={self.gamma}]',
                    u=self.out, obj=self.obj)
