from benchopt import BaseSolver, safe_import_context
import torch

with safe_import_context() as import_ctx:
    import deepinv as dinv
    from benchmark_utils.deepinv_funcs import L12Prior
    import torch.optim as optim


def func(A, L, x, y, yk, vk, gamma):
    diff = L(x) - yk + vk
    dtf = 0.5 * torch.sum(diff ** 2)
    diff2 = A.operator(x) - y
    pen = torch.sum(diff2 ** 2) / (2 * gamma)
    return dtf + pen


class Solver(BaseSolver):
    name = 'ADMM'
    parameters = {'gamma': [0.5, 1, 2]}

    def skip(self, A, Anorm2, reg, data_fit, y, isotropy):
        if isotropy == 'anisotropic':
            return True, f"solver does not work with {isotropy} regularization"
        return False, None

    def set_objective(self, A, Anorm2, reg, data_fit, y, isotropy):
        self.A, self.reg, self.y = A, reg, y
        self.data_fit = data_fit
        self.isotropy = isotropy

    def run(self, n_iter):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        y = self.y.clone().to(device)
        L = dinv.optim.TVPrior().nabla
        prior = L12Prior()
        data_fidelity = dinv.optim.L2()

        xk = torch.zeros_like(y, device=device).requires_grad_()
        yk = torch.zeros_like(L(xk), device=device)
        vk = torch.zeros_like(yk, device=device)
        optimizer = optim.SGD([xk])

        def closure():
            optimizer.zero_grad()
            loss = func(self.A, L, xk, y, yk, vk, self.gamma)
            loss.backward(retain_graph=True)
            return loss

        for _ in range(n_iter):
            optimizer.step(closure)

            yk = prior.prox(vk + L(xk), gamma=self.reg/self.gamma)

            vk += L(xk) - yk

        self.out = xk.detach().clone().to('cpu')
        self.obj = data_fidelity(xk, y, self.A.physics) + self.reg*prior.g(L(
            xk))

    def get_result(self):
        return dict(name=f'ADMM DeepInv[gamma={self.gamma}]',
                    u=self.out, obj=self.obj)
