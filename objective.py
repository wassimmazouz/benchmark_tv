from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.matrix_op import grad_rgb


class Objective(BaseObjective):
    min_benchopt_version = "1.5"
    name = "TV"

    parameters = {
        'reg': [0.1, 0.5, 1],
        'delta': [0.9],
        'isotropy': ["anisotropic", "isotropic"],
        'data_fit': ["lsq"]
    }
    requirements = ['pip::torch', 'deepinv']

    def set_data(self, x_true, type_A, A, y, Anorm2):
        self.A = A
        self.y = y
        self.Anorm2 = Anorm2
        self.x_true = x_true
        self.type_A = type_A

    def evaluate_result(self, name, u, obj):
        return obj

    def get_one_result(self):
        return dict(u=np.zeros(self.y.shape))

    def get_objective(self):
        return dict(A=self.A,
                    Anorm2=self.Anorm2,
                    reg=self.reg,
                    data_fit=self.data_fit,
                    y=self.y,
                    isotropy=self.isotropy)

    def isotropic_tv_value(self, u):
        gh, gv = grad_rgb(u)
        return (np.sqrt(gh ** 2 + gv ** 2)).sum()

    def anisotropic_tv_value(self, u):
        gh, gv = grad_rgb(u)
        return (np.abs(gh) + np.abs(gv)).sum()

    def save_final_results(self, name, u):
        return [name+f', type_A = {self.type_A}', self.y, self.x_true,
                u]
