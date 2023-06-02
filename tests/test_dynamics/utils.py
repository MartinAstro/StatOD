import matplotlib.pyplot as plt
import numpy as np


# a function that takes in x_0 and computes x_i with the STM and plots the
# true x_i and the propagated x_i
def plot_true_and_estimated_X(X_pert, X_STM):
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(X_pert[:, 0], label="True")
    plt.plot(X_STM[:, 0], label="STM", linestyle="--")
    plt.legend()
    plt.title("x")
    plt.subplot(3, 2, 3)
    plt.plot(X_pert[:, 1], label="True")
    plt.plot(X_STM[:, 1], label="STM", linestyle="--")
    plt.title("y")
    plt.subplot(3, 2, 5)
    plt.plot(X_pert[:, 2], label="True")
    plt.plot(X_STM[:, 2], label="STM", linestyle="--")
    plt.title("z")

    plt.subplot(3, 2, 2)
    plt.plot(X_pert[:, 3], label="True")
    plt.plot(X_STM[:, 3], label="STM", linestyle="--")
    plt.title("vx")
    plt.subplot(3, 2, 4)
    plt.plot(X_pert[:, 4], label="True")
    plt.plot(X_STM[:, 4], label="STM", linestyle="--")
    plt.title("vy")
    plt.subplot(3, 2, 6)
    plt.plot(X_pert[:, 5], label="True")
    plt.plot(X_STM[:, 5], label="STM", linestyle="--")
    plt.title("vz")

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.linalg.norm(X_pert[:, 0:3] - X_STM[:, 0:3], axis=1))
    plt.title("Position Error")
    plt.subplot(2, 1, 2)
    plt.plot(np.linalg.norm(X_pert[:, 3:6] - X_STM[:, 3:6], axis=1))
    plt.title("Velocity Error")

    plt.show()


# A function that plots the 3d trajectory of a state vector X
def plot_trajectory(X, title=""):
    import matplotlib.pyplot as plt

    # make a 3d figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X[:, 0], X[:, 1], X[:, 2], linewidth=1)
    ax.set_title(title)


def test_phi(logs, t_idx, tol=1e-3):
    X_0 = logs.x_hat_i_plus[0]
    X_f = logs.x_hat_i_plus[t_idx]
    phi = logs.phi_ti_t0[t_idx]

    X_f_prop = phi @ X_0
    dx = X_f - X_f_prop
    (X_f - X_f_prop) / X_f * 100

    np.linalg.norm(dx)

    print(f"Time:{t_idx}")
    print(f"X_i True:{X_f.reshape((-1,))}")
    print(f"X_i STM:{X_f_prop.reshape((-1,))}")
    print(f"dX:{(X_f - X_f_prop).reshape((-1,))}")
    print(f"dX_i:{np.linalg.norm(X_f - X_f_prop)}")
    assert np.linalg.norm(X_f - X_f_prop) < tol


# A model wrapper class that takes a model and keeps all the same attributes and methods,
# and simply renames the .compute_dfdx method to .generate_dadx
class ModelWrapper:
    def __init__(self, model, dim_constants):
        self.model = model
        self.planet = model.celestial_body
        self.dim_constants = dim_constants

    def compute_acceleration(self, X):
        l_star = self.dim_constants["l_star"]
        t_star = self.dim_constants["t_star"]

        ms2 = l_star / t_star**2
        X_dim = X * self.dim_constants["l_star"]

        return self.model.compute_acceleration(X_dim.reshape((-1, 3)) * 1e3) / 1e3 / ms2

    def generate_dadx(self, X):
        t_star = self.dim_constants["t_star"]
        X_dim = X * self.dim_constants["l_star"]

        c = 1.0 / t_star**2
        return self.model.compute_dfdx(X_dim.reshape((-1, 3)) * 1e3) / c


def f_ivp(t, Z, f, dfdX, f_args, N, pbar):
    N = int(1 / 2 * (np.sqrt(4 * len(Z) + 1) - 1))

    X_inst = Z[0:N]
    phi_inst = Z[N:].reshape((N, N))

    f_inst = np.array(f(X_inst, f_args)).reshape((N))
    dfdx_inst = np.array(dfdX(X_inst, f_inst, f_args)).reshape((N, N))

    phi_dot = dfdx_inst @ phi_inst
    Zd = np.hstack((f_inst, phi_dot.reshape((-1))))

    pbar.update(t)
    return Zd
