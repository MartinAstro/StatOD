import numpy as np


# a function that takes in x_0 and computes x_i with the STM and plots the
# true x_i and the propagated x_i
def plot_true_and_estimated_X(X, phi):
    X_0 = X[0]
    X_i = X
    X_i_prop = phi @ X_0

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(X_i[:, 0:3], label=["True" + str(i) for i in range(3)])
    plt.plot(X_i_prop[:, 0:3], label=[str(i) for i in range(3)])
    plt.ylabel("Position")
    plt.legend()

    plt.figure()
    plt.plot(X_i[:, 3:6], label=["True" + str(i) for i in range(3)])
    plt.plot(X_i_prop[:, 3:6], label=[str(i) for i in range(3)])
    plt.ylabel("Velocity")
    plt.legend()

    # plt.figure()
    # plt.plot(X_i[:, 6:9], label=["True" + str(i) for i in range(3)])
    # plt.plot(X_i_prop[:, 6:9], label=[str(i) for i in range(3)])
    # plt.ylabel("Acceleration")
    # plt.legend()

    dX = X_i[:, 0:3] - X_i_prop[:, 0:3]
    dV = X_i[:, 3:6] - X_i_prop[:, 3:6]
    # dA = X_i[:, 6:9] - X_i_prop[:, 6:9]

    plt.figure()
    plt.plot(dX)
    plt.ylabel("dX")
    plt.legend()

    plt.figure()
    plt.plot(dV)
    plt.ylabel("dV")
    plt.legend()

    # plt.figure()
    # plt.plot(dA)
    # plt.ylabel("dA")
    # plt.legend()

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
