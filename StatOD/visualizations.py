from os import stat
import os
import matplotlib.pyplot as plt
import numpy as np
import sigfig
from StatOD.utils import ECEF_2_ECI
import matplotlib.tri as mtri

plt.rc('font', size= 11.0)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes.formatter', limits=(-1,1))
plt.rc('figure', figsize=(3.0*1.5, 2.25*1.5))
plt.rc('axes.grid', axis='both')
plt.rc('axes.grid', which='both')
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
plt.rc('grid', linewidth='0.1')
plt.rc('grid', color='.25')
plt.rc('figure',max_open_warning=30)
plt.rc("figure", autolayout=True) # https://matplotlib.org/stable/tutorials/intermediate/tight_layout_guide.html


def reject_outliers(data, m = 3.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def annotate_stats(values, variable_str='x', xy=(0.3, 1.05), start_idx=0):
    rms_avg = sigfig.round(np.nanmean(values[start_idx:]), sigfigs=2, notation='scientific')
    rms_std = sigfig.round(np.nanstd(values[start_idx:]), sigfigs=2, notation='scientific')
    plt.annotate(r"$\bar{" + variable_str + r'[' + str(start_idx) + r":]" + r"}_{RMS} = $" + str(rms_avg) + "±" + str(rms_std), xy=xy, xycoords='axes fraction')

def annotate_residual(values, variable_str='x', xy=(0.3, 1.05), start_idx=0):
    rms_avg = sigfig.round(np.nanmean(values[start_idx:]), sigfigs=2, notation='scientific')
    rms_std = sigfig.round(np.nanstd(values[start_idx:]), sigfigs=2, notation='scientific')
    plt.annotate('Residual RMS ' + variable_str + " = " + str(rms_avg) + "±" + str(rms_std), xy=xy, xycoords='axes fraction')



class VisualizationBase():
    def __init__(self, logger, save_dir=None, save=None):
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.logger = logger
        pass

    ###########
    ## States #
    ###########

    def plot_states(self):
        N = len(self.logger.x_i[0])
        t = self.logger.t_i
        for i in range(N):
            x = self.logger.x_i[:,i]
            y_label = self.logger.state_labels[i]
            self.__plot_state(t, x, y_label)
        
    def __plot_state(self, t, x, label):
        plt.figure()
        plt.plot(t, x)
        plt.ylabel(label)
        plt.xlabel('Time')


    ################
    ## State Error #
    ################

    def plot_state_errors(self, x_true):
        N = len(self.logger.x_i[0])
        x_hat = self.logger.x_hat_i_plus
        P = self.logger.P_i_plus
        labels = self.logger.state_labels
        for i in range(N):
            self.plot_state_error(x_hat[:,i], x_true[:,i], np.sqrt(P[:,i,i]), labels[i])


    def plot_state_error(self, x_hat, x_true, sigma, y_label):
        state_error = x_hat - x_true
        cov_upper = 3*sigma
        cov_lower = -3*sigma
        t = self.logger.t_i
        plt.figure()
        plt.plot(t, state_error, marker='o', markersize=0.5, linewidth=0.5)
        plt.plot(t, cov_upper, color='r', alpha=0.5, linewidth=0.5)
        plt.plot(t, cov_lower, color='r', alpha=0.5, linewidth=0.5)
        plt.ylabel(y_label)

        start_idx = len(state_error) // 10
        y_lim = 10*np.nanmedian(sigma[start_idx:])

        if np.isnan(y_lim):
            y_lim = 10*np.median(np.abs(state_error))
        plt.ylim([-y_lim, y_lim])


    ##########################
    ## Measurement Residuals #
    ##########################

    def plot_residuals(self, y, y_hat, R, y_labels):
        M = len(y[0])
        t = self.logger.t_i
        for i in range(M):
            self.__plot_residual(t, y[:,i], y_hat[:,i], y_labels[i])
            plt.plot(t, 3*np.sqrt(R[:,i,i]), color='red')
            plt.plot(t, -3*np.sqrt(R[:,i,i]), color='red')
        
    def __plot_residual(self, t, y, y_hat, label):
        plt.figure()
        plt.scatter(t, y-y_hat, s=2)
        plt.xlabel("Time")
        plt.ylabel(label)


    
    ########################
    ## Plotting Utilities ##
    ########################

    def save(self,fig_name):
        if self.save_dir is not None:
            os.makedirs(self.save_dir ,exist_ok=True)
            fig_name = fig_name.replace("$", "").replace('\\', "")
            plt.savefig(self.save_dir + fig_name + ".pdf")



    def plot_vlines(self, idx_list):
        for i in plt.get_fignums():
            plt.figure(i)
            ylim = plt.gca().get_ylim()
            for idx in idx_list:
                plt.vlines(self.logger.t_i[idx], ylim[0], ylim[1], colors='black', zorder=-10, linewidths=0.5, alpha=0.3)










def plot_measurement_residuals(t, Y, x, station_list_ECEF, omega, theta_0, h, R0, logger):
        t_day = 24*3600

        t_vec = []
        dy1_vec = []
        dy2_vec = []
        M = len(logger.t_i)
        for i in range(M):
            t_i = t[i]
            obs_idx_i = int(Y[i,0])
            X_obs_ECEF = station_list_ECEF[obs_idx_i]
            X_obs_ECI = ECEF_2_ECI(t_i, X_obs_ECEF, omega, theta_0)
            y_i = Y[i,1:]
            # print(logger.x_hat_i_plus[i])

            y_hat_i = h(x[i], X_obs_ECI)
            dy = y_hat_i - y_i
            t_vec.append(t_i)
            dy1_vec.append(dy[0])
            dy2_vec.append(dy[1])

        t_vec = np.array(t_vec)
        plt.figure()
        plt.subplot(2,1,1)
        plt.scatter(t_vec / t_day, dy1_vec, s=2)
        # print(dy1_vec)
        annotate_stats(dy1_vec, variable_str=r'\delta y_{1}')
        plt.xlabel("Time [days]")
        plt.ylabel("$\delta y_1$")

        plt.subplot(2,1,2)
        plt.scatter(t_vec / t_day, dy2_vec, s=2)
        annotate_stats(dy2_vec, variable_str=r'\delta y_{2}')
        plt.xlabel("Time [days]")
        plt.ylabel("$\delta y_2$")


        plt.subplot(2,1,1)
        plt.hlines(np.sqrt(R0[0,0]), logger.t_i[0]/ t_day, logger.t_i[-1]/ t_day, colors='red')
        plt.hlines(-np.sqrt(R0[0,0]), logger.t_i[0]/ t_day, logger.t_i[-1]/ t_day, colors='red')

        plt.subplot(2,1,2)
        plt.hlines(np.sqrt(R0[1,1]), logger.t_i[0]/ t_day, logger.t_i[-1]/ t_day, colors='red')
        plt.hlines(-np.sqrt(R0[1,1]), logger.t_i[0]/ t_day, logger.t_i[-1]/ t_day, colors='red')

def plot_RMS_measurement_error(t, Y, station_list_ECEF, omega, theta_0, h, R0, logger):
    t_vec = []
    dy1_vec = []
    dy2_vec = []
    M = len(logger.t_i)
    for i in range(M):
        t_i = t[i]
        obs_idx_i = int(Y[i,0])
        X_obs_ECEF = station_list_ECEF[obs_idx_i]
        X_obs_ECI = ECEF_2_ECI(t_i, X_obs_ECEF, omega, theta_0)
        y_i = Y[i,1:]
        
        y_hat_i = h(logger.x_hat_i[i], X_obs_ECI)
        dy = y_hat_i - y_i
        t_vec.append(t_i)
        dy1_vec.append(dy[0])
        dy2_vec.append(dy[1])


def plot_RMS_state_NLBF(x0_list, f, f_args, h, logger, t, Y, x_true, omega, theta_0, station_list_ECEF):
    from scipy.integrate import solve_ivp
    RMS_state_component_error = []
    RMS_state_r_error = []
    RMS_state_v_error = []
    RMS_y1_error = []
    RMS_y2_error = []
    for x0 in x0_list:
        
        # Propagate the initial state
        est_sol = solve_ivp(lambda t, x: f(x,f_args), 
                    [0, t[-1]], 
                    x0, 
                    atol=1E-14, rtol=1E-14, 
                    t_eval=logger.t_i)       
        x_est = est_sol.y.T

        
        # Calculate the measurement residuals
        t_vec = []
        dy1_vec = []
        dy2_vec = []
        M = len(logger.t_i)
        for i in range(M):
            t_i = t[i]
            obs_idx_i = int(Y[i,0])
            X_obs_ECEF = station_list_ECEF[obs_idx_i]
            X_obs_ECI = ECEF_2_ECI(t_i, X_obs_ECEF, omega, theta_0)
            y_i = Y[i,1:]
            
            y_hat_i = h(x_est[i], X_obs_ECI)
            dy = y_hat_i - y_i
            t_vec.append(t_i)
            dy1_vec.append(dy[0])
            dy2_vec.append(dy[1])
        

        RMS_state_component_error.append(np.mean(np.sqrt((x_est - x_true)**2), axis=0)) # Nx6
        RMS_state_r_error.append(np.mean(np.sqrt(np.sum(((x_est - x_true)**2)[:,0:3], axis=1)))) # Nx1
        RMS_state_v_error.append(np.mean(np.sqrt(np.sum(((x_est - x_true)**2)[:,3:6], axis=1)))) # Nx1

        RMS_y1_error.append(np.sum(np.sqrt(np.square(dy1_vec)))) # Nx1
        RMS_y2_error.append(np.sum(np.sqrt(np.square(dy2_vec)))) # Nx1

    RMS_state_component_error = np.array(RMS_state_component_error)
    RMS_state_r_error = np.array(RMS_state_r_error)
    RMS_state_v_error = np.array(RMS_state_v_error)
    RMS_y1_error = np.array(RMS_y1_error)
    RMS_y2_error = np.array(RMS_y2_error)

    iterations = np.arange(0, len(RMS_state_component_error),1)
    fig1 = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(iterations[1:], RMS_state_component_error[1:,0], label=r'$\delta x_0$')
    plt.plot(iterations[1:], RMS_state_component_error[1:,1], label=r'$\delta x_1$')
    plt.plot(iterations[1:], RMS_state_component_error[1:,2], label=r'$\delta x_2$')
    plt.plot(iterations[1:], RMS_state_r_error[1:], label=r'$||\delta x[0:3]||$')
    plt.ylabel("RMS Position Error [m]")
    plt.xlabel("Iterations")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(iterations[1:], RMS_state_component_error[1:,3], label=r'$\delta x_3$')
    plt.plot(iterations[1:], RMS_state_component_error[1:,4], label=r'$\delta x_4$')
    plt.plot(iterations[1:], RMS_state_component_error[1:,5], label=r'$\delta x_5$')
    plt.plot(iterations[1:], RMS_state_v_error[1:], label=r'$||\delta x[3:6]||$')
    plt.ylabel("RMS Velocity Error [m]")
    plt.xlabel("Iterations")
    plt.legend()

    fig2 = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(iterations[1:], RMS_y1_error[1:], label=r'$||\delta \rho||$')
    plt.ylabel('RMS Range Error')
    plt.xlabel("Iterations")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(iterations[1:], RMS_y2_error[1:], label=r'$||\delta \dot{\rho}||$')
    plt.ylabel('RMS Range Rate Error')
    plt.xlabel("Iterations")
    plt.legend()

    return fig1, fig2



def plot_covariance_trace(t_vec, P_vec, y_label, directory):
    plt.figure()
    t_vec_axis = t_vec / 3600.0
    plt.semilogy(t_vec_axis, np.trace(P_vec, axis1=1, axis2=2))
    plt.xlabel("Time [hr]")
    plt.ylabel(y_label)
    if directory is not None:
        os.makedirs(directory ,exist_ok=True)
        fig_name = y_label.replace("$", "").replace('\\', "").replace(" ", "_")
        plt.savefig(directory+ fig_name + "_trace.pdf")

def plot_covariance_ellipsoid(P_vec, axis_labels, directory=None):
    u = np.linspace(0, 2*np.pi, num=50, endpoint=True)
    v = np.linspace(0, np.pi, num=50, endpoint=True)

    u, v = np.meshgrid(u, v)
    u, v = u.flatten(), v.flatten()

    a, b, c = np.nan_to_num(np.sqrt(np.diag(P_vec)))


    x = a*np.cos(u)*np.sin(v)
    y = b*np.sin(u)*np.sin(v)
    z = c*np.cos(v)

    tri = mtri.Triangulation(u, v)

    # Plot the surface.  The triangles in parameter space determine which x, y, z
    # points are connected by an edge.
    ax = plt.figure().add_subplot(1, 1, 1, scenarioion='3d')
    ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral)

    max_val = np.max(np.concatenate((x,y,z)))
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])

    if directory is not None:
        os.makedirs(directory ,exist_ok=True)
        fig_name = axis_labels[0] + "_" + axis_labels[1] + "_" + axis_labels[2]
        fig_name = fig_name.replace('$', "").replace("\\", "")
        plt.savefig(directory+ fig_name + "_trace.pdf")   



def plot_state_residual_and_covariance(t_ref, t_vec, x_hat, x_ref, y_labels, covariances=None, directory=None, custom_ylim=None, use_std=False,sigma=3):
    if x_ref is None:
        dx = x_hat
        t_joint = t_vec
        idx_2 = np.arange(0, len(t_joint),1)
    else:
        t_joint, idx_1, idx_2 = np.intersect1d(t_ref, t_vec, return_indices=True)

        dx = x_ref[idx_1] - x_hat[idx_2]

    def reject_outliers(data, m = 3.):
        data_wo_nan = data[~np.isnan(data)]
        d = np.abs(data_wo_nan - np.median(data_wo_nan))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        return data_wo_nan[s<m]
    t_vec_axis = t_joint / 3600.0 
    for i in range(len(dx[0])):
        plt.figure()
        if x_ref is not None:
            plt.scatter(t_vec_axis, dx[:,i], s=2)
        plt.xlabel("Time [hr]")
        plt.ylabel(y_labels[i])
        annotate_residual(np.sqrt(np.square(dx[:,i])), y_labels[i])

        if covariances is not None:
            cov = np.sqrt(covariances[idx_2,i,i])
            plt.plot(t_vec_axis,sigma*cov, c='r')
            plt.plot(t_vec_axis,-sigma*cov, c='r')
            mean = np.nanmean(sigma*cov)
            # mean = np.mean(reject_outliers(cov))
            plt.ylim([-5*mean, 5*mean])

        if custom_ylim is not None:
            plt.ylim(custom_ylim)
        if use_std:
            std = np.std(reject_outliers(dx[:,i]))
            plt.ylim([-5*std, 5*std])
        if directory is not None:
            os.makedirs(directory ,exist_ok=True)
            fig_name = y_labels[i].replace("$", "").replace('\\', "")
            plt.savefig(directory + fig_name + ".pdf")

    RMS_vals = np.sqrt(np.mean(np.square(dx),axis=0))
    return RMS_vals



def plot_3D_state_residual_and_covariance(t_vec, x_hat, x_ref, y_labels, covariances=None, directory=None, custom_ylim=None, use_std=False):
    dx_N = x_ref - x_hat

    dx_i = np.sqrt(np.sum(np.square(dx_N[:,0:3]),axis=1)).reshape((len(t_vec), 1))
    dv_i = np.sqrt(np.sum(np.square(dx_N[:,3:6]),axis=1)).reshape((len(t_vec), 1))

    dx = np.hstack((dx_i, dv_i))

    sigma_x = np.sqrt((np.sum(np.diagonal(covariances[:,0:3,0:3],1,2),axis=1))).reshape((len(t_vec), 1))
    sigma_v = np.sqrt((np.sum(np.diagonal(covariances[:,3:6,3:6],1,2),axis=1))).reshape((len(t_vec), 1))

    sigma = np.hstack((sigma_x, sigma_v))
    t_vec_axis = t_vec / 3600.0 

    def reject_outliers(data, m = 3.):
        data_wo_nan = data[~np.isnan(data)]
        d = np.abs(data_wo_nan - np.median(data_wo_nan))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        return data_wo_nan[s<m]
    for i in range(len(dx[0])):
        plt.figure()
        plt.scatter(t_vec_axis, dx[:,i], s=2)
        # plt.plot(t_vec_axis, 3*sigma[:,i], 'r')
        
        plt.xlabel("Time [hr]")
        plt.ylabel(y_labels[i])
        annotate_residual(dx[:,i], y_labels[i])

        mean = np.nanmax(3*sigma[:,i])
        # mean = np.mean(reject_outliers(cov))
        plt.ylim([0, mean])
        # if covariances is not None:

        if custom_ylim is not None:
            plt.ylim(custom_ylim)
        if use_std:
            std = np.std(reject_outliers(dx[:,i]))
            plt.ylim([0, 5*std])
        if directory is not None:
            os.makedirs(directory ,exist_ok=True)
            fig_name = y_labels[i].replace("$", "").replace('\\', "").replace(" ","_")
            plt.savefig(directory + fig_name + ".pdf")

    RMS_vals = np.sqrt(np.mean(np.square(dx),axis=0))
    return RMS_vals



def plot_state_residual_and_covariance_consider(t_vec, x_hat, x_ref, y_labels, covariances=None, consider_covariances=None, directory=None, custom_ylim=None, use_std=False, sigma=3):
    dx = x_ref - x_hat

    def reject_outliers(data, m = 3.):
        data_wo_nan = data[~np.isnan(data)]
        d = np.abs(data_wo_nan - np.median(data_wo_nan))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        return data_wo_nan[s<m]
    t_vec_axis = t_vec / 3600.0 
    for i in range(len(dx[0])):
        plt.figure()
        plt.scatter(t_vec_axis, dx[:,i], s=2)
        plt.xlabel("Time [hr]")
        plt.ylabel(y_labels[i])
        annotate_residual(np.sqrt(np.square(dx[:,i])), y_labels[i])

        if covariances is not None:
            cov = np.sqrt(covariances[:,i,i])
            plt.plot(t_vec_axis,sigma*cov, c='r')
            plt.plot(t_vec_axis,-sigma*cov, c='r')
            mean = np.nanmean(sigma*cov)
            plt.ylim([-5*mean, 5*mean])

        if consider_covariances is not None:
            cov = np.sqrt(consider_covariances[:,i,i])
            plt.plot(t_vec_axis,sigma*cov, c='m')
            plt.plot(t_vec_axis,-sigma*cov, c='m')
            mean = np.nanmean(sigma*cov)

            # mean = np.mean(reject_outliers(cov))
            plt.ylim([-5*mean, 5*mean])

        if custom_ylim is not None:
            plt.ylim(custom_ylim)
        if use_std:
            std = np.std(reject_outliers(dx[:,i]))
            plt.ylim([-5*std, 5*std])
        if directory is not None:
            os.makedirs(directory ,exist_ok=True)
            fig_name = y_labels[i].replace("$", "").replace('\\', "")
            plt.savefig(directory + fig_name + ".pdf")

    RMS_vals = np.sqrt(np.mean(np.square(dx),axis=0))
    return RMS_vals


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse_2d(x, y, ax, cov=None, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
    
    See how and why this works: https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
    
    This function has made it into the matplotlib examples collection:
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py
    
    Or, once matplotlib 3.1 has been released:
    https://matplotlib.org/gallery/index.html#statistics
    
    I update this gist according to the version there, because thanks to the matplotlib community
    the code has improved quite a bit.
    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    if cov is None:
        cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor='red',
        fill=False,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_suite_example8(t, Y, x_truth, y_truth, R0, h, logger, filter, directory):
    # 1-a-i state estimate
    y_labels = np.array(["$x$", "$v_x$"])
    dX_RMS = plot_state_residual_and_covariance(
        logger.t_i,
        filter.logger.x_hat_i_plus,
        x_truth,
        y_labels,
        covariances=filter.logger.P_i_plus,
        directory=directory,
        use_std=False,
        sigma=2
    )

    # 1-a-ii True, noisy, estimated measurement plot
    y_pred = np.squeeze(np.array([h(filter.logger.x_hat_i_plus[i], np.array([])) for i in range(len(Y))]))
    r_sigma = np.sqrt(R0[0,0])

    plt.figure()
    plt.plot(t, y_truth, label='truth')
    plt.plot(t, Y, label='truth noisy')
    plt.plot(t, y_pred, label='estimated')
    plt.xlabel('time')
    plt.ylabel('Measurement')
    plt.legend()
    plt.savefig(directory+"measurement_plot.pdf")

    # 1-a-ii Measurement Residual
    plt.figure()
    plt.scatter(t, Y-y_pred, label='truth noisy')
    plt.hlines(2*r_sigma, t[0], t[-1], colors='red')
    plt.hlines(-2*r_sigma, t[0], t[-1], colors='red')
    plt.xlabel('time')
    plt.ylabel(r'$\delta y$')
    annotate_stats(Y-y_pred,"r")
    plt.savefig(directory+"measurement_residual.pdf")


    # 1-a-iii true and estimated state trajectory
    plt.figure()
    plt.plot(x_truth[:,0], x_truth[:,1], label='truth')
    plt.plot(
        filter.logger.x_hat_i_plus[:,0], 
        filter.logger.x_hat_i_plus[:,1], 
        label='estimated'
        )
    plt.scatter(
        filter.logger.x_hat_i_plus[0,0],
        filter.logger.x_hat_i_plus[0,1], 
        label='initial guess', 
        s=10
        )
    plt.scatter(
        filter.logger.x_hat_i_plus[-1,0],
        filter.logger.x_hat_i_plus[-1,1], 
        label='final state', s=10)
    plt.ylabel(r'$\dot{x}$')
    plt.xlabel(r'$x$')
    plt.legend()
    plt.savefig(directory+"state_trajectory.pdf")

    plt.figure()

    sigma_x = np.sqrt(filter.logger.sigma_i[-1,0])
    sigma_xd = np.sqrt(filter.logger.sigma_i[-1,1])

    try:
        x_coord, y_coord = filter.x_i_m1[:,0], filter.x_i_m1[:,1] # PF
        plt.scatter(x_coord, y_coord)
        confidence_ellipse_2d(filter.x_i_m1[:,0], filter.x_i_m1[:,1],plt.gca())

    except:
        x_coord, y_coord = filter.logger.x_hat_i_plus[-1,0], filter.logger.x_hat_i_plus[-1,1] #KF
        plt.scatter(x_coord, y_coord)
        confidence_ellipse_2d(x_coord, y_coord, plt.gca(), cov=filter.logger.P_i_plus[-1])

    # ellipse = matplotlib.patches.Ellipse(xy=(np.mean(x_coord),np.mean(y_coord)), 
    #         width=3*sigma_x, 
    #         height=3*sigma_xd,
    #         fill=False
    #         )
    # plt.gca().add_patch(ellipse)

    # plt.ylabel(r'$3\sigma_{\dot{x}}$')
    # plt.xlabel(r'$3\sigma_{x}$')
    plt.ylabel(r'$\dot{x}$')
    plt.xlabel(r'$x$')
    # plt.xlim([-4*sigma_x, 4*sigma_x])
    # plt.ylim([-4*sigma_xd, 4*sigma_xd])
    plt.savefig(directory+"state_cov.pdf")
