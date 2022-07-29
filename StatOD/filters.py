import numpy as np
from scipy.integrate import solve_ivp
from StatOD.dynamics import *
from StatOD.rotations import no_rotation
from numba import njit, jit, prange
import time
import copy
from abc import ABC, abstractmethod

from StatOD.utils import ProgressBar

@njit(cache=True)
def numba_inv(A):
    return np.linalg.inv(A)

def invert(A):
    eps = np.finfo(float).eps
    precision = np.log(1.0/eps) - np.log(np.linalg.cond(A))
    if precision <= 0.0:
        try:
            L = np.linalg.cholesky(A)
            A_inv = numba_inv(L).T@numba_inv(L)
        except:
            A_inv = numba_inv(A)
    else:
        A_inv = numba_inv(A)
    return A_inv

class FilterLogger():
    def __init__(self, N, samples, M=None, state_labels=None):

        self.state_labels = state_labels

        if self.state_labels is None:
            self.state_labels = np.array([r'$x_' + str(i) + r"$" for i in range(N)])

        self.N = N
        self.i = np.zeros((samples,))
        self.t_i = np.zeros((samples,))

        # Current Reference (?)
        self.x_i = np.zeros((samples, N))
        self.c_i = np.zeros((samples, N))
        
        # Current Deviation from State Reference
        self.dx_i_minus = np.zeros((samples, N,))
        self.dx_i_plus = np.zeros((samples, N,))

        # Covariance
        self.P_i_minus = np.zeros((samples, N, N))
        self.P_i_plus = np.zeros((samples, N, N))

        # Current Best Estimate
        self.x_hat_i_minus = np.zeros((samples, N))
        self.x_hat_i_plus = np.zeros((samples, N))

        # Current Best Estimate (w/ consider parameters)
        self.x_hat_c_i_minus = np.zeros((samples, N))
        self.x_hat_c_i_plus = np.zeros((samples, N))

        # State Transition Matrices 
        self.phi_ti_ti_m1 = np.zeros((samples, N, N))
        self.phi_ti_t0 = np.zeros((samples, N, N))

        self.sigma_i = np.zeros((samples, N))


        self.M = M # consider parameters

        if self.M is not None:
            # Current Deviation from State Reference (using consider parameters)
            self.dx_c_i_minus = np.zeros((samples, N,))
            self.dx_c_i_plus = np.zeros((samples, N,))

            self.P_c_i_minus = np.zeros((samples, N+M, N+M))
            self.P_c_i_plus = np.zeros((samples, N+M, N+M))
            
            self.theta_ti_ti_m1 = np.zeros((samples, N, M))
            self.theta_ti_t0 = np.zeros((samples, N, M))

    def log(self, data):
        empty_vec = np.full((self.N), np.nan)
        empty_mat = np.full((self.N, self.N), np.nan)

        idx = data.get('i', 0) - 1

        self.i[idx] = data.get('i', np.nan)
        self.t_i[idx] = data.get('t_i', np.nan)

        # Current Reference (?)
        self.x_i[idx] = data.get('x_i', empty_vec)
        
        # Current Deviation from State Reference
        self.dx_i_minus[idx] = data.get('dx_i_minus', empty_vec)
        self.dx_i_plus[idx] = data.get('dx_i_plus', empty_vec)

        # Covariance
        self.P_i_minus[idx] = data.get('P_i_minus', empty_mat)
        self.P_i_plus[idx] = data.get('P_i_plus', empty_mat)
        
        # Current Best Estimate
        self.x_hat_i_minus[idx] = data.get('x_hat_i_minus', empty_vec)
        self.x_hat_i_plus[idx] = data.get('x_hat_i_plus', empty_vec)

        # State Transition Matrices 
        self.phi_ti_ti_m1[idx] = data.get('phi_ti_ti_m1', empty_mat)
        self.phi_ti_t0[idx] = data.get('phi_ti_t0', empty_mat)

        self.sigma_i[idx] = data.get('sigma_i', empty_vec)

        if self.M is not None:        
            self.dx_c_i_minus[idx] = data.get('dx_c_i_minus', empty_vec)
            self.dx_c_i_plus[idx] = data.get('dx_c_i_plus', empty_vec)

            self.x_hat_c_i_minus[idx] = data.get('x_hat_c_i_minus', empty_vec)
            self.x_hat_c_i_plus[idx] = data.get('x_hat_c_i_plus', empty_vec)

            self.P_c_i_minus[idx] = data.get('P_c_i_minus', empty_mat)
            self.P_c_i_plus[idx] = data.get('P_c_i_plus', empty_mat)

            empty_mat = np.full((self.N, self.M), np.nan)

            self.theta_ti_ti_m1[idx] = data.get('theta_ti_ti_m1', empty_mat)
            self.theta_ti_t0[idx] = data.get('theta_ti_t0', empty_mat)
            

    def clear(self):
        samples = len(self.i)
        N = len(self.x_i[0])
        self.__init__(N, samples)

class FilterBase(ABC):
    def __init__(self, f_dict, h_dict, logger, events):

        self.logger = logger

        self.f = f_dict['f']
        self.dfdx = f_dict['dfdx']
        self.f_args = f_dict['f_args']
        self.f_integrate = f_dict.get('f_integrate', dynamics_ivp)


        self.Q_dt_fcn = f_dict.get('Q_fcn', None)
        self.Q_0 = f_dict.get('Q', None)
        self.Q_args = f_dict.get('Q_args', [])
        self.Q_DCM = f_dict.get('Q_DCM', no_rotation)

        self.h = h_dict['h']
        self.dhdx = h_dict['dhdx']
        self.h_args = h_dict['h_args']

        if hasattr(events, 'terminal'):
            self.terminate_upon_event = getattr(events, 'terminal')
        else:
            self.terminate_upon_event = False
        self.event_triggered = False
        self.events = events
        self.t_events = None
        self.y_events = None

        pass

    def get_process_noise(self, t_i, x_i):
        N = len(x_i)

        if self.Q_dt_fcn == None:
            Q_i_i_m1 = np.zeros((N,N))
        else:
            dt = t_i - self.t_i_m1
            if dt == 0:
                Q_i_i_m1 = np.zeros((N,N))
            else:
                dt = t_i - self.t_i_m1
                Q_i_i_m1 = np.zeros((N,N))
                if dt > 10*60: # 10 minutes
                    tk_list = np.arange(0, dt, step=1*60) # 3 minutes
                    tk_list = np.append(tk_list, dt)
                    for k in range(1,len(tk_list)):
                        dt = tk_list[k] - tk_list[k-1]
                        Q_i_i_m1 += np.array(self.Q_dt_fcn(dt, x_i, self.Q_0, self.Q_DCM(x_i), self.Q_args))
                else:   
                    Q_i_i_m1 = np.array(self.Q_dt_fcn(dt, x_i, self.Q_0, self.Q_DCM(x_i), self.Q_args))
        return Q_i_i_m1

    def predict_measurement(self, x_i, dx_i, h_args):
        h_i = np.array(self.h(x_i, h_args))
        H_i = np.array(self.dhdx(x_i, h_i, h_args))
        y_hat_i = h_i + H_i@dx_i
        return y_hat_i 

    def gather_event_data(self, t_i, sol):
        if type(self.events) != type(None):
            if len(sol.t_events[0]) > 0:
                self.event_triggered = True
                print("Found crossing")
                if self.t_events is not None:
                    self.t_event = np.vstack((self.t_events, sol.t_events))
                    self.y_event = np.vstack((self.y_events, sol.y_events))
                else:
                    self.t_events = sol.t_events
                    self.y_events = sol.y_events[:N]
            if self.terminate_upon_event:
                t_i = sol.t_events[0] # overwrite the current time step 

    @abstractmethod
    def propagate_forward(self):
        # take t_i_m1 to t_i
        # return t_i, x_i, phi(t_i, t_i_m1)
        pass
    
    @abstractmethod
    def time_update(self):
        # update dx_i_minus, P_i_minus
        pass
        
    @abstractmethod
    def process_observations(self):
        # return r_i, H_i, K_i
        pass

    @abstractmethod
    def measurement_update(self):
        # return dx_i_plus, P_i_plus, i
        pass

    @abstractmethod
    def get_logger_dict(self):
        pass

    
    def run(self, t_vec, y_vec, R_vec, f_arg_vec, h_arg_vec):
        pbar = ProgressBar(len(t_vec)-1, enable=True)
        for i in range(len(t_vec)):
            t_i = t_vec[i]
            y_i = y_vec[i]
            R_i = R_vec[i]
            f_args_i = f_arg_vec[i] # could be fcn
            h_args_i = h_arg_vec[i] # could be fcn
            fail = self.update(t_i, y_i, R_i, f_args_i, h_args_i)
            pbar.update(i)
        pbar.close()

class KalmanFilter(FilterBase):
    def __init__(self, t0, x0, dx0, P0, f_dict, h_dict, logger=None, events=None):
        super().__init__(f_dict, h_dict, logger, events)

        self.i = 0
        self.t_i_m1 = t0
        self.x_i_m1 = x0
        
        self.dx_i_m1_minus = np.zeros(np.shape(dx0))
        self.dx_i_m1_plus = dx0

        self.P_i_m1_minus = np.zeros(np.shape(P0))
        self.P_i_m1_plus = P0

        self.phi_0 = np.eye(len(x0))
        self.phi_i_m1 = np.eye(len(x0))
        
    def propagate_forward(self, t_i, x_i_m1, phi_i_m1):
        if t_i == self.t_i_m1:
            # return x_hat_i_m1_plus, phi_i_m1
            return copy.deepcopy(x_i_m1), copy.deepcopy(phi_i_m1)
        N = len(x_i_m1)
        Z_i_m1 = np.hstack((x_i_m1, phi_i_m1.reshape((-1))))
        sol = solve_ivp(self.f_integrate, 
                        [self.t_i_m1, t_i], 
                        Z_i_m1, 
                        args=(self.f, self.dfdx, self.f_args), 
                        atol=1E-14,
                        rtol=2.23E-14, 
                        method='RK45',
                        events=self.events)
        x_i = sol.y[:N,-1]
        phi_i = sol.y[N:,-1].reshape((N,N))
        self.gather_event_data(t_i,sol)
        return x_i, phi_i

    def time_update(self, dx_i_m1_plus, P_i_m1_plus, phi_i, Q_i_i_m1):
        dx_i_minus = phi_i@dx_i_m1_plus
        P_i_minus = phi_i@P_i_m1_plus@phi_i.T + Q_i_i_m1
        return dx_i_minus, P_i_minus

    def process_observations(self, x_i, P_i_minus, R_i, y_i):
        h_i = np.array(self.h(x_i, self.h_args))
        H_i = np.array(self.dhdx(x_i, h_i, self.h_args))
        r_i = y_i - h_i 
        K_i = P_i_minus@H_i.T@invert(H_i@P_i_minus@H_i.T + R_i)
        return r_i, H_i, K_i

    def measurement_update(self, dx_i_minus, P_i_minus, K_i, H_i, R_i, r_i):
        dx_i_plus = dx_i_minus + K_i@(r_i - H_i@dx_i_minus)
        G = np.eye(len(K_i)) - K_i@H_i
        P_i_plus = G@P_i_minus@G.T + K_i@R_i@K_i.T
        return dx_i_plus, P_i_plus

    def get_logger_dict(self):
        logger_data = {
            "i" : self.i,
            "t_i" : self.t_i_m1,
            "x_i" : self.x_i_m1,
            "dx_i_minus" : self.dx_i_m1_minus,
            "dx_i_plus" : self.dx_i_m1_plus,
            "P_i_minus" : self.P_i_m1_minus,
            "P_i_plus" : self.P_i_m1_plus,
            "x_hat_i_minus" : self.x_i_m1 + self.dx_i_m1_minus,
            "x_hat_i_plus" :self.x_i_m1 + self.dx_i_m1_plus,
            "phi_ti_ti_m1" : self.phi_i_m1,
            "sigma_i" : np.sqrt(np.diag(self.P_i_m1_plus))
        }
        return logger_data

    def update(self, t_i, y_i, R_i, f_args=None, h_args=None):
        fail = False
        if f_args is not None:
            self.f_args = f_args
        if h_args is not None:
            self.h_args = h_args

        x_i, phi_i = self.propagate_forward(t_i, self.x_i_m1, self.phi_0)#self.phi_i_m1)
        Q_i = self.get_process_noise(t_i, x_i)
        dx_i_minus, P_i_minus = self.time_update(self.dx_i_m1_plus, self.P_i_m1_plus, phi_i, Q_i)
        if np.any(np.isnan(dx_i_minus)) or np.any(np.isnan(P_i_minus)):
            print("NaNs Encountered")
            fail = True
            return fail

        if np.any(np.isnan(y_i)) or \
            (self.event_triggered and self.terminate_upon_event): 
            dx_i_plus, P_i_plus = dx_i_minus, P_i_minus
        else:
            r_i, H_i, K_i = self.process_observations(x_i, P_i_minus, R_i, y_i)
            dx_i_plus, P_i_plus = self.measurement_update(dx_i_minus, P_i_minus, K_i, H_i, R_i, r_i)
            if np.any(np.isnan(dx_i_plus)) or np.any(np.isnan(P_i_plus)):
                print("NaNs Encountered")
                fail = True
                return fail

        self.i += 1
        self.t_i_m1 = t_i
        self.x_i_m1 = x_i

        self.dx_i_m1_minus = dx_i_minus
        self.dx_i_m1_plus = dx_i_plus

        self.P_i_m1_minus = P_i_minus       
        self.P_i_m1_plus = P_i_plus       

        self.phi_i_m1 = phi_i
        
        if self.logger is not None:
            data = self.get_logger_dict()
            self.logger.log(data) 
        return fail

class ExtendedKalmanFilter(FilterBase):
    def __init__(self, t0, x0, dx0, P0, f_dict, h_dict, logger=None, events=None):
        super().__init__(f_dict, h_dict, logger, events)

        self.i = 0
        self.t_i_m1 = t0

        self.x_hat_i_m1_minus = x0
        self.x_hat_i_m1_plus = x0
        
        self.P_i_m1_minus = P0
        self.P_i_m1_plus = P0

        self.phi_i_m1 = np.eye(len(x0))
        self.phi_0 = np.eye(len(x0))

    def propagate_forward(self, t_i, x_hat_i_m1_plus, phi_i_m1):
        if t_i == self.t_i_m1:
            # return x_hat_i_m1_plus, phi_i_m1
            return copy.deepcopy(x_hat_i_m1_plus), copy.deepcopy(phi_i_m1)

        N = len(x_hat_i_m1_plus)
        Z_i_m1 = np.hstack((x_hat_i_m1_plus, phi_i_m1.reshape((-1))))
        event_fcn = self.events
        sol = solve_ivp(self.f_integrate, [self.t_i_m1, t_i], Z_i_m1, args=(self.f, self.dfdx, self.f_args), atol=1E-14, rtol=2.23E-14, events=event_fcn)
        x_i = sol.y[:N,-1]
        phi_i = sol.y[N:,-1].reshape((N,N))

        self.gather_event_data(t_i,sol)

        return x_i, phi_i

    def time_update(self, P_i_m1_plus, phi_i, Q_i):
        P_i_minus = phi_i@P_i_m1_plus@phi_i.T + Q_i
        if np.any(np.isnan(P_i_minus)):
            print("NaNs Encountered")
            self.failed = True
        return P_i_minus

    def process_observations(self, x_hat_i_minus, P_i_minus, R_i, y_i):
        h_i = np.array(self.h(x_hat_i_minus, self.h_args))
        H_i = np.array(self.dhdx(x_hat_i_minus, h_i, self.h_args))
        r_i = y_i - h_i 
        K_i = P_i_minus@H_i.T@invert(H_i@P_i_minus@H_i.T + R_i)
        return r_i, H_i, K_i

    def measurement_update(self, x_hat_i_minus, P_i_minus, K_i, H_i, R_i, r_i):
        x_hat_i_plus = x_hat_i_minus + K_i@r_i
        G = np.eye(len(K_i)) - K_i@H_i
        P_i_plus = G@P_i_minus@G.T + K_i@R_i@K_i.T
        if np.any(np.isnan(P_i_plus)):
            print("NaNs Encountered")
            self.failed = True

        return x_hat_i_plus, P_i_plus

    def get_logger_dict(self):
        logger_data = {
            "i" : self.i,
            "t_i" : self.t_i_m1,

            "P_i_minus" : self.P_i_m1_minus,
            "P_i_plus" : self.P_i_m1_plus,

            "x_hat_i_minus" : self.x_hat_i_m1_minus,
            "x_hat_i_plus" : self.x_hat_i_m1_plus,

            "phi_ti_ti_m1" : self.phi_i_m1,

            "sigma_i" : np.sqrt(np.diag(self.P_i_m1_plus))
        }
        return logger_data

    def update(self, t_i, y_i, R_i, f_args=None, h_args=None):
        if f_args is not None:
            self.f_args = f_args
        if h_args is not None:
            self.h_args = h_args

        x_hat_i_minus, phi_i = self.propagate_forward(t_i, self.x_hat_i_m1_plus, self.phi_0)
        Q_i = self.get_process_noise(t_i, x_hat_i_minus)
        P_i_minus = self.time_update(self.P_i_m1_plus, phi_i, Q_i)

        # if no obs, or event triggered and filter terminates upon event
        if np.any(np.isnan(y_i)) or \
            (self.event_triggered and self.terminate_upon_event): 
           x_hat_i_plus, P_i_plus = copy.deepcopy(x_hat_i_minus), copy.deepcopy(P_i_minus)
        else:
            r_i, H_i, K_i = self.process_observations(x_hat_i_minus, P_i_minus, R_i, y_i)
            x_hat_i_plus, P_i_plus = self.measurement_update(x_hat_i_minus, P_i_minus, K_i, H_i, R_i, r_i)

        self.i += 1
        self.t_i_m1 = t_i

        self.x_hat_i_m1_minus = x_hat_i_minus
        self.x_hat_i_m1_plus = x_hat_i_plus

        self.P_i_m1_minus = P_i_minus        
        self.P_i_m1_plus = P_i_plus        

        self.phi_i_m1 = phi_i

        if self.logger is not None:
            data = self.get_logger_dict()
            self.logger.log(data)
        
class NonLinearBatchFilter(FilterBase):
    def __init__(self, t0, x0, dx0, P0, f_dict, h_dict, logger=None, events=None, iterations=3, update_gamma=1.0):
        super().__init__(f_dict, h_dict, logger, events)

        self.i = 0
        self.k = 0 # major iteration

        # Initial conditions
        self.t_0 = t0
        self.phi_0 = np.eye(len(x0))
        self.x_hat_0 = x0
        self.P_0 = P0

        # full trajectory
        self.x_hat = np.zeros((len(logger.t_i), len(x0)))
        self.phi = np.zeros((len(logger.t_i), len(x0), len(x0)))

        # used only for speeding up dynamics propagation
        self.t_i = t0
        self.x_hat_i = x0
        self.phi_i = np.eye(len(x0))

        self.dx_k = np.zeros_like(x0)
        # self.dx_k = dx0
        self.Lambda, self.N = self.init_matrices(P0, self.dx_k)

        
        self.max_iterations = iterations
        self.update_gamma = update_gamma

    def init_matrices(self, P0, dx):
        if dx is not None and self.k != 0:
            Lambda = invert(P0)
            N = Lambda@dx
        else:
            Lambda = np.zeros(np.shape(P0))
            N = np.zeros(np.shape(dx))
        return Lambda, N

    def propagate_trajectory(self, t_vec, x_hat_0, phi_0):
        N = len(x_hat_0)
        Z_i_m1 = np.hstack((x_hat_0, phi_0.reshape((-1))))
        sol = solve_ivp(dynamics_ivp, [t_vec[0], t_vec[-1]], Z_i_m1, args=(self.f, self.dfdx, self.f_args), atol=1E-14, rtol=2.23E-14, t_eval=t_vec, method='RK45')

        x_hat = sol.y[:N].T.reshape((-1,N))
        phi = sol.y[N:].T.reshape((-1,N,N))

        return x_hat, phi

    def propagate_forward(self, t_i, x_hat_i_m1, phi_i_m1):
        pass

    def process_observations(self, x_hat_i_minus, phi_i, R_i, y_i):
        h_i = np.array(self.h(x_hat_i_minus, self.h_args))
        H_i = np.array(self.dhdx(x_hat_i_minus, h_i, self.h_args))
        r_i = y_i - h_i 

        if np.any(np.isnan(r_i)):
            return self.Lambda, self.N
        R_i_inv = np.linalg.inv(R_i)
        Lambda_i = self.Lambda + (H_i@phi_i).T@R_i_inv@H_i@phi_i
        N_i = self.N + (H_i@phi_i).T@R_i_inv@r_i
        return Lambda_i, N_i

    def time_update(self):
        pass

    def measurement_update(self):
        pass

    def update(self, t_i, y_i, R_i, f_args=None, h_args=None):
        if f_args is not None:
            self.f_args = f_args
        if h_args is not None:
            self.h_args = h_args

        # These attributes are needed for the logging
        self.t_i = t_i
        self.x_hat_i = self.x_hat[self.i]
        self.phi_i = self.phi[self.i]
        self.Lambda, self.N = self.process_observations(self.x_hat_i, self.phi_i, R_i, y_i)
        self.i += 1

        if self.logger is not None:
            data = self.get_logger_dict()
            self.logger.log(data)

    def get_logger_dict(self):
        if np.any(np.diag(self.Lambda) == 0.0):
            P_i = np.full(np.shape(self.Lambda), np.nan)
        else:
            try:
                P_i = invert(self.Lambda)
            except:
                P_i = np.full(np.shape(self.Lambda), np.nan)
        logger_data = {
            "i" : self.i,
            "t_i" : self.t_i,
            
            # dx_i_plus is infused into propagated x_hat_i_plus
            # i.e. no correction after the batch update
            "dx_i_plus" : np.zeros_like(self.x_hat_i),
            "x_hat_i_plus" : self.x_hat_i,
            "P_i_plus" : P_i,
            "phi_ti_t0" : self.phi_i,
        }
        return logger_data

    def run(self, t_vec, y_vec, R_vec, f_args_vec, h_args_vec, tol=1E-8):
        while (np.linalg.norm(self.dx_k) > tol or self.k == 0) and self.k < self.max_iterations:
            
            # Propagate the initial trajectory
            self.t = t_vec
            self.x_hat, self.phi = self.propagate_trajectory(t_vec, self.x_hat_0, self.phi_0)
            
            # Initialize information matrices
            self.Lambda, self.N = self.init_matrices(self.P_0, self.dx_k)

            # Add to the Lambda and N arguments by accumulating observations
            for i in range(len(t_vec)):
                self.update(
                    t_vec[i], 
                    y_vec[i], 
                    R_vec[i], 
                    np.array(f_args_vec[i]), 
                    np.array(h_args_vec[i])
                    )

            # Solve for the IC correction 
            Lambda_inv = invert(self.Lambda)
            dx_k_plus = Lambda_inv@self.N

            # Apply the correction 
            self.x_hat_0 += self.update_gamma*dx_k_plus
            self.dx_k = self.dx_k - dx_k_plus

            # Print state and update k
            print("Major Iteration: %d \t ||dx_k|| = %f" % (self.k, np.linalg.norm(self.dx_k)))
            self.k += 1

        # Once termination criteria is reached, propagate the updated IC and log
        self.x_hat, self.phi = self.propagate_trajectory(self.t, self.x_hat_0, self.phi_0)
        self.i = 0 # reset time index
        for i in range(len(t_vec)):
            self.update(
                t_vec[i], 
                y_vec[i], 
                R_vec[i], 
                np.array(f_args_vec[i]), 
                np.array(h_args_vec[i])
                )
        return

class SquareRootInformationFilter(FilterBase):
    def __init__(self, t0, x0, dx0, P0, f_dict, h_dict, logger=None, events=None):
        super().__init__(f_dict, h_dict, logger, events)

        self.i = 0
        self.t_i_m1 = t0
        self.x_i_m1 = x0
        
        self.dx_i_m1_minus = np.zeros(np.shape(dx0))
        self.dx_i_m1_plus = dx0

        self.P_i_m1_minus = np.zeros(np.shape(P0))
        self.P_i_m1_plus = P0

        self.phi_0 = np.eye(len(x0))
        self.phi_i_m1 = np.eye(len(x0))

        self.mu_i_m1_plus = np.zeros((3,1))

        self.R0 = np.linalg.cholesky(invert(P0)).T
        self.db0 = self.R0@dx0
        self.db_i_m1_plus = self.db0
        self.R_i_m1_plus = self.R0

    def test_failure(self, matrices):
        fail = False
        for matrix in matrices:
            if np.any(np.isnan(matrix)):
                print("NaNs Encountered")
                fail = True
        return fail

    def state_2_info(self, dx, P):
        R = np.linalg.cholesky(P).T
        db = R@dx
        return db, R

    def info_2_state(self, db, R):
        R_inv = invert(R)
        dx = R_inv@db
        P = R_inv@R_inv.T
        return dx, P

    def householder_transformation(self,A_input, N):
        A = copy.deepcopy(A_input)
        L = len(A) # Augmented Matrix = N + M
        for k in range(N):
            u = np.zeros((L,))
            sigma = np.sign(A[k,k])*np.linalg.norm(A[k:L,k])
            u[k] = A[k,k] + sigma
            A[k,k] = -sigma
            u[k+1:] = A[k+1:,k]

            for i in range(k+1, L):
                u[i] = A[i,k]
            beta = 1 / (sigma * u[k])
            # beta = 1.0/(sigma*u[k])

            for j in range(k+1, N+1):
                gamma = beta*np.sum([u[i] * A[i,j] for i in range(k, L)])
                for i in range(k,L):
                    A[i,j] = A[i,j] - gamma*u[i]

            for i in range(k+1, L):
                A[i,k] = 0.0
        return A

    def propagate_forward(self, t_i, x_i_m1, phi_i_m1):
        if t_i == self.t_i_m1:
            # return x_hat_i_m1_plus, phi_i_m1
            return copy.deepcopy(x_i_m1), copy.deepcopy(phi_i_m1)

        N = len(x_i_m1)
        Z_i_m1 = np.hstack((x_i_m1, phi_i_m1.reshape((-1))))
        event_fcn = self.events
        sol = solve_ivp(self.f_integrate, [self.t_i_m1, t_i], Z_i_m1, args=(self.f, self.dfdx, self.f_args), atol=1E-14, rtol=2.23E-14, method='RK45',events=event_fcn)
        x_i = sol.y[:N,-1]
        phi_i = sol.y[N:,-1].reshape((N,N))

        self.gather_event_data(t_i,sol)

        return x_i, phi_i

    def get_process_noise(self, t_i, x_i):
        N = len(x_i)
        if self.Q_dt_fcn == None:
            Q_i_i_m1 = np.zeros((N,N))
        else:
            dt = t_i - self.t_i_m1
            Q_i_i_m1 = np.array(self.Q_dt_fcn(dt, x_i, self.Q_0, self.Q_DCM(x_i), self.Q_args))
            Q_i_i_m1 = np.zeros_like(Q_i_i_m1)

            if dt > 10*60: # 10 minutes
                tk_list = np.arange(0, dt, step=1*60) # 3 minutes
                tk_list = np.append(tk_list, dt)
                for k in range(1,len(tk_list)):
                    dt = tk_list[k] - tk_list[k-1]
                    Q_i_i_m1 += np.array(self.Q_dt_fcn(dt, x_i, self.Q_0, self.Q_DCM(x_i), self.Q_args))
            else:   
                Q_i_i_m1 = np.array(self.Q_dt_fcn(dt, x_i, self.Q_0, self.Q_DCM(x_i), self.Q_args))
        return Q_i_i_m1

    def time_update(self, dt, db_i_m1_plus, R_i_m1_plus, phi_i, Gamma_i_i_m1, mu_i_m1):

        # w/ process noise
        n = len(db_i_m1_plus)
        q = len(mu_i_m1)

        # If no change in time, then skip 
        if dt == 0.0:
            db_tilde_u_i = np.zeros_like(mu_i_m1).squeeze()
            R_u_i_minus = np.eye(len(mu_i_m1))
            R_ux_i = np.zeros((len(mu_i_m1), len(db_i_m1_plus)))
            return db_i_m1_plus, R_i_m1_plus, db_tilde_u_i, R_u_i_minus, R_ux_i

        # Check if there is process noise
        try:
            R_u_i_m1 = np.linalg.cholesky(invert(self.Q_0)).T
        except:
            # If not, perform smaller householder transform.

            # Time update for information covariance 
            R_v_i = R_i_m1_plus@invert(phi_i) # TSB 5.10.69 (R_k_tilde) = Pre-transform R_i_minus

            # Time update for information state through householder transform, T
            MM = np.hstack((R_v_i, db_i_m1_plus.reshape((-1,1)))) # 5.10.88 second row
            TMM = self.householder_transformation(MM, n)

            R_i_minus = TMM[:n,:n]
            db_i_minus = TMM[:n,n]

            db_tilde_u_i = np.zeros_like(mu_i_m1)
            R_u_i_minus = np.eye(len(mu_i_m1))
            R_ux_i = np.zeros((len(mu_i_m1), len(db_i_m1_plus)))

            return db_i_minus, R_i_minus, db_tilde_u_i, R_u_i_minus, R_ux_i

        
        # convert control into information space 
        db_u_i_m1 = R_u_i_m1@mu_i_m1 # Recall that mu is the average control, not the actual control. 
        R_v_i = R_i_m1_plus@invert(phi_i) # TSB 5.10.69 (R_k_tilde) = Pre-transform R_i_minus

        #  pg 363 T.S.B.
        zeros_q_x_n = np.zeros((q,n))
        A = np.vstack([
            np.hstack((R_u_i_m1, zeros_q_x_n, db_u_i_m1.reshape(q,1))),
            np.hstack((-R_v_i@Gamma_i_i_m1, R_v_i, db_i_m1_plus.reshape((n,1))))
        ])
        A_prime = self.householder_transformation(A,len(A[0])-1)

        R_u_i_minus = A_prime[0:q, 0:q] # Time updated noise/control covariance 
        R_ux_i = A_prime[0:q, q:(q+n)]  # Time updated state contribution to control state 
        R_i_minus = A_prime[q:,q:(q+n)] # Time updated state covariance

        db_tilde_u_i = A_prime[0:q, -1] # Time updated noise state
        db_i_minus = A_prime[q:,-1]     # Time updated state  
        
        return db_i_minus, R_i_minus, db_tilde_u_i, R_u_i_minus, R_ux_i

    def process_observations(self, x_i, dx_i, P_i_minus, R_i, y_i):
        h_i = np.array(self.h(x_i, self.h_args))
        H_i = np.array(self.dhdx(x_i, h_i, self.h_args))
        r_i = y_i - h_i 
        eps_i = r_i - H_i@dx_i

        V_i = np.linalg.cholesky(R_i).T
        V_i_inv = invert(V_i)
        H_tilde_i = V_i_inv@H_i
        eps_tilde_i = V_i_inv@eps_i
        r_tilde_i = V_i_inv@r_i

        return r_tilde_i, H_tilde_i, eps_tilde_i

    def measurement_update(self, db_i_minus, R_i_minus, H_tilde_i, r_tilde_i):
        N = len(db_i_minus)
        db_i_minus_k = db_i_minus
        R_i_minus_k = R_i_minus
        N = len(R_i_minus_k)

        # each observation in Y_i must be processed separately (index k)
        for k in range(len(r_tilde_i)): 
            G = np.vstack([
                np.hstack([R_i_minus_k, db_i_minus_k.reshape((-1,1))]),
                np.hstack([H_tilde_i[k], r_tilde_i[k]])]
            )
            G_new = self.householder_transformation(G, N)
            R_i_plus = G_new[:N,:N]
            db_i_plus = G_new[:N,N]
            e_tilde_i_k = G_new[N,N]

            R_i_minus_k = R_i_plus
            db_i_minus_k = db_i_plus

        return db_i_plus, R_i_plus, e_tilde_i_k

    def get_logger_dict(self):
        logger_data = {
            "i" : self.i,
            "t_i" : self.t_i_m1,

            "x_i" : self.x_i_m1,

            "dx_i_minus" : self.dx_i_m1_minus,
            "dx_i_plus" : self.dx_i_m1_plus,

            "P_i_minus" : self.P_i_m1_minus,
            "P_i_plus" : self.P_i_m1_plus,

            "x_hat_i_minus" : self.x_i_m1 + self.dx_i_m1_minus,
            "x_hat_i_plus" :self.x_i_m1 + self.dx_i_m1_plus,

            "phi_ti_ti_m1" : self.phi_i_m1,
        }
        return logger_data
        
    def update(self, t_i, y_i, R_i, f_args=None, h_args=None, mu_i=None):
        fail = False
        if f_args is not None:
            self.f_args = f_args
        if h_args is not None:
            self.h_args = h_args
        if mu_i is None:
            self.mu_i = np.zeros((3,1)) # no control input

        x_i, phi_i = self.propagate_forward(t_i, self.x_i_m1, self.phi_0)#self.phi_i_m1)
        Gamma_i = self.get_process_noise(t_i, x_i)

        db_i_minus, R_i_minus, db_tilde_u_i, R_u_i_minus, R_ux_i = self.time_update(
            t_i - self.t_i_m1,
            self.db_i_m1_plus, 
            self.R_i_m1_plus, 
            phi_i,
            Gamma_i,
            self.mu_i
            )
        fail = self.test_failure([db_i_minus, R_i_minus])
        dx_i_minus, P_i_minus = self.info_2_state(db_i_minus, R_i_minus)
        if np.any(np.isnan(y_i)) or \
            (self.event_triggered and self.terminate_upon_event): 
            db_i_plus, R_i_plus = db_i_minus, R_i_minus
        else:
            r_tilde_i, H_tilde_i, eps_tilde_i = self.process_observations(x_i, dx_i_minus, P_i_minus, R_i, y_i)
            db_i_plus, R_i_plus, e_tilde_i_k = self.measurement_update(db_i_minus, R_i_minus, H_tilde_i, r_tilde_i)
            fail = self.test_failure([db_i_plus, R_i_plus])

        dx_i_plus, P_i_plus = self.info_2_state(db_i_plus, R_i_plus)

        # measurement updated past control value (that isn't used in the next time update (?))  
        mu_i_m1_plus = invert(R_u_i_minus)@(db_tilde_u_i - R_ux_i@dx_i_plus) 
        
        self.i += 1
        self.t_i_m1 = t_i
        self.x_i_m1 = x_i

        self.dx_i_m1_minus = dx_i_minus
        self.dx_i_m1_plus = dx_i_plus

        self.P_i_m1_minus = P_i_minus       
        self.P_i_m1_plus = P_i_plus       

        self.phi_i_m1 = phi_i

        self.db_i_m1_plus = db_i_plus
        self.R_i_m1_plus = R_i_plus
        
        if self.logger is not None:
            data = self.get_logger_dict()
            self.logger.log(data) 
        return fail

    
    # def smooth(self):
        # TSB 5.10.105 -- Smoothed Covariance
        # TSB 5.10.96 -- Smoothed State 

class UnscentedKalmanFilter(FilterBase):
    def __init__(self, t0, x0, dx0, P0, alpha, kappa, beta, f_dict, h_dict, logger=None, events=None):
        super().__init__(f_dict, h_dict, logger, events)

        self.i = 0
        self.t_i_m1 = t0

        self.x_hat_i_m1_minus = x0
        self.x_hat_i_m1_plus = x0
        
        self.P_i_m1_minus = P0
        self.P_i_m1_plus = P0

        self.phi_i_m1 = np.eye(len(x0))

        self.alpha = alpha
        self.kappa = kappa
        self.beta = beta

        n = len(x0)
        self.gamma = np.zeros((n,))
        for k in range(n):
            self.gamma[k] = np.sqrt(n + alpha**2*(self.kappa+n) - n)

        self.w_m, self.w_c = self.generate_sigma_weights()

        self.logger = logger
        self.phi_0 = np.eye(len(x0))

    def generate_sigma_weights(self):
        n = len(self.x_hat_i_m1_minus)
        w_m = np.zeros((2*n+1))
        w_c = np.zeros((2*n+1))

        lambda_k = self.alpha**2*(self.kappa + n) - n
        for k in range(2*n+1):
            if k == 0:
                w_c[k] = lambda_k/(lambda_k + n) + (1 - self.alpha**2 + self.beta)
            else:
                w_c[k] = 1/(2*(lambda_k + n))

        for k in range(2*n+1):
            if k == 0:
                w_m[k] = lambda_k/(lambda_k + n) 
            else:
                w_m[k] = 1/(2*(lambda_k + n))

        return w_m, w_c

    def get_sigma_vecs(self,P):
        # # S = np.linalg.cholesky(P).T
        # U, D, V = np.linalg.svd(P)
        # S = V@np.diag(np.sqrt(D))@invert(V)

        W, V = np.linalg.eig(P)
        W = np.real_if_close(W)
        S = np.sqrt(W)*V

        return S

    def propagate_forward(self, t_i, x_hat_i_m1_plus, P_i_m1_plus):
        N = len(x_hat_i_m1_plus)
        sigma_vecs = self.get_sigma_vecs(P_i_m1_plus)
        sigma_i_m1_plus = np.zeros((2*N+1,N))
        sigma_i_m1_plus[0] = x_hat_i_m1_plus
        event_fcn = self.events
        for k in range(1, len(sigma_i_m1_plus)):
            if k < N+1:
                sigma_i_m1_plus[k] = x_hat_i_m1_plus + self.gamma[k-1]*sigma_vecs[:,k-1]
            else:
                sigma_i_m1_plus[k] = x_hat_i_m1_plus - self.gamma[(k-1)-N]*sigma_vecs[:,(k-1)-(N)]

        sol = solve_ivp(self.f_integrate, [self.t_i_m1, t_i], sigma_i_m1_plus.reshape((-1,)), args=(self.f, self.dfdx, self.f_args), atol=1E-14, rtol=2.23E-14, events=event_fcn)
        x_i = sol.y[:,-1].reshape((2*N+1,N))
        self.gather_event_data(t_i,sol)

        return x_i

    def time_update(self, sigma_points, Q_i):
        N = len(sigma_points[0])
        P_i_minus = np.zeros((N,N))
        x_hat_i_minus = np.sum([
            self.w_m[k]*sigma_points[k] for k in range(0, 2*N+1)
            ], axis=0)
        for k in range(0,2*N+1):
            P_i_minus += self.w_c[k]*np.outer((sigma_points[k] - x_hat_i_minus),(sigma_points[k] - x_hat_i_minus)) 
        
        P_i_minus += Q_i
        return x_hat_i_minus, P_i_minus

    def process_observations(self, x_hat_i_minus, sigma_points, y_i, R_i):
        y_hat_vec = np.zeros((len(sigma_points), len(y_i)))
        for k in range(0, len(sigma_points)):
            y_hat_vec[k] = np.array(self.h(sigma_points[k], self.h_args))
        
        N = len(x_hat_i_minus)
        M = len(y_i)
        y_hat_i_minus = np.zeros((M,))
        P_xy_i_minus = np.zeros((N, M))
        P_yy_i_minus = np.zeros((M,M))
        for k in range(len(y_hat_vec)):
            y_hat_i_minus += self.w_m[k]*y_hat_vec[k]

        for k in range(len(y_hat_vec)):
            P_yy_i_minus += self.w_c[k]*np.outer((y_hat_vec[k] - y_hat_i_minus), (y_hat_vec[k] - y_hat_i_minus)) #+ R_i
            P_xy_i_minus += self.w_c[k]*np.outer((sigma_points[k]-x_hat_i_minus), (y_hat_vec[k] - y_hat_i_minus))

        P_yy_i_minus += R_i
        K_i = P_xy_i_minus@invert(P_yy_i_minus)
        if np.any(np.isnan(K_i)) or np.any(np.isnan(P_yy_i_minus)):
            print("NaNs Encountered")
            self.failed = True

        return y_hat_i_minus, K_i, P_yy_i_minus

    def measurement_update(self, x_hat_i_minus, P_i_minus, K_i, P_yy_i, y_i, y_hat_i):
        x_hat_i_plus = x_hat_i_minus + K_i@(y_i - y_hat_i)
        P_i_plus = P_i_minus - K_i@(P_yy_i)@K_i.T
        if np.any(np.isnan(P_i_plus)):
            print("NaNs Encountered")
            self.failed = True

        return x_hat_i_plus, P_i_plus

    def get_logger_dict(self):
        logger_data = {
            "i" : self.i,
            "t_i" : self.t_i_m1,

            "P_i_minus" : self.P_i_m1_minus,
            "P_i_plus" : self.P_i_m1_plus,

            "x_hat_i_minus" : self.x_hat_i_m1_minus,
            "x_hat_i_plus" : self.x_hat_i_m1_plus,

            "phi_ti_ti_m1" : self.phi_i_m1,
        }
        return logger_data

    def update(self, t_i, y_i, R_i, f_args=None, h_args=None):
        if f_args is not None:
            self.f_args = f_args
        if h_args is not None:
            self.h_args = h_args

        Q_i = self.get_process_noise(t_i, self.x_hat_i_m1_plus)
        sigma_points = self.propagate_forward(t_i, self.x_hat_i_m1_plus, self.P_i_m1_plus)
        x_hat_i_minus, P_i_minus = self.time_update(sigma_points, Q_i)
        
        # if no obs, or event triggered and filter terminates upon event
        if np.any(np.isnan(y_i)) or \
            (self.event_triggered and self.terminate_upon_event): 
            x_hat_i_plus, P_i_plus = x_hat_i_minus, P_i_minus
        else:
            y_hat_i_minus, K_i, P_yy_i_minus= self.process_observations(x_hat_i_minus, sigma_points, y_i, R_i)
            x_hat_i_plus, P_i_plus = self.measurement_update(x_hat_i_minus, P_i_minus, K_i, P_yy_i_minus, y_i, y_hat_i_minus)

        self.i += 1
        self.t_i_m1 = t_i

        self.x_hat_i_m1_minus = x_hat_i_minus
        self.x_hat_i_m1_plus = x_hat_i_plus

        self.P_i_m1_minus = P_i_minus        
        self.P_i_m1_plus = P_i_plus        

        self.phi_i_m1 = None

        if self.logger is not None:
            data = self.get_logger_dict()
            self.logger.log(data)

class Smoother():
    def __init__(self, logger):
        self.logger = copy.deepcopy(logger)

    def update(self):
        # https://arl.cs.utah.edu/resources/Kalman%20Smoothing.pdf or
        # https://en.wikipedia.org/wiki/Kalman_filter#Rauch%E2%80%93Tung%E2%80%93Striebel 
        i = len(self.logger.t_i) - 2 # total updates - 1 

        while i >= 0:
            # (timestep = i, measurements = i) 
            x_hat_i_plus = self.logger.x_hat_i_plus[i] 
            P_i_plus = self.logger.P_i_plus[i]

            # (timestep = i + 1, measurements = i) 
            phi_i_p1_i = self.logger.phi_ti_ti_m1[i+1] 
            P_i_p1_minus = self.logger.P_i_minus[i+1] 
            P_i_p1_plus = self.logger.P_i_plus[i+1] 
            x_hat_i_p1_minus = self.logger.x_hat_i_minus[i+1] 
            x_hat_i_p1_plus = self.logger.x_hat_i_plus[i+1] 

            S_i = P_i_plus@phi_i_p1_i.T@invert(P_i_p1_minus)
            new_x_hat_i_plus = x_hat_i_plus + S_i@(x_hat_i_p1_plus - x_hat_i_p1_minus)
            new_P_i_plus = P_i_plus + S_i@(P_i_p1_plus - P_i_p1_minus)@S_i.T

            self.logger.P_i_plus[i] = new_P_i_plus
            self.logger.x_hat_i_plus[i] = new_x_hat_i_plus
            self.logger.dx_i_plus[i] = new_x_hat_i_plus - self.logger.x_i[i]

            i -= 1

class ConsiderCovarianceFilter(FilterBase):
    def __init__(self, t0, x0, dx0, c0, dc0, P_xx_0, P_cc_0, f_dict, h_dict, logger=None, events=None):
        super().__init__(f_dict, h_dict, logger, events)

        self.i = 0
        self.t_i_m1 = t0
        self.x_i_m1 = x0
        
        self.dx_i_m1_minus = np.zeros(np.shape(dx0))
        self.dx_i_m1_plus = dx0

        self.dc_i_m1_minus = np.zeros(np.shape(dc0))
        self.dc_i_m1_plus = dc0

        self.P_xx_i_m1 = P_xx_0
        self.P_cc_i_m1 = P_cc_0
        self.P_xc_i_m1 = np.zeros_like(P_cc_0)

        self.M_xx_i_m1 = invert(P_xx_0)
        self.M_cc_i_m1 = invert(P_cc_0)


        # self.P_i_m1_minus = np.zeros(np.shape(P0))
        # self.P_i_m1_plus = P0

        self.phi_0 = np.eye(len(x0))
        self.phi_i_m1 = np.eye(len(x0))
           
    def propagate_forward(self, t_i, x_i_m1, phi_i_m1):
        N = len(x_i_m1)
        Z_i_m1 = np.hstack((x_i_m1, phi_i_m1.reshape((-1))))
        sol = solve_ivp(dynamics_ivp, [self.t_i_m1, t_i], Z_i_m1, args=(self.f, self.dfdx, self.f_args), atol=1E-14, rtol=2.23E-14, method='RK45')
        x_i = sol.y[:N,-1]
        phi_i = sol.y[N:,-1].reshape((N,N))
        self.gather_event_data(t_i,sol)

        return x_i, phi_i

    def time_update(self, dx_i_m1_plus, P_i_m1_plus, phi_i, Q_i_i_m1):
        dx_i_minus = phi_i@dx_i_m1_plus
        P_i_minus = phi_i@P_i_m1_plus@phi_i.T + Q_i_i_m1
        return dx_i_minus, P_i_minus

    def process_observations(self, x_i, c_i, P_i_minus, R_i, y_i):
        h_i = np.array(self.h(x_i, c_i, self.h_args))
        r_i = y_i - h_i 

        H_x_i = np.array(self.dhdx(x_i, h_i, self.h_args))
        H_c_i = np.array(self.dhdc(c_i, h_i, self.h_args))

        return r_i, H_x_i, H_c_i

    def measurement_update(self, dn_x, M_xx_i_plus, M_xc_plus, phi_i, H_x_i, H_c_i, R_i, r_i):

        R_inv = invert(R_i)
        M_xx_i_plus = M_xx_i_plus + phi_i.T@H_x_i.T@R_inv@H_x_i@phi_i
        M_xc_i_plus = M_xc_plus + phi_i.T@H_x_i@R_inv@H_c_i

        P_x_plus = invert(M_xx_i_plus)
        S_xc_plus = -P_x_plus@M_xc_i_plus

        dn_x = dn_x + phi_i.T@H_x_i.T@R_inv@r_i

        return dn_x, M_xx_i_plus, M_xc_i_plus, S_xc_plus, P_x_plus

    def get_logger_dict(self):
        logger_data = {
            "i" : self.i,
            "t_i" : self.t_i_m1,

            "x_i" : self.x_i_m1,

            "dx_i_minus" : self.dx_i_m1_minus,
            "dx_i_plus" : self.dx_i_m1_plus,

            "P_i_minus" : self.P_i_m1_minus,
            "P_i_plus" : self.P_i_m1_plus,

            "x_hat_i_minus" : self.x_i_m1 + self.dx_i_m1_minus,
            "x_hat_i_plus" :self.x_i_m1 + self.dx_i_m1_plus,

            "phi_ti_ti_m1" : self.phi_i_m1,
        }
        return logger_data

    def update(self, t_i, y_i, R_i, f_args=None, h_args=None):
        fail = False
        if f_args is not None:
            self.f_args = f_args
        if h_args is not None:
            self.h_args = h_args

        x_i, phi_i = self.propagate_forward(t_i, self.x_i_m1, self.phi_0)#self.phi_i_m1)
        Q_i = self.get_process_noise(t_i, x_i)
        dx_i_minus, P_i_minus = self.time_update(self.dx_i_m1_plus, self.P_i_m1_plus, phi_i, Q_i)
        if np.any(np.isnan(dx_i_minus)) or np.any(np.isnan(P_i_minus)):
            print("NaNs Encountered")
            fail = True
            return fail

        if np.any(np.isnan(y_i)):
            dx_i_plus, P_i_plus = dx_i_minus, P_i_minus
        else:
            r_i, H_i, K_i = self.process_observations(x_i, P_i_minus, R_i, y_i)
            dx_i_plus, P_i_plus = self.measurement_update(dx_i_minus, P_i_minus, K_i, H_i, R_i, r_i)
            if np.any(np.isnan(dx_i_plus)) or np.any(np.isnan(P_i_plus)):
                print("NaNs Encountered")
                fail = True
                return fail

        self.i += 1
        self.t_i_m1 = t_i
        self.x_i_m1 = x_i

        self.dx_i_m1_minus = dx_i_minus
        self.dx_i_m1_plus = dx_i_plus

        self.P_i_m1_minus = P_i_minus       
        self.P_i_m1_plus = P_i_plus       

        self.phi_i_m1 = phi_i
        
        if self.logger is not None:
            data = self.get_logger_dict()
            self.logger.log(data) 
        return fail

class SequentialConsiderCovarianceFilter(FilterBase):
    def __init__(self, t0, x0, dx0, c0, dc0, P_xx_0, P_cc_0, f_dict, h_dict, logger=None):
        self.i = 0
        self.t_i_m1 = t0
        self.x_i_m1 = x0
        self.c_i_m1 = c0

        # state error, parameters NOT considered (CKF) 
        self.dx_i_m1_minus = np.zeros(np.shape(dx0))
        self.dx_i_m1_plus = dx0

        # consider parameter error
        self.dc_i_m1_minus = np.zeros(np.shape(dc0))
        self.dc_i_m1_plus = dc0

        # state error, parameters considered (SCCF)
        self.dx_c_i_m1_minus = np.zeros(np.shape(dx0))
        self.dx_c_i_m1_plus = dx0

        # state covariance, parameters NOT considered (CKF)
        self.P_i_m1_minus = np.zeros(np.shape(P_xx_0))
        self.P_i_m1_plus = P_xx_0

        # state & parameter covariance, parameters considered (SCCF)
        self.P_xx_i_m1 = P_xx_0
        self.P_cc_i_m1 = P_cc_0
        self.P_xc_i_m1 = np.zeros_like(P_cc_0)

        # STMs
        self.phi_0 = np.eye(len(x0))
        self.phi_i_m1 = np.eye(len(x0))

        self.theta_0 = np.zeros((len(x0), len(c0)))
        self.theta_i_m1 = np.zeros((len(x0), len(c0)))

        self.f = f_dict['f']
        self.dfdx = f_dict['dfdx']
        self.dfdc = f_dict['dfdc']
        self.f_args = f_dict['f_args']
        self.f_consider_mask = f_dict['f_consider_mask'].astype(bool)

        self.Q_dt_fcn = f_dict.get('Q_fcn', None)
        self.Q_0 = f_dict.get('Q', None)
        self.Q_args = f_dict.get('Q_args', [])
        self.Q_DCM = f_dict.get('Q_DCM', no_rotation)

        self.h = h_dict['h']
        self.dhdx = h_dict['dhdx']
        self.dhdc = h_dict['dhdc']
        self.h_args = h_dict['h_args']
        self.h_consider_mask = h_dict['h_consider_mask'].astype(bool)

        # Compute S_xc_i
        h0 = np.array(self.h(x0, self.h_args))
        H_x_0 = np.array(self.dhdx(x0, h0, self.h_args))

        # Sympy still needs the state and non-consider parameters
        required_args = np.append(x0, self.h_args[~self.h_consider_mask])
        H_c_0 = np.array(self.dhdc(c0, h0, required_args))
        M = len(h0)
        R_i = np.zeros((M,M))
        K0 = self.P_i_m1_plus@H_x_0.T@(H_x_0@self.P_i_m1_plus@H_x_0.T + R_i)
        
        self.S_xc_i_m1_plus = -K0@H_c_0

        self.logger = logger

    def propagate_forward(self, t_i, x_i_m1, c_i_m1, phi_i_m1, theta_i_m1):
        N = len(x_i_m1)
        M = len(c_i_m1)
        Z_i_m1 = np.hstack((x_i_m1, c_i_m1, phi_i_m1.reshape((-1)), theta_i_m1.reshape((-1))))
        sol = solve_ivp(consider_dynamics_ivp, [self.t_i_m1, t_i], Z_i_m1, 
                        args=(self.f, self.dfdx, self.dfdc, self.f_args, N, M, self.f_consider_mask), 
                        atol=1E-14, 
                        rtol=2.23E-14, 
                        method='RK45')
        x_i = sol.y[:N,-1]
        c_i = sol.y[N:N+M,-1]

        phi_i = sol.y[N+M:(N+M + N**2),-1].reshape((N,N))
        theta_i = sol.y[(N+M + N**2):,-1].reshape((N,M))
        self.gather_event_data(t_i,sol)

        return x_i, c_i, phi_i, theta_i

    def time_update(self, dx_i_m1_plus, dx_c_i_m1_plus, dc, P_i_m1_plus, S_xc_i_m1_plus, P_cc_i_minus, phi_i, theta_i, Q_i_i_m1):
        P_i_minus = phi_i@P_i_m1_plus@phi_i.T + Q_i_i_m1 # Covariance without knowledge of consider parameters
        S_xc_i_minus = phi_i@S_xc_i_m1_plus + theta_i # Sensitivity to measurements given consider parameters (?)
        P_xx_i_minus = P_i_minus + S_xc_i_minus@P_cc_i_minus@S_xc_i_minus.T # State covariance with consider parameter covariance inflation
        P_xc_i_minus = S_xc_i_minus@P_cc_i_minus # Coupling between consider parameters and state 

        P_c_i_minus = np.block([[P_xx_i_minus, P_xc_i_minus],[P_xc_i_minus.T, P_cc_i_minus]]) # Augmented State Covariance, parameters considered

        dx_i_minus = phi_i@dx_i_m1_plus # state time update
        # dx_c_i_minus = phi_i@dx_c_i_m1_plus + theta_i@dc # state time update, parameters considered
        dx_c_i_minus = phi_i@dx_c_i_m1_plus + S_xc_i_minus@dc # Discussion Board 
        
        return dx_i_minus, dx_c_i_minus, P_i_minus, P_c_i_minus, S_xc_i_minus

    def process_observations(self, x_i, c_i, y_i):
        h_i = np.array(self.h(x_i, self.h_args))
        r_i = y_i - h_i 

        required_args = np.append(x_i, self.h_args[~self.h_consider_mask])
        H_x_i = np.array(self.dhdx(x_i, h_i, self.h_args))
        H_c_i = np.array(self.dhdc(c_i, h_i, required_args))

        return r_i, H_x_i, H_c_i

    def measurement_update(self, dx_i_minus, dc, dx_c_i_minus, P_i_minus, P_c_i_minus, S_xc_i_minus, H_x_i, H_c_i, R_i, r_i):
        
        N = len(dx_i_minus)
        eye = np.eye(N)
        K_i = P_i_minus@H_x_i.T@invert(H_x_i@P_i_minus@H_x_i.T + R_i)
        P_i_plus = (eye - K_i@H_x_i)@P_i_minus@(eye - K_i@H_x_i).T + K_i@R_i@K_i.T # Update covariance (parameters NOT considered)

        S_xc_i_plus = (eye - K_i@H_x_i)@S_xc_i_minus - K_i@H_c_i
        P_cc_minus = P_c_i_minus[N:, N:]
        P_xx_i_plus = P_i_plus + S_xc_i_plus@P_cc_minus@S_xc_i_plus.T #CKF covariance inflated by consider covariances
        P_xc_i_plus = S_xc_i_plus@P_cc_minus

        P_c_i_plus = np.block([[P_xx_i_plus, P_xc_i_plus],[P_xc_i_plus.T, P_cc_minus]])
        
        dx_i_plus = dx_i_minus + K_i@(r_i - H_x_i@dx_i_minus)
        dx_c_i_plus = dx_i_plus + S_xc_i_plus@dc # Notes
        # dx_c_i_plus = dx_c_i_minus + S_xc_i_plus@dc # Intuition

        return dx_i_plus, dx_c_i_plus, P_i_plus, P_c_i_plus, S_xc_i_plus

    def get_logger_dict(self):
        logger_data = {
            "i" : self.i,
            "t_i" : self.t_i_m1,
            "x_i" : self.x_i_m1,
            "c_i" : self.c_i_m1,

            "dx_i_minus" : self.dx_i_m1_minus,
            "dx_i_plus" : self.dx_i_m1_plus,

            "dx_c_i_minus" : self.dx_c_i_m1_minus,
            "dx_c_i_plus" : self.dx_c_i_m1_plus,

            "P_i_minus" : self.P_i_m1_minus,
            "P_i_plus" : self.P_i_m1_plus,

            "P_c_i_minus" : self.P_c_i_m1_minus,
            "P_c_i_plus" : self.P_c_i_m1_plus,

            "x_hat_i_minus" : self.x_i_m1 + self.dx_i_m1_minus,
            "x_hat_i_plus" :self.x_i_m1 + self.dx_i_m1_plus,

            "x_hat_c_i_minus" : self.x_i_m1 + self.dx_c_i_m1_minus,
            "x_hat_c_i_plus" :self.x_i_m1 + self.dx_c_i_m1_plus,

            "phi_ti_ti_m1" : self.phi_i_m1,
            "theta_ti_ti_m1" : self.theta_i_m1,
        }
        return logger_data

    def update(self, t_i, y_i, R_i, f_args=None, h_args=None):
        fail = False
        if f_args is not None:
            self.f_args = f_args
        if h_args is not None:
            self.h_args = h_args

        x_i, c_i, phi_i, theta_i = self.propagate_forward(t_i, self.x_i_m1, self.c_i_m1, self.phi_0, self.theta_0)#self.phi_i_m1)
        Q_i = self.get_process_noise(t_i, x_i)
        dx_i_minus, dx_c_i_minus, P_i_minus, P_c_i_minus, S_xc_i_minus = self.time_update(
                                                                self.dx_i_m1_plus, 
                                                                self.dx_c_i_m1_plus, 
                                                                self.dc_i_m1_plus, 
                                                                self.P_i_m1_plus, 
                                                                self.S_xc_i_m1_plus, 
                                                                self.P_cc_i_m1, 
                                                                phi_i, 
                                                                theta_i, 
                                                                Q_i)
        if np.any(np.isnan(dx_i_minus)) or np.any(np.isnan(P_i_minus)):
            print("NaNs Encountered")
            fail = True
            return fail

        if np.any(np.isnan(y_i)):
            dx_i_plus, P_i_plus = dx_i_minus, P_i_minus
            dx_c_i_plus, P_c_i_plus = dx_c_i_minus, P_c_i_minus
            S_xc_i_plus = S_xc_i_minus
        else:
            r_i, H_x_i, H_c_i = self.process_observations(x_i, c_i, y_i)
            dx_i_plus, dx_c_i_plus, P_i_plus, P_c_i_plus, S_xc_i_plus = self.measurement_update(
                                                                dx_i_minus, 
                                                                self.dc_i_m1_plus, 
                                                                dx_c_i_minus,
                                                                P_i_minus, 
                                                                P_c_i_minus, 
                                                                S_xc_i_minus, 
                                                                H_x_i, 
                                                                H_c_i, 
                                                                R_i, 
                                                                r_i)                
            if np.any(np.isnan(dx_i_plus)) or np.any(np.isnan(P_i_plus)):
                print("NaNs Encountered")
                fail = True
                return fail

        self.i += 1
        self.t_i_m1 = t_i
        self.x_i_m1 = x_i
        self.c_i_m1 = c_i

        self.dx_i_m1_minus = dx_i_minus
        self.dx_i_m1_plus = dx_i_plus

        self.dx_c_i_m1_minus = dx_c_i_minus
        self.dx_c_i_m1_plus = dx_c_i_plus

        self.P_i_m1_minus = P_i_minus       
        self.P_i_m1_plus = P_i_plus       

        self.P_c_i_m1_minus = P_c_i_minus       
        self.P_c_i_m1_plus = P_c_i_plus      

        self.S_xc_i_m1_minus = S_xc_i_minus 
        self.S_xc_i_m1_plus = S_xc_i_plus 

        self.phi_i_m1 = phi_i
        self.theta_i_m1 = theta_i
        
        if self.logger is not None:
            data = self.get_logger_dict()
            self.logger.log(data) 
        return fail

class ParticleFilter(FilterBase):
    def __init__(self, t0, x_0_k, f_dict, h_dict, logger=None, events=None):
        super().__init__(f_dict, h_dict, logger, events)

        self.i = 0
        self.N = len(x_0_k)

        self.t_i_m1 = t0
        self.x_i_m1 = x_0_k
        self.w_i_m1 = np.full((self.N,), 1.0/self.N)
        
    def propagate_forward(self, t_i, x_i_m1):
        N = len(x_i_m1)
        Z_i_m1 = x_i_m1.reshape((-1,))
        sol = solve_ivp(dynamics_ivp_particle, [self.t_i_m1, t_i], Z_i_m1, args=(self.f, N, self.f_args), atol=1E-14, rtol=2.23E-14, method='RK45')
        x_i = sol.y[:,-1].reshape((N,-1))
        self.gather_event_data(t_i,sol)

        return x_i

    def time_update(self, x_i, Q_i_i_m1):
        v0 = np.array([0.0,0.0])
        noise = np.random.multivariate_normal(v0, Q_i_i_m1, size=len(x_i))
        x_i_minus = x_i + noise
        return x_i_minus

    def process_observations(self, x_i, y_i):
        M = getattr(y_i, '__len__', lambda:1)()
        r_i = np.zeros((len(x_i), M))
        for i in prange(len(r_i)):
            r_i[i] = y_i - self.h(x_i[i], self.h_args)
        return r_i

    def measurement_update(self, R_i, r_i):
        w_i_k = np.zeros((len(r_i)))
        m = len(r_i[0])
        R_half = np.linalg.det(R_i)**(1/2.0)
        R_inv = np.linalg.inv(R_i)
        coef = 1/((2*np.pi)**(m/2)*R_half)
        for i in prange(len(r_i)):
            r = r_i[i]
            expo = -1/2*r.T@R_inv@r
            w_i_k[i] = coef*np.exp(expo)
        return w_i_k
    
    def resample(self,x_i_k, w_i_k):
        x_i_k_plus = np.zeros_like(x_i_k)
        w_i_k = w_i_k/np.sum(w_i_k)
        cdf = np.cumsum(w_i_k)
        u_i_k = np.random.uniform(0,1,size=len(x_i_k))
        for i in prange(len(x_i_k)):
            k = np.where(cdf >= u_i_k[i])[0][0] 
            x_i_k_plus[i] = copy.deepcopy(x_i_k[k])
        w_i_k = 1/self.N
        return x_i_k_plus, w_i_k

    def get_logger_dict(self):
        logger_data = {
            "i" : self.i,
            "t_i" : self.t_i_m1,
            "x_i" : np.mean(self.x_i_m1, axis=0),

            "x_hat_i_plus" : np.mean(self.x_i_m1, axis=0),
            
            "P_i_plus" : np.diag(np.std(self.x_i_m1,axis=0)**2),
            "sigma_i" : np.std(self.x_i_m1,axis=0),

        }
        return logger_data

    def update(self, t_i, y_i, R_i, f_args=None, h_args=None):
        fail = False
        if f_args is not None:
            self.f_args = f_args
        if h_args is not None:
            self.h_args = h_args

        x_i = self.propagate_forward(t_i, self.x_i_m1)#self.phi_i_m1)
        Q_i = self.get_process_noise(t_i, x_i)
        x_i_minus = self.time_update(x_i, Q_i)

        if np.any(np.isnan(y_i)):
            x_i_plus = x_i_minus
        else:
            r_i = self.process_observations(x_i_minus, y_i)
            w_i_minus = self.measurement_update(R_i, r_i)
            x_i_plus, w_i_plus = self.resample(x_i_minus, w_i_minus)

        self.i += 1
        self.t_i_m1 = t_i
        self.x_i_m1 = x_i_plus
        self.w_i_m1 = w_i_plus
        
        if self.logger is not None:
            data = self.get_logger_dict()
            self.logger.log(data) 
        return fail