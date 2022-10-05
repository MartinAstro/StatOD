from ScenarioBaseClass import ScenarioBaseClass

class ScenarioPositions(ScenarioBaseClass):
    def __init__(self, config):
        super().__init__(config)
    
    def transform(self, function):

        # Define dimensionalization constants
        t_star = self.t_star
        l_star = self.l_star
        ms = self.ms
        ms2 = self.ms2

        # transform initial parameters
        self.x0[0:3] = function(self.x0[0:3], self.l_star)
        self.x0[3:6] = function(self.x0[3:6], self.ms)
        self.x0[6:9] = function(self.x0[6:9], self.ms2)
        
        self.P0[0:3,0:3] = function(self.P0[0:3,0:3], self.l_star**2)
        self.P0[3:6,3:6] = function(self.P0[3:6,3:6], self.ms**2)
        self.P0[6:9,6:9] = function(self.P0[6:9,6:9], self.ms2**2)

        self.t = function(self.t, t_star)
        self.Y = function(self.Y, l_star)
        self.R = function(self.R, l_star**2)
        self.Q0 = function(self.Q0,self.ms2**2)      

        try:
            self.tau = function(self.tau,self.ms2**2)      
        except:
            pass

        try:
            # transform the logger if the filter is available
            self.filter.logger.t_i = function(self.filter.logger.t_i, t_star)

            self.filter.logger.x_hat_i_plus[:,0:3] = function(self.filter.logger.x_hat_i_plus[:,0:3], l_star)
            self.filter.logger.x_hat_i_plus[:,3:6] = function(self.filter.logger.x_hat_i_plus[:,3:6], ms)
            self.filter.logger.x_hat_i_plus[:,6:9] = function(self.filter.logger.x_hat_i_plus[:,6:9], ms2)

            self.filter.logger.P_i_plus[:,0:3,0:3] = function(self.filter.logger.P_i_plus[:,0:3,0:3], l_star**2)
            self.filter.logger.P_i_plus[:,3:6,3:6] = function(self.filter.logger.P_i_plus[:,3:6,3:6], ms**2)
            self.filter.logger.P_i_plus[:,6:9,6:9] = function(self.filter.logger.P_i_plus[:,6:9,6:9], ms2**2)
        except:
            pass


