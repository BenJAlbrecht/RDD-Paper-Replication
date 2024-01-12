import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

class models_class:

    def __init__(self, df):
        self.df = df
        self.cluster = None
        
        # Model estimates.
        self.b1 = None      # Quadratic estimates.
        self.b2 = None      # Cubic estimates.
        self.b3 = None      # Quartic estimates.

    
    # Fit function, gets intercept and coefficient estimates for each model.
    def fit(self):

        self.cluster = self.df['distid'].astype(str) + '_' + self.df['decade'].astype(str) + '_' + self.df['state'].astype(str)

        # Create quadratic, cubic, quartic forcing variables.
        self.df['fv2'], self.df['fv3'], self.df['fv4'] = self.df['fv'] ** 2, self.df['fv'] ** 3, self.df['fv'] ** 4

        # Fit additional models.
        self.quadratic_fit()
        self.cubic_fit()
        self.quartic_fit()

    
    # Quadratic model fitting.
    def quadratic_fit(self):
        quad_formula = 'next ~ fv + fv2 + win + fv:win + fv2:win'
        reg_quad = smf.ols(quad_formula, data=self.df)

        quad_result = reg_quad.fit(cov_type='cluster', cov_kwds={'groups': self.cluster}, use_t=True)
        self.b1 = quad_result.params


    # Cubic model fitting.
    def cubic_fit(self):
        cubic_formula = 'next ~ fv + fv2 + fv3 + win + fv:win + fv2:win + fv3:win'
        reg_cubic = smf.ols(cubic_formula, data=self.df)

        cubic_result = reg_cubic.fit(cov_type='cluster', cov_kwds={'groups': self.cluster}, use_t=True)
        self.b2 = cubic_result.params


    # Quartic model fitting.
    def quartic_fit(self):
        quart_formula = 'next ~ fv + fv2 + fv3 + fv4 + win + fv:win + fv2:win + fv3:win + fv4:win'
        reg_quart = smf.ols(quart_formula, data=self.df)

        quart_result = reg_quart.fit(cov_type='cluster', cov_kwds={'groups': self.cluster}, use_t=True)
        self.b3 = quart_result.params


    # Output graphs.
    def graph_models(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7, 5))
        fig.suptitle('Additional Models', fontsize=14)

        # X axis for all models
        x1 = np.linspace(-1, 0)
        x2 = np.linspace(0, 1)


        # Quadratic specification.
        y_quad1 = self.b1[0] + self.b1[1] * x1 + self.b1[2] * (x1 **2)
        y_quad2 = (self.b1[0] + self.b1[3]) + (self.b1[1] + self.b1[4]) * x2 + (self.b1[2] + self.b1[5]) * (x2 ** 2)

        ax1.plot(x1, y_quad1)
        ax1.plot(x2, y_quad2)
        ax1.axvline(x=0, color='k', linestyle='--', ymax=0.92)
        ax1.set_title('Quadratic')

        treatment1 = str(round(self.b1[3],3))
        ax1.annotate('Treatment:' + treatment1, 
             xy=(-1, 0.89), c='red', weight='bold')
        

        # Cubic specification.
        y_cubic1 = self.b2[0] + self.b2[1] * x1 + self.b2[2] * (x1**2) + self.b2[3] * (x1**3)
        y_cubic2 = (self.b2[0] + self.b2[4]) + (self.b2[1] + self.b2[5]) * x2 + (self.b2[2] + self.b2[6]) * (x2**2) + (self.b2[3] + self.b2[7]) * (x2**3)

        ax2.plot(x1, y_cubic1)
        ax2.plot(x2, y_cubic2)
        ax2.axvline(x=0, color='k', linestyle='--', ymax=0.92)
        ax2.set_title("Cubic")

        treatment2 = str(round(self.b2[4],3))
        ax2.annotate('Treatment:' + treatment2,
             xy=(-1, 0.89), c='red', weight='bold')
        

        # Quartic specification.
        y_quart1 = self.b3[0] + self.b3[1] * x1 + self.b3[2] * (x1**2) + self.b3[3] * (x1**3) + self.b3[4] * (x1**4)
        y_quart2 = (self.b3[0] + self.b3[5]) + (self.b3[1] + self.b3[6]) * x2 + (self.b3[2] + self.b3[7]) * (x2**2) + (self.b3[3] + self.b3[8]) * (x2**3) + (self.b3[4] + self.b3[9]) * (x2**4)

        ax3.plot(x1, y_quart1)
        ax3.plot(x2, y_quart2)
        ax3.axvline(x=0, color='k', linestyle='--', ymax=0.92)
        ax3.set_title("Cubic")

        treatment3 = str(round(self.b3[5],3))
        ax3.annotate('Treatment:' + treatment3,
             xy=(-1, 0.89), c='red', weight='bold')


        plt.show()
