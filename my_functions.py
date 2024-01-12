import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function for mapping years to decades.
def map_decade(year):
    return (year - 1949) // 10 + 1 if year >= 1950 else 1


# Function for graphing linear model.
def graph_linear(lin_result):
    estimates = lin_result.params

    xx1 = np.linspace(-1, 0)
    xx2 = np.linspace(0, 1)

    yy1 = estimates[0] + (estimates[1])*xx1 
    yy2 = estimates[0] + (estimates[1] + estimates[3])*xx2 + estimates[2]

    plt.plot(xx1, yy1, label='Control')
    plt.plot(xx2, yy2, label='Treated')
    plt.axvline(x=0, color='k', linestyle='--', label='Threshold')

    plt.title('Incumbency Effect')
    plt.xlabel('Election t')
    plt.ylabel('Election t+1')
    plt.legend(loc='upper left')

    plt.show()