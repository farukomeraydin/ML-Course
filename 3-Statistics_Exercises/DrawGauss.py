import numpy as np 
import matplotlib.pyplot as plt 

def draw_gauss(mu=0, sigma=1, axispos=None): 
    if axispos == None: 
        axispos = mu 
    x = np.linspace(axispos - 5 * sigma, axispos + 5 * sigma , 1000) 
    y = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.e ** (-(x - mu) ** 2 / (2 * sigma ** 2)) 
    axis = plt.gca() 
    axis.set_title('Gauss Eğrisi', fontsize=14, fontweight='bold', pad=20) 
    axis.set_ylim([-0.4 / sigma, 0.4 / sigma]) 

    axis.set_xticks(np.arange(int(mu - 5 * sigma), int(mu + 5 * sigma + sigma), sigma)) 
    axis.spines['left'].set_position('center') 
    axis.spines['bottom'].set_position('center') 
    axis.spines['top'].set_color(None) 
    axis.spines['right'].set_color(None) 
    plt.plot(x, y) 
    plt.show()
    
draw_gauss(mu=15, sigma=2, axispos=15) 
