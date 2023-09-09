import statistics 

def disp_ztable(): 
    nd = statistics.NormalDist() 
    print('-' * 76) 
    print(' z' + 6 * ' ', end='') 
    for i in range(10): 
        f = i * 0.01 
        print(f'{f:<7.2f}', end='') 
    print() 
    print('-' * 76) 
    z = -3.6 
    while z <= 0.05: 
        print(f'{-0.0:<8.1f}' if z > 0 else f'{z:<8.1f}', end='') 
        for i in range(10): 
            f = i * 0.01 
            cd = nd.cdf(z - f) 
            print(f'{cd:<8.4f}'[1:], end='') 
        print()
        z += 0.1
        
disp_ztable()
