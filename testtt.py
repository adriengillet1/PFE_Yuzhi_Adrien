import numpy as np


x = [1, 3, 6, 8]
cdf_x = [1/4, 2/4, 3/4, 1]

y = [2, 4, 6, 8, 9]
cdf_y = [1/5, 2/5, 3/5, 4/5, 1]

t = [0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0]





def create_fdr(cdf, ech, t):
    cdf_index = []
    F = []

    for i in range(len(cdf)):
        for j in range(len(t)):  
            if cdf[i] == t[j]:
                cdf_index.append(j)
                break

    for i in range(len(cdf_index)):
        if i == 0:
            F.append(np.repeat(ech[i], cdf_index[i] + 1))
        else:
            F.append(np.repeat(ech[i], cdf_index[i] - cdf_index[i - 1]))
    
    F = np.concatenate(F)
    return(F)

def calculate_k(t):
    k = []
    k.append(t[0])
    for i in range(1, len(t)):
        k.append(round(t[i] - t[i - 1], 10))
    return k


print("FDR X:", create_fdr(cdf_x, x, t))
print("FDR Y:", create_fdr(cdf_y, y, t))
print("K values:", calculate_k(t))