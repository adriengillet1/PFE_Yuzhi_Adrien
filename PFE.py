import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import pandas as pd


np.random.seed(123)
n = 100

# -----------------------------------------------------------
#   CREATION DES MATRICES
# -----------------------------------------------------------

def creation_matrice_ref():
    nb_mat_exterieur = 1000
    matrices_exterieur = [np.random.randn(n, n) for _ in range(nb_mat_exterieur)]
    mean_mat_exterieur = sum(matrices_exterieur) / nb_mat_exterieur
    mean_mat_exterieur[n//5:4*n//5, n//5:4*n//5] = 0

    nb_mat_milieu = 10000
    matrices_milieu = [np.random.randn(3*n//5, 3*n//5) for _ in range(nb_mat_milieu)]
    mean_mat_milieu = sum(matrices_milieu) / nb_mat_milieu
    mean_mat_milieu[n//5:2*n//5, n//5:2*n//5] = 0

    temp = np.zeros((n, n))
    temp[n//5:4*n//5, n//5:4*n//5] = mean_mat_milieu
    mean_mat_milieu = temp

    nb_mat_interieur = 100000
    matrices_interieur = [np.random.randn(n//5, n//5) for _ in range(nb_mat_interieur)]
    mean_mat_interieur = sum(matrices_interieur) / nb_mat_interieur

    temp = np.zeros((n, n))
    temp[2*n//5:3*n//5, 2*n//5:3*n//5] = mean_mat_interieur
    mean_mat_interieur = temp

    return mean_mat_exterieur + mean_mat_milieu + mean_mat_interieur


def creation_matrice_bruit(sd):
    nb_mat_exterieur = 1000
    matrices_exterieur = [
        np.random.randn(n, n) + np.random.normal(0, sd, (n, n))
        for _ in range(nb_mat_exterieur)
    ]
    mean_mat_exterieur = sum(matrices_exterieur) / nb_mat_exterieur
    mean_mat_exterieur[n//5:4*n//5, n//5:4*n//5] = 0

    nb_mat_milieu = 10000
    matrices_milieu = [
        np.random.randn(3*n//5, 3*n//5) + np.random.normal(0, sd, (3*n//5, 3*n//5))
        for _ in range(nb_mat_milieu)
    ]
    mean_mat_milieu = sum(matrices_milieu) / nb_mat_milieu
    mean_mat_milieu[n//5:2*n//5, n//5:2*n//5] = 0

    temp = np.zeros((n, n))
    temp[n//5:4*n//5, n//5:4*n//5] = mean_mat_milieu
    mean_mat_milieu = temp

    nb_mat_interieur = 100000
    matrices_interieur = [
        np.random.randn(n//5, n//5) + np.random.normal(0, sd, (n//5, n//5))
        for _ in range(nb_mat_interieur)
    ]
    mean_mat_interieur = sum(matrices_interieur) / nb_mat_interieur

    temp = np.zeros((n, n))
    temp[2*n//5:3*n//5, 2*n//5:3*n//5] = mean_mat_interieur
    mean_mat_interieur = temp

    return mean_mat_exterieur + mean_mat_milieu + mean_mat_interieur


def creation_matrice_leger_bruit():
    return creation_matrice_bruit(sd=0.05)

def creation_matrice_fort_bruit():
    return creation_matrice_bruit(sd=0.5)


# -----------------------------------------------------------
#   METRIQUES
# -----------------------------------------------------------

def mse(A, B):
    return np.mean((A - B)**2)

def mae(A, B):
    return np.mean(np.abs(A - B))

def t_test_value(A, B):
    tval, _ = ttest_ind(A.ravel(), B.ravel())
    return tval

def t_test_pvalue(A, B):
    _, pval = ttest_ind(A.ravel(), B.ravel())
    return pval


# -----------------------------------------------------------
#   SSIM (implémentation fidèle au code R)
# -----------------------------------------------------------

def ssim_small(img1, img2, K1=0.01, K2=0.03, L=1, window_size=10, sigma=1):

    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    img1 = normalize(img1)
    img2 = normalize(img2)

    half = window_size // 2
    coords = np.arange(-half, half+1)
    window = np.exp(-(coords[:, None]**2 + coords[None, :]**2) / (2 * sigma**2))
    window /= np.sum(window)

    def conv2(X, kernel):
        nr, nc = X.shape
        out = np.zeros_like(X)
        k = kernel.shape[0] // 2

        for i in range(nr):
            for j in range(nc):
                imin = max(0, i-k); imax = min(nr, i+k+1)
                jmin = max(0, j-k); jmax = min(nc, j+k+1)

                partX = X[imin:imax, jmin:jmax]
                partK = kernel[(imin-i+k):(imax-i+k),
                               (jmin-j+k):(jmax-j+k)]

                out[i,j] = np.sum(partX * partK)
        return out

    mu1 = conv2(img1, window)
    mu2 = conv2(img2, window)

    sigma1_sq = conv2(img1**2, window) - mu1**2
    sigma2_sq = conv2(img2**2, window) - mu2**2
    sigma12   = conv2(img1 * img2, window) - mu1 * mu2

    C1 = (K1*L)**2
    C2 = (K2*L)**2

    ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.nanmean(ssim_map)


# -----------------------------------------------------------
#   GAMMA INDEX
# -----------------------------------------------------------

def gamma_index(A, B, dose_diff_crit=0.03, dist_crit=3, pixel=1):
    A = np.asarray(A)
    B = np.asarray(B)
    nr, nc = A.shape

    # --- Grille des coordonnées ---
    x = np.arange(nr)
    y = np.arange(nc)
    X, Y = np.meshgrid(x, y, indexing='ij')  # X[i,j] = i, Y[i,j] = j

    # Préallocation du résultat
    gamma_map = np.zeros((nr, nc))

    # --- Parcours vectorisé pixel par pixel ---
    for i in range(nr):
        for j in range(nc):

            # Distance spatiale (n×n)
            dist = np.sqrt((pixel * (X - i))**2 + (pixel * (Y - j))**2)

            # Différence de dose (n×n)
            dose_diff = np.abs(A[i, j] - B)

            # Gamma (n×n)
            gamma_matrix = np.sqrt(
                (dose_diff / dose_diff_crit)**2 +
                (dist / dist_crit)**2
            )

            # Gamma minimal
            gamma_map[i, j] = np.min(gamma_matrix)

    return gamma_map



# -----------------------------------------------------------
#   FONCTION DE CALCUL GLOBALE
# -----------------------------------------------------------

def calcul_distances(matrice_reference, matrice_test):

    MSE  = mse(matrice_reference, matrice_test)
    MAE  = mae(matrice_reference, matrice_test)
    tval = t_test_value(matrice_reference, matrice_test)
    pval = t_test_pvalue(matrice_reference, matrice_test)
    SSIM = ssim_small(matrice_reference, matrice_test)
    GAMMA = gamma_index(matrice_reference, matrice_test)

    return {
        "MSE": MSE,
        "MAE": MAE,
        "t_value": tval,
        "p_value": pval,
        "SSIM": SSIM,
        "gamma_pass_rate": np.mean(GAMMA < 1),
    }


# -----------------------------------------------------------
#   EXECUTION
# -----------------------------------------------------------

matrice_reference = creation_matrice_ref()
matrice_test_sans_bruit = creation_matrice_ref()
matrice_test_leger_bruit = creation_matrice_leger_bruit()
matrice_test_fort_bruit = creation_matrice_fort_bruit()



plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.title("Référence")
plt.imshow(matrice_reference, cmap='viridis')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.title("Sans bruit")
plt.imshow(matrice_test_sans_bruit, cmap='viridis')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.title("Léger bruit")
plt.imshow(matrice_test_leger_bruit, cmap='viridis')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.title("Fort bruit")
plt.imshow(matrice_test_fort_bruit, cmap='viridis')
plt.colorbar()

plt.tight_layout()
plt.show()




result_sans_bruit = calcul_distances(matrice_reference, matrice_test_sans_bruit)
result_leger_bruit = calcul_distances(matrice_reference, matrice_test_leger_bruit)
result_fort_bruit = calcul_distances(matrice_reference, matrice_test_fort_bruit)


resultats = {
    "Sans bruit": result_sans_bruit,
    "Léger bruit": result_leger_bruit,
    "Fort bruit": result_fort_bruit
}

df = pd.DataFrame(resultats).T
df = df.applymap(lambda x: float(x))

print(df)

