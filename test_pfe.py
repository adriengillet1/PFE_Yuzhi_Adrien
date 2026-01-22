import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import convolve2d
from tqdm import tqdm

def creation_matrice_ref(n):
    matrice_ref = np.zeros((n, n))
    matrices_simulees = [[[] for _ in range(n)] for _ in range(n)]

    for i in range (n):
        for j in range (n):
            if (i < n//5 or i >= 4*n//5 or j < n//5 or j >= 4*n//5):
                taille_ech_aleatoire = np.random.randint(1, 400) + 800
                matrices_simulees[i][j].append(np.random.randn(taille_ech_aleatoire))
                matrice_ref[i][j] = np.array(matrices_simulees[i][j]).mean()
            elif (i >= 2*n//5 and i < 3*n//5 and j >= 2*n//5 and j < 3*n//5):
                taille_ech_aleatoire = np.random.randint(1, 40000) + 80000
                matrices_simulees[i][j].append(np.random.randn(taille_ech_aleatoire))
                matrice_ref[i][j] = np.array(matrices_simulees[i][j]).mean()
            else:
                taille_ech_aleatoire = np.random.randint(1, 4000) + 8000
                matrices_simulees[i][j].append(np.random.randn(taille_ech_aleatoire))
                matrice_ref[i][j] = np.array(matrices_simulees[i][j]).mean()

    return matrices_simulees, matrice_ref


def creation_matrice_bruit(n, sd):
    matrice_bruitee = np.zeros((n, n))
    matrices_simulees = [[[] for _ in range(n)] for _ in range(n)]

    for i in range (n):
        for j in range (n):
            if (i < n//5 or i >= 4*n//5 or j < n//5 or j >= 4*n//5):
                taille_ech_aleatoire = np.random.randint(1, 400) + 800
                if (j < 11*n//20 and j >= 9*n//20):
                    matrices_simulees[i][j].append(np.random.normal(0, sd + 1, taille_ech_aleatoire))
                    matrice_bruitee[i][j] = np.array(matrices_simulees[i][j]).mean()
                else:
                    matrices_simulees[i][j].append(np.random.randn(taille_ech_aleatoire))
                    matrice_bruitee[i][j] = np.array(matrices_simulees[i][j]).mean()

            elif (i >= 2*n//5 and i < 3*n//5 and j >= 2*n//5 and j < 3*n//5):
                taille_ech_aleatoire = np.random.randint(1, 40000) + 80000
                if (j < 11*n//20 and j >= 9*n//20):
                    matrices_simulees[i][j].append(np.random.normal(0, sd + 1, taille_ech_aleatoire))
                    matrice_bruitee[i][j] = np.array(matrices_simulees[i][j]).mean()
                else:
                    matrices_simulees[i][j].append(np.random.randn(taille_ech_aleatoire))
                    matrice_bruitee[i][j] = np.array(matrices_simulees[i][j]).mean()

            else:
                taille_ech_aleatoire = np.random.randint(1, 4000) + 8000
                if (j < 11*n//20 and j >= 9*n//20):
                    matrices_simulees[i][j].append(np.random.normal(0, sd + 1, taille_ech_aleatoire))
                    matrice_bruitee[i][j] = np.array(matrices_simulees[i][j]).mean()
                else:
                    matrices_simulees[i][j].append(np.random.randn(taille_ech_aleatoire))
                    matrice_bruitee[i][j] = np.array(matrices_simulees[i][j]).mean()

    return matrices_simulees, matrice_bruitee


def mse(A, B):
    return (A - B) ** 2


def mae(A, B):
    return np.abs(A - B)


def t_test_value(matrices_simulees_ref, matrice_simulees_bruitee):
    t_test_matrix = np.zeros((100, 100))
    n, m = 100, 100

    for i in tqdm(range(n)):
        for j in range(m):
            x = np.array(matrices_simulees_ref[i][j]).ravel()
            y = np.array(matrice_simulees_bruitee[i][j]).ravel()
            pval = ttest_ind(x, y).pvalue

            if (pval < 0.05):
                t_test_matrix[i][j] = 1
            else:
                t_test_matrix[i][j] = 0

    return t_test_matrix


def ssim(img1, img2, K1=0.01, K2=0.03, L=1, window_size=7, sigma=1):
    """
    Calcule le SSIM (Structural Similarity Index) entre deux images img1 et img2.

    Parameters :
        img1, img2 : images (matrices numpy)
        K1, K2     : constantes de stabilisation
        L          : plage dynamique des valeurs (ici on normalise donc L=1)
        window_size : taille du noyau gaussien utilisé pour lisser (fenêtre)
        sigma      : écart-type du noyau gaussien

    Return :
        SSIM moyen entre les deux images (float)
    """

    # Normalisation dans [0,1]
    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    img1 = normalize(img1)
    img2 = normalize(img2)

    # Construction d'un noyau gaussien pour le lissage local
    half = window_size // 2
    coords = np.arange(-half, half+1)
    # Noyau gaussien : window[i, j] = exp(-(i² + j²) / (2 sigma²))
    window = np.exp(-(coords[:, None]**2 + coords[None, :]**2) / (2 * sigma**2))
    window /= np.sum(window)

    # Convolution avec gestion des bords
    # "same" : même taille que l’image
    # "symm" : symétrie des bords
    conv = lambda X: convolve2d(X, window, mode='same', boundary='symm')

    # Moyennes locales
    mu1 = conv(img1)
    mu2 = conv(img2)

    # Variances et covariance locales
    sigma1_sq = conv(img1**2) - mu1**2
    sigma2_sq = conv(img2**2) - mu2**2
    sigma12   = conv(img1 * img2) - mu1 * mu2

    # Termes de stabilisation (évite les divisions par zéro)
    C1 = (K1*L)**2
    C2 = (K2*L)**2

    # Formule du SSIM pixel par pixel
    ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))

    # Retourne le SSIM moyen sans les valeurs NaN aux bords
    return ssim_map



def gamma_index(A, B, dose_diff_crit=0.03, dist_crit=3, pixel=1):
    """
    Calcule l'indice gamma entre deux matrices A et B qui représentent des cartes de dose.

    Parameters :
    -----------
    A, B : matrices numpy
    dose_diff_crit : float
        Critère de différence de dose 
    dist_crit : float
        Critère de glissement spatial entre les pixels (mm)
    pixel : float
        Taille d’un pixel (mm), utilisé pour convertir en distance réelle

    Return :
    --------
    gamma_map : matrice contenant, pour chaque pixel (i,j),
                la valeur minimale du gamma par rapport à tous les autres.
    """

    # Dimensions des matrices
    nr, nc = A.shape

    # Meshgrid permet de vectoriser les calculs de distance
    x = np.arange(nr)
    y = np.arange(nc)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Carte gamma initialisée à zéro (remplie pixel par pixel)
    gamma_map = np.zeros((nr, nc))

    for i in range(nr):
        for j in range(nc):
            # Calcul des distances et différences de dose entre le pixel (i,j) de la matrice A
            # et tous les pixels de la matrice B
            dist = np.sqrt((pixel * (X - i))**2 + (pixel * (Y - j))**2)
            dose_diff = np.abs(A[i, j] - B)

            # Calcul de la matrice gamma pour le pixel (i,j)
            gamma_matrix = np.sqrt(
                (dose_diff / dose_diff_crit)**2 +
                (dist / dist_crit)**2
            )

            # Valeur minimale du gamma pour le pixel (i,j)
            gamma_map[i, j] = np.min(gamma_matrix)

    return gamma_map






np.random.seed(123)
n = 100

matrices_simulees_ref, matrice_reference = creation_matrice_ref(n)
matrices_simulees_bruitees, matrice_bruitee = creation_matrice_bruit(n, 1)


plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)
plt.title("Matrice de référence")
plt.imshow(matrice_reference, cmap='viridis')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Matrice bruitée")
plt.imshow(matrice_bruitee, cmap='viridis')
plt.colorbar()

plt.tight_layout()
plt.show()


MSE = mse(matrice_reference, matrice_bruitee)
MAE = mae(matrice_reference, matrice_bruitee)
SSIM = ssim(matrice_reference, matrice_bruitee)
GAMMA_INDEX = gamma_index(matrice_reference, matrice_bruitee)
TTEST = t_test_value(matrices_simulees_ref, matrices_simulees_bruitees)

plt.subplot(2, 2, 1)
plt.title("TTEST")
plt.imshow(TTEST, cmap='viridis')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.title("MAE")
plt.imshow(MAE, cmap='viridis')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.title("SSIM")
plt.imshow(SSIM, cmap='viridis')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.title("Gamma Index")
plt.imshow(GAMMA_INDEX, cmap='viridis')
plt.colorbar()

plt.tight_layout()
plt.show()
