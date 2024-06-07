import numpy as np
import scipy as sp

def convmtx(h, N):
    """
    Determina uma matriz de convolução a partir de um vetor de entrada 'h'

    Args:
        h (np.array): Vetor de entrada, especificado como linha ou coluna.
        N (int): Comprimento do vetor a ser convoluído, especificado como um número inteiro positivo.

    Returns:
        np.array: Retorna a matriz de convolução H
    """
    H = sp.linalg.toeplitz(np.hstack((h, np.zeros(N-1))), np.zeros(N))
    return H.T