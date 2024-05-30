import numpy as np

def next_power_of_2(n):
    """
        Determina a próxima potência de 2
    Args:
        n (int): valor a ser aproximado para uma potência de 2
    Returns:
        int: potência de 2 mais próxima de n
    """
    return 1 << (int(np.log2(n - 1)) + 1)

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