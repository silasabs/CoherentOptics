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

def fft_convolution(x, h):
    """ 
    Obtem o produto de x e h no domínio da frequência.

    Args:
        x (np.array): sinal de entrada
        h (np.array): coeficientes do filtro

    Returns:
        y np.array: sinal de saída após a convolução.
    """
    # Tamanho da saída
    Ny = x.shape[0] + h.shape[0] - 1 

    # Calcule as transformadas rápidas de Fourier
    # dos sinais no domínio do tempo
    X = np.fft.fft(x)
    H = np.fft.fft(h)

    # Realiza a convolução circular no domínio da frequência
    Y = X * H

    # Volta ao domínio do tempo
    y = np.fft.ifft(Y)

    # Corte o sinal para o comprimento de saida esperado
    return y[:Ny]

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