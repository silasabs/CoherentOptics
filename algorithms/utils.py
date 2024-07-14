import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def convmtx(h, N):
    """
    Determina uma matriz de convolução a partir de um vetor de entrada 'h'

    Parameters
    ----------
    h : np.array
        matriz de entrada, especificado como linha ou coluna.
    
    N : int
        Comprimento do vetor a ser convoluído, especificado como um número inteiro positivo.

    Returns
    -------
    H: np.array:
        Matriz de convolução H
    """

    H = sp.linalg.toeplitz(np.hstack((h, np.zeros(N-1))), np.zeros(N))
    return H.T

def plot4thPower(sigRx, axisFreq):
    """

    Plote o espectro da quarta potência do sinal sigRx
    em dB.
    
    Args:
        sigRx (np.array): sinal de entrada.
        axisFreq (np.array): eixo de frequências.
    """
    
    plt.plot(axisFreq, 10*np.log10(np.abs(np.fft.fft(sigRx[:, 0]**4))), label=r"$|FFT(s[k]^4)|$")
    plt.ylabel('Amplitude [dB]')
    plt.xlabel(r'$f$')
    plt.legend()
    plt.grid()