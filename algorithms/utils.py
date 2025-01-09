import numpy as np
from numpy.fft import fft
import scipy as sp
import matplotlib.pyplot as plt

def convmtx(h, N):
    """
    Determina uma matriz de convolução a partir de um vetor de entrada 'h'

    Parameters
    ----------
    h : np.array
        Matriz de entrada, especificado como linha ou coluna.
    
    N : int
        Comprimento do vetor a ser convoluído, especificado como um número inteiro positivo.

    Returns
    -------
    H: np.array:
        Matriz de convolução H
    """

    H = sp.linalg.toeplitz(np.hstack((h, np.zeros(N-1))), np.zeros(N))
    return H.T

def freqHCD(Fc, Fs, D, NFFT, L):
    """
    Resposta em frequência da dispersão cromática.

    Parameters
    ----------
    Fc : float
        Frequência central.

    Fs : float
        Frequência de amostragem.

    D : int
        Dispersão cromática [ps/nm/km]

    NFFT : int
        Tamanho da FFT

    L : int
        Comprimento do enlace [m].

    Returns
    -------
    np.array
        Resposta em frequência para dispersão cromática.
    """
   
    c = 299792458   # velocidade da luz no vacuum
    λ = c/Fc        # comprimento de onda

    beta2 = -(D * λ**2) / (2 * np.pi * c)
    omega = 2 * np.pi * Fs * np.fft.fftfreq(NFFT)

    return np.exp(-1j * beta2/2 * omega**2 * L)

def plot4thPower(sigRx, axisFreq):
    """
    Plote o espectro da quarta potência do sinal sigRx em dB.

    Parameters
    ----------
    sigRx : np.array
        Sinal de entrada.

    axisFreq : np.array
        Eixo de frequências normalizadas.
    """
    f4 = 10*np.log10(np.abs(fft(sigRx**4)))
    fo = np.argmax(f4)

    plt.plot(axisFreq, f4, label=r"$|FFT(s[k]^4)|$")
    plt.plot(axisFreq[fo], f4[fo], 'x', label=r"$4f_0$")
    plt.ylabel('Amplitude [dB]')
    plt.xlabel(r'$f$')
    plt.legend()
    plt.grid()