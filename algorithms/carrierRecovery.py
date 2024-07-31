import numpy as np
from optic.dsp.core import pnorm
from utils import plot4thPower
import matplotlib.pyplot as plt

def fourthPower(sigRx, Fs, plotSpectrum=False):
    """
    Compensa o deslocamento de frequência utilizando o método
    de quarta potência.

    Parameters
    ----------
    sigRx : np.array
        Sinal a ser compensado.

    Fs : int
        taxa de amostragem.

    plotSpectrum : bool, optional
        Retorna o espectro do sinal em quarta potência, by default False

    Returns
    -------
    tuple
        - sigRx (np.array): Sinal compensado.
        - indFO (float): Estimativa do deslocamento de frequência.
    
    Referências
    -----------
    [1] Digital Coherent Optical Systems, Architecture and Algorithms
    """
    
    try:
        nModes = sigRx.shape[1]
    except IndexError:
        sigRx = sigRx.reshape(len(sigRx), 1)
    
    NFFT     = sigRx.shape[0]
    axisFreq = Fs * np.fft.fftfreq(NFFT)
    
    time = np.arange(0, sigRx.shape[0]) * 1/Fs

    for indMode in range(nModes):
        
        # Elevar a quarta potência e aplica a FFT
        fourth_power = np.fft.fft(sigRx[:, indMode]**4)

        # Inferir o índice de valor máximo na frequência
        indFO = np.argmax(np.abs(fourth_power))
        
        # Obtenha a estimativa do deslocamento de frequência
        indFO = axisFreq[indFO]/4       
        
        # Compense o deslocamento de frequência
        sigRx[:, indMode] *= np.exp(-1j * 2 * np.pi * indFO * time)
    
    # Plote o espectro da quarta potência de um dos modos.
    if plotSpectrum:
        plot4thPower(sigRx, axisFreq)
        
    return sigRx, indFO

def viterbiCPR(sigRx, N=85, M=4):
    """
    Recupera a fase da portadora com o algoritmo Virterbi & Viterbi

    Parameters
    ----------
    sigRx : np.array
        Sinal de entrada para se obter a referência de fase.

    N : int, optional
        Comprimento do filtro, by default 85

    M : int, optional
        Ordem da potência, by default 4

    Returns
    -------
    tuple:
        sigRx (np.array): Constelação com referência de fase.
        phiTime (np.array): Estimativa de fase em cada modo.
    """

    phiTime = np.unwrap((np.angle(movingAverage(sigRx**M, N, window='laplacian')) / M) - np.pi/M, axis=0)
    phiTime = np.unwrap(4 * phiTime, axis=0) / 4

    sigRx = pnorm(sigRx * np.exp(-1j * phiTime))
    
    return sigRx, phiTime

def ddCPR(sigRx, symbTx, N=85):
    """
    Recupera a fase da portadora com o algoritmo direcionado por decisão.

    Parameters
    ----------
    sigRx : np.array
        Sinal de entrada para se obter a referência de fase.
        
    symbTx : np.array
        sequência de símbolos transmitido.

    N : int, optional
        Comprimento do filtro, by default 85

    Returns
    -------
    tuple:
        sigRx (np.array): Constelação com referência de fase.
        phiTime (np.array): Estimativa de fase em cada modo.
    """

    phiTime = np.angle(movingAverage(sigRx * pnorm(np.conj(symbTx)), N, window='DDlaplacian'))
    # remove as descontinuidades de fase.
    phiTime = np.unwrap(4 * phiTime) / 4
    # compensa o ruído de fase.
    sigRx = pnorm(sigRx * np.exp(-1j * phiTime))

    return sigRx, phiTime

def movingAverage(x, N, alpha=0.03, window='constant'):
    """
    Calcula a média móvel para um array 2D ao longo de cada coluna.

    Parameters
    ----------
    x : np.array
        Matriz 2D do tipo (M,N), onde M é a quantidade das amostras
        e N o número de colunas.

    N : int
        Comprimento da janela.

    alpha : float, optional
        Parâmetro de escala (dispersão da distribuição laplaciana), by default 0.03

    window : str, optional
        Define a janela da média móvel [constant, laplacian, DDlaplacian], by default 'constant'

    Returns
    -------
    np.array
        Matriz 2D contendo a média móvel ao longo de cada coluna.
    
    Raises
    ------
    ValueError
        Caso a janela não seja especificada de forma correta.
    
    ValueError
        Caso a janela tenha um comprimento maior que o sinal de entrada.
    """
    
    nModes = x.shape[1]
    
    if window == 'constant':
        h = np.ones(N) / N
    
    elif window == 'laplacian':
        w = np.arange(-N, N)
        h = np.exp(-np.abs(w)*alpha)
    
    elif window == 'DDlaplacian':
        w = np.arange(0, N)
        h = np.exp(-np.abs(w)*alpha)

    else:
        raise ValueError('Janela especificada incorretamente.')

    if len(h) > x.shape[0]:
        raise ValueError('A janela deve ser menor ou igual ao comprimento do sinal de entrada.')
    
    y = np.zeros(x.shape, dtype=x.dtype)

    for index in range(nModes):
        
        # calcula a média móvel 
        average = np.convolve(x[:, index], h, mode='same')
        
        # obtém a saída de mesmo comprimento da entrada
        y[:, index] = average
        
    return y