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
    Recupera a fase da portadora com o algoritmo Virterbi & Viterbi considerando
    uma janela laplaciana

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

    phiTime = np.unwrap(np.angle(movingAverage(sigRx**M, N, alpha, window='laplacian')) / M - np.pi/M, period=2*np.pi/M, axis=0)
    # compensa o ruído de fase
    sigRx = pnorm(sigRx * np.exp(-1j * phiTime))
    
    return sigRx, phiTime

def mlviterbiCPR(sigRx, H, M=4):
    """
    Recupera a fase da portadora com o algoritmo Virterbi & Viterbi considerando
    um filtro ótimo.

    Parameters
    ----------
    sigRx : np.array
        Sinal de entrada para se obter a referência de fase.

    H : np.array
        Coeficientes do filtro de máxima verosimilhança.

    M : int, optional
        Ordem da potência, by default 4

    Returns
    -------
    tuple:
        sigRx (np.array): Constelação com referência de fase.
        phiTime (np.array): Estimativa de fase em cada modo.
    """
    
    phiTime = np.unwrap(np.angle(movingAverage(sigRx**M, H=H, window='viterbi')) / M - np.pi/M, period=2*np.pi/M, axis=0)
    # compensa o ruído de fase
    sigRx = pnorm(sigRx * np.exp(-1j * phiTime))
    
    return sigRx, phiTime

def mlfilterVV(sigRx, OSNRdB, delta_lw, Rs, N, M=4):
    """
    Calcula os coeficientes do filtro de máxima verossimilhança (ML) para o algoritmo 
    Viterbi&Viterbi, que depende da relação sinal-ruído e da magnitude do ruído de fase.

    Args:
        sigRx (np.array): sinal compensado pelo deslocamento de frequência. 
        
        OSNRdB (float): OSNR do canal em dB.
        
        delta_lw (int): Soma das larguras de linha do laser do oscilador local
                        e transmissor.
        
        Rs (int): Taxa de símbolos. [símbolos/segundo]
        
        N (int): Número de símbolos passados e futuros na janela. O comprimento
                 do filtro é então L = 2*N+1
        
        M (int, optional): ordem do esquema de modulação M-PSK. Defaults to 4.

    Returns:
        np.array: Coeficientes do filtro de máxima verossimilhança a ser usado em Viterbi & Viterbi.
    
    Referências:
        [1] Digital Coherent Optical Systems, Architecture and Algorithms
    """
    
    try:
        nModes = sigRx.shape[1]
    except IndexError:
        sigRx = sigRx.reshape(len(sigRx), 1)

    Ts = 1/Rs
    
    Es = np.mean(np.abs(sigRx) ** 2)
    L  = 2 * N + 1
    
    # Parâmetros para matriz de covariância:
    SNRLin       = 10**(OSNRdB/10) * (2 * 12.5e9) / (nModes*Rs)
    σ_deltaTheta = 2 * np.pi * delta_lw * Ts
    σ_eta        = Es / (2*SNRLin)
    
    K = np.zeros((L, L))
    B = np.zeros((N + 1, N + 1))
    
    # obtem a matriz K 
    for indlin in range(N + 1):
        for indcol in range(N + 1):
            B[indlin][indcol] = np.min([indlin, indcol])
    
    K[:N + 1, :N + 1] = np.rot90(B, 2)
    K[N:L, N:L] = B
    
    I = np.eye(L)
    
    # obtém a matriz de covariância
    C = Es**M * M**2 * σ_deltaTheta * K + Es**(M - 1) * M**2 * σ_eta * I
    
    # obtem os coeficientes do filtro de máxima verossimilhança
    wML = np.linalg.inv(C) @ np.ones(L)
    
    return wML/np.max(wML)

def movingAverage(x, N=45, alpha=0.03, H=None, window='constant'):
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

    H : np.array
        Matriz de coeficientes do filtro de máxima verossimilhança.
    
    window : str, optional
        Define a janela da média móvel [constant, laplacian, viterbi], by default 'constant'

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

    elif window == 'viterbi':
        h = H

    else:
        raise ValueError('Janela especificada incorretamente.')

    if len(h) > x.shape[0]:
        raise ValueError('A janela deve ser menor ou igual ao comprimento do sinal de entrada.')
    
    y = np.zeros(x.shape, dtype=x.dtype)

    for index in range(nModes):
        
        # calcula a média móvel ao longo de cada coluna 
        average = np.convolve(x[:, index], h, mode='same')
        
        # obtém a saída de mesmo comprimento da entrada
        y[:, index] = average
        
    return y