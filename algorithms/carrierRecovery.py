import numpy as np
from optic.dsp.core import pnorm
from utils import plot4thPower, convmtx
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

def laplaceViterbiCPR(sigRx, alpha=0.03, weight='constant', N=85, M=4):
    """
    Recupera a fase da portadora com o algoritmo Virterbi & Viterbi considerando
    uma janela laplaciana.

    Parameters
    ----------
    sigRx : np.array
        Sinal normalizado em potência, no qual a recuperação de fase será realizada.
    
    alpha : float
        Parâmetro de dispersão da distribuição de Laplace, by default 0.03
    
    weight : str
        Defina os coeficientes da janela do filtro.

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

    phiTime = np.unwrap(np.angle(movingAverage(sigRx**M, N, alpha, window = weight)) / M - np.pi/M, \
                        period=2*np.pi/M, axis=0)
    
    # compensa o ruído de fase
    sigRx = pnorm(sigRx * np.exp(-1j * phiTime))
    
    return sigRx, phiTime

def avgViterbiCPR(sigRx, Rs, OSNRdB, lw, N, M=4):
    """
    Recupera a fase da portadora com o algoritmo Virterbi & Viterbi considerando
    um filtro ótimo.

    Parameters
    ----------
    sigRx : np.array
        Sinal normalizado em potência, no qual a recuperação de fase será realizada.

    Rs : int
        Taxa de símbolos. [símbolos/segundo].

    OSNRdB : float
        OSNR do canal em dB.

    lw : int
        Soma das larguras de linha do laser do oscilador local e transmissor.
    
    N : int
        Número de símbolos passados e futuros na janela. O comprimento
        do filtro é então L = 2*N+1.

    M : int, optional
        Ordem da potência, by default 4

    Returns
    -------
    tuple:
        sigRx (np.array): Constelação com referência de fase.
        phiTime (np.array): Estimativa do ruído de fase em cada modo.
    """
    
    try:
        nModes = sigRx.shape[1]
    except IndexError:
        sigRx = sigRx.reshape(len(sigRx), 1)
    
    Es = np.mean(np.abs(sigRx) ** 2)
    
    # obtem os coeficientes ótimos 
    wML = mlFilterVV(Es, nModes, OSNRdB, lw, Rs, N)

    phiTime = np.unwrap(np.angle(movingAverage(sigRx**M, H=wML, window='viterbi')) / M - np.pi/M, period=2*np.pi/M, axis=0)
    # compensa o ruído de fase
    sigRx = pnorm(sigRx * np.exp(-1j * phiTime))
    
    return sigRx, phiTime

def viterbiCPR(z, lw, Rs, OSNRdB, N, M=4):
    """
    Compensa o ruído de fase com o algoritmo Viterbi & Viterbi.
    
    Parameters
    ----------
    z : np.array
        Sinal normalizado em potência, no qual a recuperação de fase será realizada.
        
    lw : int
        Soma das larguras de linha do laser do oscilador local e transmissor.

    Rs : int
        Taxa de símbolos. [símbolos/segundo].
        
    OSNRdB : float
        OSNR do canal em dB.
        
    N : int
        Número de símbolos passados e futuros na janela. O comprimento
        do filtro é então L = 2*N+1.
        
    M : int, optional
        Ordem da potência, by default 4

    Returns
    -------
    tuple:
        sigRx (np.array): Constelação com referência de fase.
        phiPU (np.array): Estimativa do ruído de fase em cada modo.
    
    Referências
    -----------
        [1] Digital Coherent Optical Systems, Architecture and Algorithms
    """
    
    try:
        nModes = z.shape[1]
    except IndexError:
        z = z.reshape(len(z), 1)
    
    # comprimento do filtro
    L = 2 * N + 1

    Es = np.mean(np.abs(z)**2)

    # obtém os coeficientes do filtro de máxima verossimilhança
    h = mlFilterVV(Es, nModes, OSNRdB, lw, Rs, N)
    
    # estimativa de fase 
    phiTime = np.zeros(z.shape)
    
    for indPhase in range(nModes):
        
        # zero padding
        sigRx = np.pad(z[:, indPhase], (L//2, L//2), mode='constant')
        
        # calcula a matriz de convolução de comprimento L
        sigRx = convmtx(sigRx, L)
        
        # up-down flip
        sigRx = np.flipud(sigRx[:, L-1: -L+1])
        
        # obtém a estimativa de fase em cada modo 
        phiTime[:, indPhase] = np.angle(np.dot(h.T, sigRx**M)) / M - np.pi/M
    
    # phase unwrap
    phiPU = np.unwrap(phiTime, period=2*np.pi/M, axis=0)
    
    # compensa o ruído de fase
    z = pnorm(z * np.exp(-1j * phiPU))
    
    return z, phiPU

def bps(z, constSymb, N, B):
    """
    Compensa o ruído de fase com o algoritmo de busca de fase cega.

    Parameters
    ----------
    z : np.array
        Sinal normalizado em potência, no qual a recuperação de fase será realizada.

    constSymb : np.array
        Símbolos da constelação.

    N : int
        Número de símbolos 'passados' e 'futuros' usados ​​no algoritmo BPS para estimativa 
        de ruído de fase. O número total de símbolos é então L = 2*N+1

    B : int
        Número de rotações de teste.
    
    Returns
    -------
    tuple:
        z (np.array): Constelação com referência de fase.
        phiPU (np.array): Estimativa do ruído de fase em cada modo.
    
    Referências
    -----------
        [1] Digital Coherent Optical Systems, Architecture and Algorithms
    """
    
    nModes = z.shape[1]

    L = 2 * N + 1 # BPS block length
    phiTest = (np.pi / 2) * np.arange(-B/2, B/2) / B # test phases

    phiTime = np.zeros(z.shape)
    
    # define buffers auxiliares.
    bufDist    = np.zeros((B, len(constSymb))) # obtém as distâncias quadráticas
    bufMinDist = np.zeros((B, L))              # obtém as distâncias mínimas de cada bloco

    # performs polarization preprocessing
    for indMode in range(nModes):

        # zero padding
        zBlocks = np.pad(z[:, indMode], (N, N), constant_values=0+0j, mode='constant')

        for indBlk in range(zBlocks.shape[0]):
            for indPhase, phi in enumerate(phiTest):
                
                # calcula a distância euclidiana do símbolo aos símbolos da constelação
                bufDist[indPhase, :] = np.abs((zBlocks[indBlk] * np.exp(-1j * phi) - constSymb) ** 2)

                # obtenha a distância mínima de cada bloco até uma janela [0, L-1]
                bufMinDist[indPhase, indBlk % L] = np.min(bufDist[indPhase, :])

                # realiza a soma das distâncias mínimas sobre a mesma fase de teste, 
                # obtendo a fase que melhor minimiza a soma total das distâncias mínimas.
                phaseMin = np.argmin(np.sum(bufMinDist, axis=1))

                # compensa o zero pad obtendo apenas os valores válidos
                phiTime[indBlk - 2*N, indMode] = phiTest[phaseMin]
    
    # phase unwrap
    phiPU = np.unwrap(phiTime, period=2*np.pi/4, axis=0)
    
    # compensa o ruído de fase.
    z = pnorm(z * np.exp(-1j * phiPU))

    return z, phiPU

def bpsVec(z, constSymb, N, B):
    """
    Compensa o ruído de fase com o algoritmo de busca de fase cega.

    Parameters
    ----------
    z : np.array
        Sinal normalizado em potência, no qual a recuperação de fase será realizada.

    constSymb : np.array
        Símbolos da constelação.

    N : int
        Número de símbolos 'passados' e 'futuros' usados ​​no algoritmo BPS para estimativa 
        de ruído de fase. O número total de símbolos é então L = 2*N+1

    B : int
        Número de rotações de teste.
    
    Returns
    -------
    tuple:
        z (np.array): Constelação com referência de fase.
        phiPU (np.array): Estimativa do ruído de fase em cada modo.
    
    Referências
    -----------
        [1] Digital Coherent Optical Systems, Architecture and Algorithms

        [2] LIU, Q.; JI, W.; LIU, P.; LI, Q.; BAI, C.; XU, H.; ZHU, Y. Blind phase search algorithm based
            on threshold simplification. 2022.
    """

    L = 2 * N + 1
    nModes = z.shape[1]
    
    phiTest = (np.pi / 2) * np.arange(-B/2, B/2) / B # fases de teste
    
    # zero padding 
    lpad = np.zeros((N, nModes))
    zBlocks = np.concatenate((lpad, z, lpad))

    # aplica os ângulos da fase de teste aos símbolos
    zRot = zBlocks[:, :, None] * np.exp(-1j * phiTest)

    # calcule a distância quadrática entre os símbolos da constelação
    distQuad = np.abs(zRot[:, :, :, None] - constSymb) ** 2
    
    # obtenha a métrica de distância mínima entre os símbolos
    minDist  = np.min(distQuad, axis = -1)
    
    # obtem as fases que melhor minimizam a soma total das distâncias mínimas
    cumSum     = np.cumsum(minDist, axis=0)
    sumMinDist = cumSum[L-1:] - np.vstack([np.zeros((1, nModes, B)), cumSum[:-L]])

    indRot = np.argmin(sumMinDist, axis = -1)
    phiPU  = np.unwrap(phiTest[indRot], period=2*np.pi/4, axis=0)

    # compensa o ruído de fase
    z = pnorm(z * np.exp(-1j * phiPU))

    return z, phiPU

def mlFilterVV(Es, nModes, OSNRdB, delta_lw, Rs, N, M=4):
    """
    Calcula os coeficientes do filtro de máxima verossimilhança (ML) para o algoritmo 
    Viterbi & Viterbi, que depende da relação sinal-ruído e da magnitude do ruído de fase.

    Parameters
    ----------
    Es : float
        Energia dos símbolos.
    
    nModes : int
        Número de polarizações.

    OSNRdB : float
        OSNR do canal em dB.

    delta_lw : int
        Soma das larguras de linha do laser do oscilador local e transmissor.

    Rs : int
        Taxa de símbolos. [símbolos/segundo]

    N : int
        Número de símbolos passados e futuros na janela. O comprimento
        do filtro é então L = 2*N+1

    M : int, optional
        Ordem do esquema de modulação M-PSK. Defaults to 4.

    Returns
    -------
    np.array
        Coeficientes do filtro de máxima verossimilhança a ser usado em Viterbi & Viterbi.
    
    Referências
    -----------
        [1] Digital Coherent Optical Systems, Architecture and Algorithms

        [2] E. Ip, J.M. Kahn, Feedforward carrier recovery for coherent optical communications. J.
            Lightwave Technol. 25(9), 2675–2692 (2007).
    """
    
    Ts = 1/Rs          # Período de símbolo
    L  = 2 * N + 1     # Comprimento do filtro
    Bref = 12.5e9      # Banda de referência [Hz]

    # dB para valor linear
    SNR = 10**(OSNRdB / 10) * (2 * Bref) / (nModes*Rs)
    
    # define a variância do ruído multiplicativo
    σ_deltaTheta = 2 * np.pi * delta_lw * Ts
    
    # define a variância do ruído aditivo
    σ_eta = Es / (2 * SNR)
    
    K = np.zeros((L, L))
    B = np.zeros((N + 1, N + 1))
    
    # Determina a matriz B de forma vetorizada evitando loop nested 
    # e overhead de loops explícitos
    index = np.arange(N + 1)
    B = np.minimum.outer(index, index)
    
    K[:N+1,:N+1] = np.rot90(B, 2)
    K[N:L,N:L] = B
    
    I = np.eye(L)
    
    # Obtém a matriz de covariância
    C = Es**M * M**2 * σ_deltaTheta * K + Es**(M - 1) * M**2 * σ_eta * I
    
    # Determina os coeficientes do filtro 
    h = np.linalg.inv(C) @ np.ones(L)
    
    return h/np.max(h)

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

    elif window == 'viterbi' and H is not None:
        h = H

    else:
        raise ValueError('Janela especificada incorretamente.')

    if len(h) > x.shape[0]:
        raise ValueError('A janela deve ser menor ou igual ao comprimento do sinal de entrada.')
    
    y = np.zeros(x.shape, dtype=x.dtype)

    for index in range(nModes):
        
        # calcula a média móvel ao longo de cada coluna 
        average = np.convolve(x[:, index], h, mode='same')
        
        # obtém a média móvel ao longo de cada coluna
        y[:, index] = average
        
    return y