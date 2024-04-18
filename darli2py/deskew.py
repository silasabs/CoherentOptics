import numpy as np
import scipy as sp

def convmtx(h, N):
    """
    Realiza a implementação da função nativa presente no matlab.
    Determina uma matriz de convolução a partir de um vetor de entrada 'h'

    Args:
        h (np.array): Vetor de entrada, especificado como linha ou coluna.
        N (int): Comprimento do vetor a ser convoluído, especificado como um número inteiro positivo.

    Returns:
        np.array: Retorna a matriz de convolução H
    """
    H = sp.linalg.toeplitz(np.hstack((h, np.zeros(N-1))), np.zeros(N))
    return H.T

def Deskew(rIn, SpS, Rs, N, ParamSkew):
    """
    Esta função realiza o enquadramento no sinal 'rIn' usando um interpolador de Lagrange de ordem 'N'. 
    O interpolador é implementado por um filtro FIR de comprimento 'N+1'. O desalinhamento temporal 
    é compensado levando em consideração o menor atraso temporal. Os atrasos temporais de cada componente
    (fase e quadratura) são especificados em 'ParamSkew'.

    Args:
        rIn (np.array): Sinal de entrada após o ADC no qual o enquadramento será realizado.
        SpS (int): Amostras por símbolo.
        Rs (int): Taxa de símbolos [símbolo/s]
        N (int): Ordem do polinômio de interpolação Lagrangeana.
        
        ParamSkew (struct): Especifica o atraso temporal (em segundos) para cada componente do sinal de entrada.
            
            - ParamSkew.TauIV: Atraso temporal para componente em fase.
            - ParamSkew.TauQV: Atraso temporal para componente em quadratura.

    Returns:
        np.array: Sinal após enquadramento.
    """
    
    rIn = rIn.reshape(-1)
    In  = np.array([rIn.real, rIn.imag]).T

    # Desvio a ser compensado tendo como referência o TADC
    TADC = 1/(SpS*Rs)
    Skew = np.array([ParamSkew.TauIV/TADC, ParamSkew.TauQV/TADC])

    Skew = Skew - np.min(Skew) 
    # Parte inteira e fracionária da inclinação.
    nTADC = np.floor(Skew);  muTADC = -(Skew-nTADC)

    # Obtendo o filtro FIR e interpolando os sinais
    NTaps = N+1
    interp = []

    for i in range(In.shape[1]):
        L = np.zeros((NTaps, 1)); Aux = 0

        # Obtenção do interpolador Lagrangeano
        for n in np.arange(0, N+1) - np.floor(np.mean(np.arange(0, N+1))) + nTADC[i]:
            m = np.arange(0, N+1) - np.floor(np.mean(np.arange(0, N+1))) + nTADC[i]
            m = np.delete(m, np.where(m == n))
            L[Aux,:] = np.prod((muTADC[i]-m)/(n-m))
            Aux += 1
        
        # Interpolando o sinal recebido:
        matrix = np.concatenate((np.zeros(1, np.floor(NTaps/2)), In[:,i].T, np.zeros(1, np.floor(NTaps/2))))
        sAux = np.flipud(convmtx(matrix, NTaps))
        sAux = sAux[:,NTaps-1:-1]

        rOut = (L.T @ sAux).T
        interp.append(rOut)
    
    sigRx = np.array(interp)
    sigRx = sigRx[0,:] + 1j*sigRx[1,:]

    return sigRx