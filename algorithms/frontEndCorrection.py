import numpy as np
import scipy as sp
from utils import convmtx

def gsop(rLn):
    """
    Ortogonalização de Gram-Schmidt

    Parameters
    ----------
    rLn : np.array
        Sinal de entrada no qual será realizada a ortogonalização.

    Returns
    -------
    np.array
       sigRx: Sinal após ortogonalização de Gram-Schmidt.
    
    Referências
    -----------
    [1] Digital Coherent Optical Systems, Architecture and Algorithms. 
        
    [2] I. Fatadin, S.J. Savory, D. Ives, Compensation of quadrature imbalance 
    in an optical QPSK coherent receiver. IEEE Photon. Technol. Lett. 20(20), 1733–1735 (2008)
    """

    Rin = np.array([rLn.real, rLn.imag]).T
    
    # Tomando como referência a componente em quadratura
    rQOrt = Rin[:,1] / np.sqrt(np.mean(Rin[:,1]**2))

    # Realiza a ortogonalização    
    rIInt = Rin[:,0] - np.mean(Rin[:,1] * Rin[:,0]) * Rin[:,1] / np.mean(Rin[:,1]**2)
    rIOrt = rIInt / np.sqrt(np.mean(rIInt**2))

    sigRx = rIOrt + 1j*rQOrt

    return sigRx

def Deskew(rIn, SpS, Rs, N, ParamSkew):
    """
    Realiza o enquadramento no sinal 'rIn' usando um interpolador de Lagrange de ordem 'N'. 
    O desalinhamento temporal é compensado levando em consideração o menor atraso temporal. 
    Os atrasos temporais de cada componente (fase e quadratura) são especificados em 'ParamSkew'.

    Parameters
    ----------
    rIn : np.array
        Sinal de entrada após o ADC no qual o enquadramento será realizado.

    SpS : int
        Amostras por símbolo.

    Rs : int
        Taxa de símbolos [símbolo/s]

    N : int
        Ordem do polinômio de interpolação Lagrangeana.

    ParamSkew : struct
        - ParamSkew.TauIV: Atraso temporal para componente em fase.
        - ParamSkew.TauQV: Atraso temporal para componente em quadratura.

    Returns
    -------
    np.array
        sigRx: Sinal após o enquadramento temporal.
    
    Referências
    -----------
    [1] Digital Coherent Optical Systems, Architecture and Algorithms
    """

    rIn = rIn.reshape(-1)
    In  = np.array([rIn.real, rIn.imag]).T

    # Desvio a ser compensado tendo como referência o TADC
    TADC = 1/(SpS*Rs)
    Skew = np.array([ParamSkew.TauIV/TADC, ParamSkew.TauQV/TADC])

    Skew = Skew - np.min(Skew) 
    # Parte inteira e fracionária da inclinação.
    nTADC = np.floor(Skew);  muTADC = -(Skew-nTADC)

    # Obtendo o filtro FIR e realizando a interpolação.
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

def InsertSkew(In, SpS, Rs, ParamSkew):    
    """
    Insera um desalinhamento temporal (inclinação) entre os componentes em fase e quadratura. 

    Parameters
    ----------
    In : np.array
        Sinal na saída do front-end óptico.

    SpS : int
        Amostras por símbolo

    Rs : int
        Taxa de símbolos [símbolo/s]

    ParamSkew : struct
        Especifica o atraso temporal (em segundos) para cada componente do sinal de entrada.
        
        - ParamSkew.TauIV: Atraso temporal para componente em fase.
        - ParamSkew.TauQV: Atraso temporal para componente em quadratura.

    Returns
    -------
    np.array
        sigRx: Sinal produzido após inserção distorcida.
    
    Notes
    -----
    O sinal 'In'  deve ser o sinal obtido na saída do front-end óptico, logo antes do 'ADC'.
    Nesta função, um atraso temporal é especificado para cada componente em 'ParamSkew'. 
    O desalinhamento temporal entre os componentes é então aplicado assumindo o 
    atraso temporal mínimo como referência.

    Referências
    -----------
    [1] Digital Coherent Optical Systems, Architecture and Algorithms
    """

    Rin = np.array([In.real, In.imag]).T
    
    iIV = Rin[:,0]
    iQV = Rin[:,1]
    
    Ts = 1/(SpS*Rs)
    Skew = np.array([ParamSkew.TauIV/Ts, ParamSkew.TauQV/Ts])
    
    # Usando a inclinação mínima como referência
    Skew = Skew - np.min(Skew)
    
    # função de interpolação para quadratura e fase
    interpI = sp.interpolate.CubicSpline(np.arange(0, len(iIV)), iIV, extrapolate=True)
    interpQ = sp.interpolate.CubicSpline(np.arange(0, len(iQV)), iQV, extrapolate=True)

    # novo eixo de interpolação 
    inPhase    = np.arange(Skew[0], len(iIV))
    quadrature = np.arange(Skew[1], len(iQV))

    iIV = interpI(inPhase)
    iQV = interpQ(quadrature)

    MinLength = np.min([len(iIV), len(iQV)])
    sigRx = iIV[:MinLength] + 1j*iQV[:MinLength]

    return sigRx