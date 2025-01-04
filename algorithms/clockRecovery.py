import logging as logg
import numpy as np

def interpolator(x, mu):
    """
    Interpolador cúbico baseado na estrutura de Farrow

    Parameters
    ----------
    x : np.array
        matriz de 4 elementos para interpolação cúbica.
    
    mu : float
        parâmetro de interpolação.

    Returns
    -------
    y : float
        sinal interpolado.
    
    Referências
    -----------
        [1] Digital Coherent Optical Systems, Architecture and Algorithms
        
        [2] C. Farrow, A continuously variable digital delay element, in IEEE International Symposium on Circuits and Systems, vol. 3 (1988), pp. 2641–2645
    """
    
    return (
        x[0] * (-1/6 * mu**3 + 0 * mu**2 + 1/6 * mu + 0) +
        x[1] * (1/2 * mu**3 + 1/2 * mu**2 - mu + 0) +
        x[2] * (-1/2 * mu**3 - mu**2 + 1/2 * mu + 1) +
        x[3] * (1/6 * mu**3 + 1/2 * mu**2 + 1/3 * mu + 0)
    )

def gardnerTED(x, isNyquist):
    """
    Gardner TED algorithm.

    Parameters
    ----------
    x : np.array
        Matriz de três amostras para o calculo do erro.

    pulseType : bool
        sinaliza um pulso da família de Nyquist.

    Returns
    -------
    error: float
        Magnitude do erro.
    
    Referências
    -----------
        [1] Digital Coherent Optical Systems, Architecture and Algorithms.

        [2] F. Gardner, A BPSK/QPSK timing-error detector for sampled receivers. IEEE Trans. Commun. 34(5), 423–429 (1986)
    """
    if isNyquist:
        return np.abs(x[1]) ** 2 * (np.abs(x[0]) ** 2 - np.abs(x[2]) ** 2)
    else:
        return np.real(np.conj(x[1]) * (x[2] - x[0]))

def godardTED(B):
    """
    Godard TED algorithm

    Parameters
    ----------
    B : np.array
        Bloco FFT de N elementos.

    Returns
    -------
    float
        Estimativa do erro do bloco normalizada.
    
    Raises
    ------
    ValueError
        Se o comprimento da FFT não for uma potência de 2.
    
    Referências
    -----------
        [1] Digital Coherent Optical Systems, Architecture and Algorithms.

        [2] Josten, A., Baeuerle, B., Dornbierer, E., Boesser, J., Hillerkuss, D., & Leuthold, J. (2017). 
            Modified Godard timing recovery for non-integer oversampling receivers. Applied Sciences (Switzerland), 7(7).
    """
    N = len(B)
    if N % 2 != 0:
        raise ValueError("The FFT length must be a power of 2")
    
    ted = np.sum(np.imag(B[:N//2] * np.conj(B[N//2:])))
    
    return ted/N

def clockRecovery(x, paramCR):
    """
    Executa a recuperação de clock no sinal 'x' usando uma estrutura DPLL 
    consistindo em um interpolador, um TED, um filtro de loop e um NCO. 

    Parameters
    ----------
    x : np.array
        sinal de entrada com duas orientações de polarização obtido em 2 Sa/Símbolo.
    
    paramCR : struct

        - paramCR.ki (float): Constante da parte integral do filtro de loop. (ganho integrativo)

        - paramCR.kp (float): Constante da parte proporcional do filtro de loop. (ganho proporcional)

        - paramCR.Nyquist (bool): Sinaliza um pulso de Nyquist.

        - paramCR.nSymbols (int): Número de símbolos transmitidos.

        - paramCR.ppm (int): Desvio esperado da taxa máxima de clock. [ppm]

    Returns
    -------
    tuple
        - y (np.array): Sinal obtido após a recuperação do clock.
        - ted_values (np.array): O sinal oscilante produzido pelo NCO.

    Referências
    -----------
        [1] Digital Coherent Optical Systems, Architecture and Algorithms
        
        [2] C. Farrow, A continuously variable digital delay element, in IEEE International Symposium on Circuits and Systems, vol. 3 (1988), pp. 2641–2645

        [3] F. Gardner, A BPSK/QPSK timing-error detector for sampled receivers. IEEE Trans. Commun. 34(5), 423–429 (1986)
    """

    length, nModes = x.shape
    nSymbols = paramCR.nSymbols

    y = np.zeros((int((1 - paramCR.ppm / 1e6) * length), nModes), dtype="complex")
    nco_values = np.zeros(x.shape, dtype="float") # pre-llocate NCO values
    
    y[:2] = x[:2]
    
    for indMode in range(nModes):
        out_nco = 0           
        loopFilterOut  = 1             
        
        n = 2
        mu = 0                      
        basePoint = n                   
        
        integrative = loopFilterOut 

        while n < length - 1 and basePoint < length - 2:
            y[n, indMode] = interpolator(x[basePoint - 2: basePoint + 2, indMode], mu)

            if n % 2 == 0:
                errorTED = gardnerTED(y[n - 2: n + 1, indMode], paramCR.Nyquist)
                
                # loop PI filter
                integrative += paramCR.ki * errorTED
                proportional = paramCR.kp * errorTED
                loopFilterOut = proportional + integrative

            out_nco -= loopFilterOut
            
            if out_nco > -1 and out_nco < 0:
                basePoint += 1 # Neste caso, a próxima amostra mn+1 é usada como ponto base para a próxima atualização do interpolador
            elif out_nco >= 0:
                basePoint += 2 # Neste caso, uma amostra é ignorada e a outra amostra é usada como ponto base para a próxima atualização do interpolador.

            out_nco = out_nco % 1
            mu = out_nco/loopFilterOut
            
            nco_values[n, indMode] = out_nco

            # atualiza o indexador temporal 'n'   
            n += 1
        
    if nSymbols * 2 < len(y):
        return y[0:nSymbols * 2, :], nco_values
    else:    
        return y, nco_values