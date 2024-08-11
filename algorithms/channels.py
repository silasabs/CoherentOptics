import numpy as np
from numpy.fft import fftshift, fftfreq, ifft

def linearFiberChannel(sigTxo, paramCh):
    """
    Modelo para um canal de fibra linear considerando as perdas.

    Parameters
    ----------
    sigTxo : np.array
        Recebe o sinal após o modulador óptico como o MZM.

    paramCh : struct
        - paramCh.L: comprimento do enlace. [m]

        - paramCh.alpha: coeficiente de perdas.

        - paramCh.D: parâmetro da dispersão cromática.

        - paramCh.Fa: frequência de amostragem do sinal. [samples/s]

    Returns
    -------
    np.array
        Sinal após a propagação no modelo linear da fibra óptica.
    
    Referências:
        [1] Digital Coherent Optical Systems, Architecture and Algorithms
    """

    L   = paramCh.L                         # comprimento do enlace [m]
    α   = 1e-3 * paramCh.alpha / 4.343      # coeficiente de perdas [1/m]
    D   = paramCh.D                         # parâmetro da dispersão cromática
    Fa  = paramCh.Fa                        # frequência de amostragem [amostras/s]

    λ = 1550e-9     # comprimento de onda
    c = 299792458   # velocidade da luz [m/s](vacuum)
    
    β2 = -(D * λ ** 2) / (2 * np.pi * c) # GVD
    
    NFFT = len(sigTxo)

    # obtém o sinal no dominio da frequência
    sigTxoFFT = fftshift(np.fft.fft(sigTxo)) / NFFT

    freq = fftshift(fftfreq(len(sigTxoFFT), 1/Fa))
    
    # função de trânsferência do canal considerando perdas
    H  = np.exp(1j * 0.5 * β2 * L * (2* np.pi * freq) ** 2) * np.exp(-0.5 * α * L)
    
    # obtém a resposta ao impulso do canal
    sigRxFFT = H * sigTxoFFT
    
    sigRxo = ifft(sigRxFFT) * NFFT
    
    return sigRxo