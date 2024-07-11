import numpy as np
import matplotlib.pyplot as plt

def fourthPower(sigRx, Fs, plotSpectrum=False):
    """
    
    Compensa o deslocamento de frequência utilizando o método
    de quarta potência.
    
    Args:
        sigRx (np.array): sinal a ser compensado.
        Fs (int): taxa de amostragem.
        plotSpectrum (bool, optional): retorna o espectro do sinal em quarta potência. Defaults to False.

    Returns:
        tuple: 
            - np.array: sinal compensado.
            - float: estimativa do deslocamento de frequência.
    """
    
    try:
        nModes = sigRx.shape[1]
    except IndexError:
        sigRx = sigRx.reshape(len(sigRx), 1)
    
    NFFT    = sigRx.shape[0]
    lenFreq = Fs * np.fft.fftfreq(NFFT)
    
    time = np.arange(0, sigRx.shape[0]) * 1/Fs

    for indMode in range(nModes):
        
        # Elevar a quarta potência e aplica a FFT
        fourth_power = np.fft.fft(sigRx[:, indMode]**4)

        # Inferir o valor máximo na frequência
        indFO = np.argmax(np.abs(fourth_power))
        
        # Obtenha Delta f
        indFO = lenFreq[indFO]/4       
        
        # Compense o deslocamento de frequência
        sigRx[:, indMode] *= np.exp(-1j * 2 * np.pi * indFO * time)
    
    # Plote o espectro de um dos modos.
    if plotSpectrum:
        plt.plot(lenFreq, 10*np.log10(np.abs(np.fft.fft(sigRx[:, 0]**4))), label=r"$|FFT(s[k]^4)|[dB]$")
        plt.ylabel('Amplitude [dB]')
        plt.xlabel(r'$f$')
        plt.legend()
        plt.grid()
        
    return sigRx, indFO