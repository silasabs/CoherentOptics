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

def gardnerTED(x, PSType):
    """
    Gardner TED Algorithm.

    Parameters
    ----------
    x : np.array
        Matriz de três amostras para o calculo do erro.

    PSType : string
        Tipo de pulso utilizado.

    Returns
    -------
    ek: float
        magnitude do erro de tempo.
    
    Referências
    -----------
        [1] Digital Coherent Optical Systems, Architecture and Algorithms

        [2] F. Gardner, A BPSK/QPSK timing-error detector for sampled receivers. IEEE Trans. Commun. 34(5), 423–429 (1986)
    """
    if PSType == "Nyquist":
        ek = np.abs(x[1]) ** 2 * (np.abs(x[0]) ** 2 - np.abs(x[2]) ** 2)
    else:
        ek = np.real(np.conj(x[1]) * (x[2] - x[0]))
    return ek