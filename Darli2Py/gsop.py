import numpy as np

def gsop(rLn):
    """ Esta função realiza a ortogonalização de Gram-Schmidt no sinal 'rLn'

    Args:
        rLn (np.array): Sinal de entrada no qual será realizada a ortogonalização.
    Returns:
        sigRx (np.array): Sinal após ortogonalização de Gram-Schmidt.
    """
    # A 1ª e 2ª colunas devem conter as componentes em fase e quadratura do sinal, respectivamente.
    Rin = np.array([rLn.real, rLn.imag]).T
    
    # Tomando como referência a componente em quadratura:
    rQOrt = Rin[:,1]/np.sqrt(np.mean(Rin[:,1]**2))
    # Orthogonalization:    
    rIInt = Rin[:,0]-np.mean(Rin[:,1]*Rin[:,0])*Rin[:,1]/np.mean(Rin[:,1]**2)
    rIOrt = rIInt/np.sqrt(np.mean(rIInt**2))

    sigRx = rIOrt + 1j*rQOrt

    return sigRx