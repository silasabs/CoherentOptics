import numpy as np
import scipy as sp

def InsertSkew(In, SpS, Rs, ParamSkew):
    """ 
    Esta função insere um desalinhamento temporal (inclinação) 
    entre os componentes em fase e quadratura do sinal 'In'. O sinal 'In' 
    deve ser o sinal obtido na saída do front-end óptico, logo antes do 'ADC'.
    Nesta função, um atraso temporal é especificado para cada componente em 'ParamSkew'. 
    O desalinhamento temporal entre os componentes é então aplicado assumindo o 
    atraso temporal mínimo como referência.

    Args:
        In (np.array): Sinal na saída do front-end óptico.
        SpS (int): Amostras por símbolo.
        Rs (int): Taxa de símbolos [símbolo/s]
        
        ParamSkew (struct): Especifica o atraso temporal (em segundos) para cada componente do sinal de entrada.

            - ParamSkew.TauIV: Atraso temporal para componente em fase.
            - ParamSkew.TauQV: Atraso temporal para componente em quadratura.
            
    Returns:
        np.array: Sinal produzido após inserção distorcida.
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