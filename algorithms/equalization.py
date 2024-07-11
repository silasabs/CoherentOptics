import numpy as np
from tqdm.notebook import tqdm

def overlap_save(x, h, NFFT):
    """

    Implementa a convolução usando o método FFT de sobreposição e salvamento
    (overlap-and-save).

    Args:
        x (np.array): sinal de entrada.
        h (np.array): coeficientes do filtro.
        NFFT (int): o tamanho da FFT deve ser maior que a ordem do filtro.
                    De preferência utilize valores em potência de 2 para 
                    se aproveitar da transformada rápida de Fourier.

    Returns:
        y_out (np.array): sinal de saída filtrado.

    Raises:
        ValueError: caso o tamanho da NFFT seja menor que o comprimento do filtro.
        
    Referências:
        [1] Processamento Digital de Sinais Paulo S. R. Diniz Eduardo A. B. da Silva Sergio L. Netto
        Projeto e Análise de Sistemas. 2º Ed.
    """

    K = len(h)
    L = len(x)

    if NFFT < K:
        raise ValueError('NFFT deve ser maior ou igual ao comprimento do filtro')
    
    # determina o atraso da filtragem FIR
    delay = (K - 1) // 2
    
    overlap = K - 1

    # obtem a quantidade de blocos de entrada
    B = np.ceil((L + overlap) / (NFFT - K + 1)).astype(int)

    # realiza o zero pad para K-1 amostras no inicio do sinal 
    # e compensa o comprimento do último bloco
    x = np.pad(x, (overlap, NFFT), mode='constant', constant_values=0+0j)

    # preenche h com zeros até o comprimento NFFT
    h = np.pad(h, (0, NFFT - K), mode='constant', constant_values=0+0j)

    # buffer para os blocos de entrada
    N = np.zeros((NFFT + delay,), dtype='complex')

    # sinal de saída filtrado
    y_out = np.zeros((B * (len(N) - overlap)), dtype='complex')

    for m in range(B):
        
        step = m * (NFFT - overlap)
        
        # janela deslizante que extrai os blocos de entrada de comprimento NFFT.
        N = x[step:step+NFFT]

        H = np.fft.fft(h)
        X = np.fft.fft(N)

        # obtém a convolução circular de cada bloco com a responta ao impulso do filtro.
        y = np.fft.ifft(X * H)

        # obtém as amostras válidas descartando a sobreposição K-1.
        y_out[step:step+(NFFT-overlap)] = y[overlap:]

    return y_out[delay:delay+L]

def lms(u, d, taps, mu):
    """ Least mean squares (LMS)
    
    Simples implementação do algoritmo LMS para filtragem adaptativa.

    Args:
        u (np.array): sinal de entrada unidimensional 
        d (np.array): sinal de referência unidimensional
        taps (int)  : número de coeficientes do filtro   
        mu (float)  : tamanho do passo para o LMS

    Returns:
        tuple: 
            - np.array: sinal de saída.
            - np.array: sinal de erro.
            - np.array: erro quadrático.
            - np.array: coeficintes do filtro após a convergência.
    
    Referências:
        [1] Adaptive Filtering: Algorithms and Practical Implementation
    """

    # número de iterações para filtragem adaptativa
    N = len(u) - taps + 1
    
    # deriva um filtro real ou complexo
    dtype = u.dtype
    
    # obtém o atraso da filtragem FIR
    delay = (taps-1) // 2

    y = np.zeros(len(u), dtype=dtype)     # saída do filtro
    e = np.zeros(len(u), dtype=dtype)     # sinal de erro
    w = np.zeros(taps, dtype=dtype)       # coeficientes iniciais do filtro.

    err_square = np.zeros(len(u), dtype=dtype)   # erro quadrático
    
    # Execulta a filtragem adaptativa
    for n in range(N):
        
        # janela deslizante correspondente a ordem do filtro
        x = np.flipud(u[n:n+taps])

        # calcula a saída no instante n
        y[n] = np.dot(x, w)
        
        # calcula o erro
        e[n] = d[n + delay] - y[n]
        
        # calcula os novos coeficientes do filtro
        w += mu * np.conj(x) * e[n]

        # calcula o erro quadrático
        err_square[n] = e[n]**2

    return y, e, err_square, w

def cma(u, constSymb, taps, mu):
    """ Constant-Modulus Algorithm

    Implementação do equalizador de módulo constante.

    Args:
        u (np.array)        : sinal de entrada unidimensional
        constSymb (np.array): símbolos da constelação
        taps (int)          : número de coeficientes do filtro   
        mu (float)          : tamanho do passo para o CMA

    Returns:
        tuple: 
            - np.array: sinal de saída.
            - np.array: sinal de erro.
            - np.array: coeficintes do filtro após a convergência.
    
    Referências:
        [1] Adaptive Filtering: Algorithms and Practical Implementation
    """

    # Constante relacionada às características da modulação.
    R = np.mean(np.abs(constSymb)**4) / np.mean(np.abs(constSymb)**2)
    
    # Número de iterações para filtragem adaptativa
    N = len(u) - taps + 1
    
    # Obtém o atraso da filtragem FIR
    delay = (taps - 1) // 2

    y = np.zeros(len(u), dtype='complex')  # saída do filtro
    e = np.zeros(len(u), dtype='complex')  # sinal de erro

    # Inicialização dos coeficientes do filtro
    w = np.zeros(taps, dtype='complex')
    w[delay] = 1  

    # Executa a filtragem adaptativa
    for n in range(N):
        
        # janela deslizante correspondente à ordem do filtro
        x = np.flipud(u[n:n + taps])

        # calcula a saída no instante n
        y[n] = np.dot(w, x)
        
        # calcula o erro
        e[n] = y[n] * (np.abs(y[n])**2 - R)
        
        # calcula os novos coeficientes do filtro
        w -= 2 * mu * e[n] * np.conj(x)
    
    return y, e, w

def mimoAdaptEq(x, constSymb, paramEq):
    """

    Equalizador adaptativo MIMO 2x2

    Args:
        x (np.array)         : sinal de entrada com duas polarizações.
        constSymb (np.array) : símbolos da constelação normalizados.
        
        paramEq (struct):

            - paramEq.taps (int): número de coeficientes dos filtros.
            
            - paramEq.lr (float): tamanho do passo para a convergência do algoritmo ['cma', 'rde'].
            
            - paramEq.alg (str): algoritmo de equalização adaptativa a ser usado: ['cma', 'rde', 'cma-to-rde'].

            - paramEq.progBar (bool): visualização da barra de progresso.
            
            - paramEq.N1 (int): número de cálculos de coeficientes a serem realizados antes 
                                da inicialização adequada dos filtros w2H e w2V.
            
            - paramEq.N2 (int): número de cálculos de coeficientes a serem realizados antes de mudar
                                de CMA para RDE

    Raises:
        ValueError: caso o sinal não possua duas polarizações.
        ValueError: caso o algoritmo seja especificado incorretamente.

    Returns:
        tuple: 
            - y (np.array): estimativa dos símbolos.
            - e (np.array): erro associado a cada modo de polarização.
            - w (np.array): matriz de coeficientes.
    """

    if x.shape[1] != 2:
        raise ValueError("O sinal deve conter duas polarizações")
    
    nModes = x.shape[1]

    # constante relacionada às características da modulação
    R = np.mean(np.abs(constSymb)**4) / np.mean(np.abs(constSymb)**2)

    if paramEq.alg == 'cma':
        y, e, w = cmaUp(x, R, nModes, paramEq)
    elif paramEq.alg == 'rde':
        y, e, w = rdeUp(x, R, nModes, paramEq)
    elif paramEq.alg == 'cma-to-rde':
        y, e, w = cmaUp(x, R, nModes, paramEq, preConv=True)
    else:
        raise ValueError("Algoritmo de equalização especificado incorretamente. \
                        as opções válidas são: 'cma', 'rde', 'cma-to-rde'.")
    
    return y, e, w

def rdeUp(x, R, nModes, paramEq, y=None, e=None, w=None, preConv=False):
    """ Radius-Directed Equalization Algorithm
    
    Implementação do algoritmo de equalização direcionada 
    ao raio para multiplexação de polarização 2x2.
    
    Args:
        x (np.array): sinal de entrada.
        R (int): constante relacionada às características da modulação.
        nModes (int): número de polarizações.
        paramEq (struct): parâmetros do equalizador.
        y (np.array, optional): sinal de saída da pre-convergência. Defaults to None.
        e (np.array, optional): sinal de erro da pre-convergência. Defaults to None.
        w (np.array, optional): matriz de coeficientes de pre-convergência. Defaults to None.
        preConv (bool, optional): sinaliza um equalizador de pre-convergência. Defaults to False.

    Returns:
        tuple:
            - y (np.array): estimativa dos símbolos.
            - e (np.array): erro associado a cada modo de polarização.
            - w (np.array): matriz de coeficientes.
    
    Referências:
        [1] Digital Coherent Optical Systems, Architecture and Algorithms
    """
    
    N = len(x) - paramEq.taps + 1

    # Obtém o atraso da filtragem FIR
    delay = (paramEq.taps - 1) // 2
    
    if preConv == False:
        
        paramEq.N2 = 0

        y = np.zeros((len(x), nModes), dtype='complex')
        e = np.zeros((len(x), nModes), dtype='complex') 
        w = np.zeros((paramEq.taps, nModes**2), dtype='complex')
        
        # single spike initialization
        w[:,0][delay] = 1
    
    for n in tqdm(range(paramEq.N2, N), disable=not (paramEq.progBar)):

        xH = np.flipud(x[:,0][n:n+paramEq.taps])
        xV = np.flipud(x[:,1][n:n+paramEq.taps])

        # calcula a saída do equalizador 2x2
        y[:,0][n] = np.dot(w[:,0], xV) + np.dot(w[:,1], xH)
        y[:,1][n] = np.dot(w[:,2], xV) + np.dot(w[:,3], xH)

        R1 = np.min(np.abs(R - np.abs(y[:,0][n])))
        R2 = np.min(np.abs(R - np.abs(y[:,1][n])))

        # calcula e atualiza erro para cada modo de polarização
        e[:,0][n] = y[:,0][n] * (R1**2 - np.abs(y[:,0][n])**2)
        e[:,1][n] = y[:,1][n] * (R2**2 - np.abs(y[:,1][n])**2)

        # atualiza os coeficientes do filtro
        w[:,0] += paramEq.lr[1] * np.conj(xV) * e[:,0][n]
        w[:,2] += paramEq.lr[1] * np.conj(xV) * e[:,1][n]
        w[:,1] += paramEq.lr[1] * np.conj(xH) * e[:,0][n]
        w[:,3] += paramEq.lr[1] * np.conj(xH) * e[:,1][n]

        if n == paramEq.N1:
            # Defina a polarização Y como ortogonal a X para evitar 
            # a convergência para a mesma polarização (evitar a singularidade CMA)
            w[:,3] =  np.conj(w[:,0][::-1])
            w[:,2] = -np.conj(w[:,1][::-1])
        
    return y, e, w

def cmaUp(x, R, nModes, paramEq, preConv=False):
    """ Constant-Modulus Algorithm

    Implementação do algoritmo de módulo constante para 
    multiplexação de polarização 2x2.

    Args:
        x (np.array): sinal de entrada.
        R (int)     : constante relacionada às características da modulação.
        nModes (int): número de polarizações.
        paramEq (struct): parâmetros do equalizador.
        preConv (bool, optional): sinaliza um equalizador de pre-convergência. Defaults to False.

    Returns:
        tuple:
            - y (np.array): estimativa dos símbolos.
            - e (np.array): erro associado a cada modo de polarização.
            - w (np.array): matriz de coeficientes.
    
    Referências:
        [1] Digital Coherent Optical Systems, Architecture and Algorithms
    """

    N = len(x) - paramEq.taps + 1

    # Obtém o atraso da filtragem FIR
    delay = (paramEq.taps - 1) // 2

    y = np.zeros((len(x), nModes), dtype='complex')
    e = np.zeros((len(x), nModes), dtype='complex') 
    w = np.zeros((paramEq.taps, nModes**2), dtype='complex')
    
    # single spike initialization
    w[:,0][delay] = 1
    
    for n in tqdm(range(N), disable=not (paramEq.progBar)):

        xH = np.flipud(x[:,0][n:n+paramEq.taps])
        xV = np.flipud(x[:,1][n:n+paramEq.taps])

        # calcula a saída do equalizador 2x2
        y[:,0][n] = np.dot(w[:,0], xV) + np.dot(w[:,1], xH)
        y[:,1][n] = np.dot(w[:,2], xV) + np.dot(w[:,3], xH)

        # calcula e atualiza erro para cada modo de polarização
        e[:,0][n] = y[:,0][n] * (R - np.abs(y[:,0][n])**2)
        e[:,1][n] = y[:,1][n] * (R - np.abs(y[:,1][n])**2)

        # atualiza os coeficientes do filtro
        w[:,0] += paramEq.lr[0] * np.conj(xV) * e[:,0][n]
        w[:,2] += paramEq.lr[0] * np.conj(xV) * e[:,1][n]
        w[:,1] += paramEq.lr[0] * np.conj(xH) * e[:,0][n]
        w[:,3] += paramEq.lr[0] * np.conj(xH) * e[:,1][n]

        if n == paramEq.N1:
            # Defina a polarização Y como ortogonal a X para evitar 
            # a convergência para a mesma polarização (evitar a singularidade CMA)
            w[:,3] =  np.conj(w[:,0][::-1])
            w[:,2] = -np.conj(w[:,1][::-1])
        
        if preConv and n == paramEq.N2:
            break
    
    if preConv:
        w = w / np.max(np.abs(w))
        y, e, w = rdeUp(x, R, nModes, paramEq, y, e, w, preConv=True)

    return y, e, w