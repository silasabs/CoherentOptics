import numpy as np

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
        [1] Digital Coherent Optical Systems, Architecture and Algorithms

        [2] Adaptive Filtering: Algorithms and Practical Implementation
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