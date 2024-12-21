# Introdução

Este repositório utiliza o livro [Digital
Coherent Optical Systems Architecture and Algorithms](https://www.amazon.com.br/Digital-Coherent-Optical-Systems-Architecture/dp/3030665402/ref=sr_1_1?__mk_pt_BR=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=3CIEB4R4W6ZSS&keywords=Digital+Coherent+Optical+Systems+Architecture+and+Algorithms&qid=1707700545&sprefix=digital+coherent+optical+systems+architecture+and+algorithms%2Caps%2C159&sr=8-1&ufe=app_do%3Aamzn1.fos.25548f35-0de7-44b3-b28e-0f56f3f96147) como principal referência para o estudo de sistemas de comunicações ópticas coerentes.

As anotações foram realizadas com o intuito de facilitar a compreensão dos conceitos abordados e fornecer exemplos de forma prática. Você pode criar o ambiente e simultaneamente instalar todos os pacotes necessários para começar:

    # obtenha o repositório 
    $ git clone https://github.com/silasabs/CoherentOptics.git
    
    # acesse o diretório do projeto
    $ cd CoherentOptics
    
    # instale as dependências necessárias
    $ conda install --file requirements.txt    

## Sumário do Material

Navegue entre os principais blocos do DSP que são abordados através de anotações de aulas, simulações e exemplos de implementação, disponibilizados em cadernos jupyter.

**Chapter 2:** [The Optical Transmitter](https://github.com/silasabs/CoherentOptics/blob/main/examples/2.%20Optical%20Transmitters.ipynb)\
**Chapter 3:** [The Optical Channel](https://github.com/silasabs/CoherentOptics/blob/main/examples/3.%20Optical%20Channel.ipynb) \
**Chapter 4:** [The Receiver Front-End, Orthogonalization, and Deskew](https://github.com/silasabs/CoherentOptics/blob/main/examples/4.%20Coherent%20Receiver%20Front-End.ipynb) \
**Chapter 5:** [Equalization](https://github.com/silasabs/CoherentOptics/blob/main/examples/5.%20Equalization.ipynb) \
**Chapter 6:** [Carrier Recovery](https://github.com/silasabs/CoherentOptics/blob/main/examples/6.%20Carrier%20Recovery.ipynb) \
**Chapter 7:** [Clock Recovery](https://github.com/silasabs/CoherentOptics/blob/main/examples/7.%20Clock%20Recovery.ipynb) \
**Chapter 8:** Performance Evaluation

Acesse implementações de diferentes algoritmos descritos no livro de referência em [/algorithms](https://github.com/silasabs/CoherentOptics/tree/main/algorithms)

<br>
<center>
    <img src="https://i.postimg.cc/0NGhT96P/Screenshot-from-2024-12-21-18-37-48.png">
</center>
<br>

## Observações

O material utiliza o [OptiCommPy](https://github.com/edsonportosilva/OptiCommPy) para implementar simulações de modelos físicos e tarefas de processamento digital de sinais.