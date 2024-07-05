# Introdução

Este repositório utiliza o livro [Digital
Coherent Optical Systems Architecture and Algorithms](https://www.amazon.com.br/Digital-Coherent-Optical-Systems-Architecture/dp/3030665402/ref=sr_1_1?__mk_pt_BR=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=3CIEB4R4W6ZSS&keywords=Digital+Coherent+Optical+Systems+Architecture+and+Algorithms&qid=1707700545&sprefix=digital+coherent+optical+systems+architecture+and+algorithms%2Caps%2C159&sr=8-1&ufe=app_do%3Aamzn1.fos.25548f35-0de7-44b3-b28e-0f56f3f96147) como principal referência para o estudo de sistemas de comunicações ópticas coerentes.

As anotações foram realizadas com o intuito de facilitar a compreensão dos conceitos abordados e fornecer exemplos de forma prática. Você pode criar o ambiente simultaneamente e instalar todos os pacotes necessários para começar a navegar pelo repositório fazendo:

```
# obtenha o repositório de forma local
$ git clone https://github.com/silasabs/CoherentOptics.git

# acesse o diretório do projeto
$ cd CoherentOptics

# crie um ambiente ideal para começar a navegar pelo projeto
$ conda env create -f environment.yml
```

## Sumário do Material

Navegue entre os principais tópicos de comunicações ópticas coerentes que são abordados através de anotações de aulas, simulações e exemplos práticos, disponibilizados em cadernos jupyter.

**Chapter 2:** [The Optical Transmitter](https://github.com/silasabs/CoherentOptics/blob/main/Jupyter/Optical%20Transmitters.ipynb)\
**Chapter 3:** [The Optical Channel](https://github.com/silasabs/CoherentOptics/blob/main/Jupyter/Optical%20Channel.ipynb) \
**Chapter 4:** [The Receiver Front-End, Orthogonalization, and Deskew](https://github.com/silasabs/CoherentOptics/blob/main/Jupyter/Coherent%20Receiver%20Front-End.ipynb) \
**Chapter 5:** [Equalization](https://github.com/silasabs/CoherentOptics/blob/main/Jupyter/Equalization.ipynb) \
**Chapter 6:** Carrier Recovery \
**Chapter 7:** Clock Recovery \
**Chapter 8:** Performance Evaluation

Acesse as implementações de diferentes algoritmos descritos no livro de referência em [/algorithms](https://github.com/silasabs/CoherentOptics/tree/main/algorithms)

<br>
<center>
    <img src="https://i.postimg.cc/Wp7vYy2q/Screenshot-from-2024-04-18-22-07-49.png">
</center>
<br>

## Observações

O material utiliza o [OptiCommPy](https://github.com/edsonportosilva/OptiCommPy) para implementar simulações de modelos físicos e tarefas de processamento digital de sinais.