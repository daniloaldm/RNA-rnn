# RNA-rnn

## Instalação e Execução
Clone o repositório:
```
git clone git@github.com:daniloaldm/RNA-rnn.git
```
Vá para o diretório:
```
cd RNA-rnn
```
Execute:
```
python3 -m venv .
. bin/activate
pip3 install -r requirements.txt
```
Para executar os arquivos no jupiter:
```
jupyter notebook
```

## Resultados
Foi ultilizado a base de dados do arquivo chennai_reviews.csv, uma taxa de treinamento de 0.005 e 1000 iterações.

Abaixo podemos ver alguns resultados obtidos durante a execução do treino em alguns exemplos. O formato dos resultados são: **texto / predição ✓** ou **texto / predição ✗ (predição correta)**.

![Figure_3](https://user-images.githubusercontent.com/51512175/114290613-45464f00-9a57-11eb-8a5f-55e48aaf5660.png)

Abaixo temos um gráfico que representa a perda histórica e que mostra o aprendizado da rede.

![Figure_1](https://user-images.githubusercontent.com/51512175/114290668-a0784180-9a57-11eb-82ba-f12a6af68eac.png)