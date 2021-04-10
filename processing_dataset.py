import pandas as pd
import numpy as np
import torch.nn as nn
import torch
# import torchvision

''' concat_all_lines_in_column()
    Esta funcao concatena todos as linhas de uma coluna específica que possua textos.
    O objetivo e abstrair um vetor com todas as palavras utilizadas nos textos.
'''
def concat_all_lines_in_column(pd_dataset, column_name):
    result_concat = []

    for line in range(18): #pd_dataset[column_name].size
        result_concat = np.concatenate(
        (result_concat, str(pd_dataset[column_name][line]).lower().split()),
            axis=0
        )

    return result_concat

''' remove_and_insert()
    Esta funcao remove textos inseridos em uma coluna da qual nao deve ter e insere o
    valor solicitado no lugar do texto.
'''
def remove_and_insert(pd_dataset, column_name, value):
    result_remove = []

    for line in range(18): #pd_dataset[column_name].size
        element = str(pd_dataset[column_name][line])

        if element.isnumeric():
            result_remove = np.concatenate(
            (result_remove, element.split()),
                axis=0
            )
        else:
            result_remove = np.concatenate(
            (result_remove, [value]),
                axis=0
            )

    return result_remove

# carregando dataset
table = pd.read_csv('chennai_reviews.csv', sep=',')

# criando dicionario (sem repeticoes) da coluna Review_Text
dictionary = np.unique(concat_all_lines_in_column(table, 'Review_Text'))

# guardando o tamanho do dicionario
n_words = len(dictionary)

# guardando a coluna de predicoes 'Sentiment' inserindo '2' nos dados invalidos
predict_column = remove_and_insert(table, 'Sentiment', '2')

n_categories = len(np.unique(predict_column))

''' word_to_tensor(), text_to_tensor()
    Estas funcoes irao transformar as palavras de um dicionario qualquer em vetores binarios.
    O formato da binarizacao sera um vetor do tamanho do dicionario com seus valores iguais
    a 0, exceto o indice do qual representara a palavra, pois este valor sera 1.
    Ex:
        dicionario = ['Estou', 'em', 'casa.']
    Sendo assim:
        'Estou' = [1, 0, 0]
        'em' = [0, 1, 0]
        'casa.' = [0, 0, 1]
    Logo:
        'Estou em casa.' = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
'''
''' funcao que monta o vetor binario dado uma palavra '''
def word_to_tensor(word):
    tensor = np.zeros((1, n_words))

    for i in range(n_words):
        if dictionary[i] == word:
            tensor[0][i] = 1

    return tensor

''' funcao que monta uma matriz binaria dado um texto '''
def text_to_tensor(text):
    token_text = str(text).split()
    tensor = np.zeros((len(token_text), n_words))
    
    for word in range(len(token_text)):
        for i in range(n_words):
            if token_text[word] == dictionary[i] :
                tensor[word] = word_to_tensor(token_text[word])
            
    return tensor


print(dictionary)
# print(word_to_tensor('its'))
# print(text_to_tensor('its really nice place'))
# '''


# Find letter index from all_letters, e.g. "a" = 0
def wordToIndex(word):
    itemindex, = np.where(dictionary==word)
    # print(itemindex[0])
    if(itemindex.size > 0):
        return itemindex[0]
    else:
        return 0

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def wordToTensor(word):
    tensor = torch.zeros(1, n_words)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_words)
    for li, word in enumerate(line):
        tensor[li][0][wordToIndex(word)] = 1
    return tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_words, n_hidden, n_categories)

# input = wordToTensor('before')
# hidden = torch.zeros(1, n_hidden)

# output, next_hidden = rnn(input, hidden)

# print(output)

input = lineToTensor('its nice')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)

print(output)
