import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import random
import time
import math

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
all_categories = remove_and_insert(table, 'Sentiment', '2')
all_text_review = table['Review_Text'].tolist()

n_categories = len(np.unique(all_categories))

# print(all_text_review)
# exit()

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
    tensor = torch.zeros(1, n_words)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

''' funcao que monta uma matriz binaria dado um texto '''
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_words)
    for li, word in enumerate(line):
        tensor[li][0][word_to_index(word)] = 1
    return tensor

    
''' funcao encontra o index da palvra '''
def word_to_index(word):
    itemindex, = np.where(dictionary==word)
    # print(itemindex[0])
    if(itemindex.size > 0):
        return itemindex[0]
    else:
        return 0

print(dictionary)
# print(word_to_tensor('its'))
# print(text_to_tensor('its really nice place'))
# '''

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

''' Teste com palavra '''
# input = wordToTensor('before')
# hidden = torch.zeros(1, n_hidden)
# output, next_hidden = rnn(input, hidden)
# print(output)

''' Teste com texto '''
input = line_to_tensor('its nice')
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input[0], hidden)
print(output)

''' funcao encontra o index da palvra '''
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

# print(categoryFromOutput(output))

# É preciso ajustar aqui na linha 159
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    n_random = random.randint(0,all_categories.size-1)

    category = all_categories[n_random]
    line = all_text_review[n_random]

    category_tensor = torch.tensor([n_random], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor

# for i in range(10):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     print('category =', category, '/ caregory_tensor =', category_tensor)

# criterion = nn.NLLLoss()

# learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

# def train(category_tensor, line_tensor):
#     hidden = rnn.initHidden()

#     rnn.zero_grad()

#     for i in range(line_tensor.size()[0]):
#         # print(line_tensor)
#         # exit()
#         output, hidden = rnn(line_tensor[i], hidden)

#     print(output.size())
#     print(output)
#     print(category_tensor.size())
#     print(category_tensor)

#     # exit()

#     loss = criterion(output, category_tensor)
#     loss.backward()

#     # Add parameters' gradients to their values, multiplied by learning rate
#     for p in rnn.parameters():
#         p.data.add_(p.grad.data, alpha=-learning_rate)

#     return output, loss.item()

# n_iters = 100000
# print_every = 5000
# plot_every = 1000



# # Keep track of losses for plotting
# current_loss = 0
# all_losses = []

# def timeSince(since):
#     now = time.time()
#     s = now - since
#     m = math.floor(s / 60)
#     s -= m * 60
#     return '%dm %ds' % (m, s)

# start = time.time()

# for iter in range(1, n_iters + 1):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     output, loss = train(category_tensor, line_tensor)
#     current_loss += loss

#     # Print iter number, loss, name and guess
#     if iter % print_every == 0:
#         guess, guess_i = categoryFromOutput(output)
#         correct = '✓' if guess == category else '✗ (%s)' % category
#         print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

#     # Add current loss avg to list of losses
#     if iter % plot_every == 0:
#         all_losses.append(current_loss / plot_every)
#         current_loss = 0