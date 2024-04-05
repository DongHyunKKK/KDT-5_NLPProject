
from joblib import load
import pandas as pd
import numpy as np
import os
import torch
from konlpy.tag import Mecab
import pickle
import torch.nn as nn

class TextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_class):
        super(TextModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, sparse=False)
        self.rnn = nn.GRU(embedding_dim, hidden_size, 1, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)  # 드롭아웃 레이어 추가
        self.fc = nn.Linear(hidden_size, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        output, hidden = self.rnn(embedded)
        output = self.dropout(output) 
        return self.fc(output)


class TextModel2(nn.Module):
    def __init__(self, VOCAB_SIZE, EMBEDD_DIM, HIDDEN_SIZE, NUM_CLASS):
        super().__init__()
        self.embedding = nn.EmbeddingBag(VOCAB_SIZE, EMBEDD_DIM, sparse = False)
        self.rnn = nn.GRU(EMBEDD_DIM, HIDDEN_SIZE, batch_first = True, bidirectional = True)
        self.fc = nn.Linear(2*HIDDEN_SIZE, NUM_CLASS)
        self._init_weights()
        self.dropout = nn.Dropout()
    
    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)  

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        output, _ = self.rnn(embedded)

        return self.fc(output)


class TextStdDia(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_Class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        output, _hidden = self.rnn(embedded)
        return self.sigmoid(self.fc(output))


class TextMulti(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_Class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_Class)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        output, _hidden = self.rnn(embedded)
        return self.fc(output)

DEVICE = torch.device('cpu')
def load_vocab(file_path):
    with open(file_path, 'rb') as f:  
        vocab = pickle.load(f)
    return vocab


front = '''
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀⠀⠀⠀⠀⠀⢀⣤⣶⣶⣶⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣠⣶⣶⣄⠀⠀⣿⣧⣀⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡿⠟⢻⣿⠀⠀⠀⠀⠀⠈⠉⠁⠸⣿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣶⡀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢀⣿⡏⠙⣿⡆⠀⢿⣿⡟⠛⠃⠀⣴⣶⣶⣄⠀⢸⣿⣦⠀⠀⢀⠀⠀⠈⣗⣀⣼⡯⠀⠀⠀⠀⢀⠀⠀⢀⠀⣿⡿⠀⠀⠀⠀⣠⣶⡄⠀⠀⡀⢼⣇⠀⠀⢀⠀⠀⢀
⢀⠡⠈⡀⢿⣿⣾⡿⠃⠠⢸⣿⣇⠠⠈⢰⣿⡉⢹⣿⡆⢈⣿⣿⣷⣾⡦⠀⠡⠈⢿⡿⠿⠃⠠⠁⠌⠐⡠⣈⡐⢠⠀⡌⣁⠠⠁⠌⠀⣿⣿⠀⠄⡁⢀⢻⣿⡿⣿⠇⠠⠁⠂
⠂⠠⠁⠠⢀⠉⡁⠐⠠⠁⡐⣿⡿⠀⡐⠈⢿⣿⣾⡿⠀⡀⢿⣿⠃⢀⠁⢤⣥⣦⣤⣤⣴⣬⣤⣬⡄⠘⠿⢿⡿⣿⢿⣿⣿⣿⣿⠃⠄⢿⣿⣶⣶⡄⠠⢸⣿⡄⠀⠄⠂⡈⠄
⠈⠐⢈⠀⢂⠠⠀⣿⡿⣿⣆⠈⡁⠐⡀⠌⢀⠉⠁⡀⠐⢀⠈⢁⠐⠠⠈⠈⠛⠿⠿⠿⠿⠿⠿⠿⠃⡐⠠⢀⠀⡀⣤⠀⠠⢀⠠⠀⠂⠄⠉⠉⠉⠁⠄⠘⣿⠇⠀⢂⠁⠠⠐
⠁⡈⠄⠐⠠⢀⠂⠀⠄⣾⡏⢀⡄⠁⠠⠐⢀⠈⠄⠠⠁⠂⢈⠠⠐⢀⠡⠈⠄⠠⠐⣷⣶⡀⠄⡀⠂⠄⡐⠀⢂⣸⣿⡀⠡⢀⠐⠈⠠⠈⠄⠡⠈⡐⠈⠠⢀⠠⠈⡀⠄⠡⠐
⠀⠠⠀⠁⠂⠀⠠⠁⠂⠿⠿⠛⠁⠈⠐⠀⠂⠈⠀⠂⠁⠈⠀⠐⠀⠂⠀⠂⠈⠀⠂⣿⣿⡧⠀⠐⠀⠂⠀⠈⠠⣿⠙⠿⣷⣄⠂⠁⠂⠁⠐⠀⠁⠀⠌⠀⠠⠐⠀⠀⠂⠐⠈
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀     ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                     [사투리 분류 모델기]
'''
while True:
    print(front)
    
    while True:
        choice = input('분류기를 이용하시겠어요? (y/n) ')
        if choice == 'y':
            break
        elif choice == 'n':
            exit()
        else:print('다시 입력해주세요')

    print('이용할 분류기를 선택하세요')
    print(''' 
    1. 경상도 사투리 VS 표준말 
    2. 전라도 사투리 VS 표준말
    3. 제주도 사투리 VS 표준말
    4. 5도 사투리 + 표준말 분류
    '''
    )


    while True:
        model_num = int(input('분류기 번호를 입력하세요. : '))
        if model_num == 1:
            model = torch.load('model/model_tkLang.pt')
            t = '경상도 사투리'
            f = '표준말'
            break
        elif model_num == 2 :
            model = torch.load('model/model_Jeolla.pht')
            t = '전라도 사투리'
            f = '표준말'
            break
        elif model_num == 3 :
            model = torch.load('model/model_jeju_nodelete.pt')
            t = '표준말'
            f = '제주도 사투리'
            break
        elif model_num == 4 :
            model = torch.load('model/model_multi.pt')
            l1 = '강원도 사투리'  # 0 
            l2 = '경상도 사투리'  # 1
            l3 = '전라도 사투리'  # 2
            l4 = '제주도 사투리'  # 3
            l5 = '충청도 사투리'  # 4
            l6 = '표준말'        # 5
            break
        else:print('번호를 제대로 다시 입력해주세요')


    if model_num == 1:
        vocab_file = 'vocab/vocab_tk.pkl'
    elif model_num == 2:
        vocab_file = 'vocab/vocab_Jeolla.pkl'
    elif model_num == 3:
        vocab_file = 'vocab/vocab_jeju.pkl'
    else:
        vocab_file = 'vocab/vocab_multi_new.pkl'


    vocab = load_vocab(vocab_file)


    text = input('사용자의 입력 문장을 입력하세요 : ')


    mecab = Mecab()

    def predict(model, text):
        text_pipeline = lambda x: vocab(x)
        with torch.no_grad():
            text = torch.tensor(text_pipeline(mecab.morphs(text)), dtype=torch.int64).to(DEVICE)       
            offsets = torch.tensor([0]).to(DEVICE)
            pre = model(text, offsets)
            if model_num == 4:
                pre = torch.softmax(pre, dim=1)
            elif model_num == 3:
                pre = pre
            elif model_num == 1:
                # print(pre)

                pre = pre

                # print(pre)

            else:
                pre = torch.sigmoid(pre)
        return (pre, model_num)


    def binary_prediction(prediction, num):
        print('================= 결과 ====================')
        if num != 1:
            if prediction > 0.5:
                print(f'{t} 입니다.')
            else:
                print(f'{f} 입니다.')

        elif num == 1:
            if prediction > 2:
                print(f'{t} 입니다.')
            else:
                print(f'{f} 입니다.')

    def multi_prediction(prediction, num):
        print('================= 결과 ====================')
        # print(prediction)
        # print(prediction.argmax(dim= 1).item() )

        if prediction.argmax(dim= 1).item() == 0:
            print(f'{l1} 입니다.')
        elif prediction.argmax(dim= 1).item() == 1:
            print(f'{l2} 입니다.')
        elif prediction.argmax(dim= 1).item() == 2:
            print(f'{l3} 입니다.')
        elif prediction.argmax(dim= 1).item() == 3:
            print(f'{l4} 입니다.')
        elif prediction.argmax(dim= 1).item() == 4:
            print(f'{l5} 입니다.')
        else:
            print(f'{l6} 입니다')


    if model_num == 4:
        multi_prediction(*predict(model, text))
    elif model_num in [0,1,2,3]:
        binary_prediction(*predict(model, text))





