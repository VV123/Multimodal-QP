from data import loaddata
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import SGD, Adam
from torch.nn import MSELoss, L1Loss
from torch.nn.init import xavier_uniform_
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import BertTokenizer, BertModel
import numpy as np
import sys

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MODEL(nn.Module):
    def __init__(self, latent_size):
        super(MODEL, self).__init__()
       
        self.vocab_size = 30522
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.sos_idx = 101
        self.eos_idx = 102
        self.pad_idx = 0
        self.unk_idx = 100

        self.max_sequence_length = 50

        # embedding
        self.embedding_size = 768 
        self.d_model = self.embedding_size 

        bertmodel = BertModel.from_pretrained('bert-base-uncased')
        bertmodel.resize_token_embeddings(self.vocab_size)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embedding = bertmodel.embeddings

        # Neighborhood Embedding
        self.latent_size = latent_size
        self.emb_tokens = nn.Embedding(5, self.d_model)
        self.emb_b = nn.Embedding(5, self.latent_size) #borough 
        self.emb_n = nn.Embedding(220, self.latent_size) #neighborhood

        #Amenity Embedding
        self.emb_a = nn.Embedding(100, self.latent_size, padding_idx=0)

        self.fcN1 = nn.Linear(8, self.d_model)
        self.fcN2 = nn.Linear(self.d_model, self.d_model * 40)
        
        self.fc1 = nn.Linear(self.d_model, latent_size) 
        self.fc2 = nn.Linear(latent_size, 1)
        
        self.fcT = nn.Linear(self.embedding_size, self.latent_size)
        
        self.fcE1 = nn.Linear(latent_size, self.d_model)

        self.fcL1 = nn.Linear(latent_size, self.d_model)
        self.fcL2 = nn.Linear(self.d_model, self.d_model * 30)

        self.act = nn.Sigmoid()

        d_model = self.embedding_size
        dropout = 0.5
        nhead = 16
        d_hid = 128
        nlayers = 1
        ntoken = self.vocab_size
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        encoder_layers1 = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder1 = TransformerEncoder(encoder_layers1, nlayers)
        self.encoder = bertmodel.embeddings
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, self.vocab_size)

        self.LN = nn.LayerNorm(self.d_model) 
        self.fcHln = nn.Linear(self.d_model * 3, self.d_model) 
    def forward(self, x, xb, xn, x_input, x_len, xa, src_mask):

        length = x_len
        input_sequence = x_input
        b, s = input_sequence.size()
    
        input_length = x_input.size(1) 
        src = self.encoder(input_sequence) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
       
        output = self.transformer_encoder(src, src_mask)
        mid_hidden = output.view(b, s, self.d_model)
        mid_hidden = torch.mean(mid_hidden, 1).squeeze(1)
        output = self.decoder(output)
               
        b,s,_ = output.size()
        output  = nn.functional.log_softmax(output, dim=-1)
        output = output.view(b, s, self.vocab_size)
        logp = output

        hE = self.emb_a(xa) #[B, amenity_len, latent_size]
        hE = self.fcE1(hE)

        hN = self.fcN1(x)
        hN = self.fcN2(hN)

        hL = self.fcL1(self.emb_b(xb) + self.emb_n(xn))
        hL = self.fcL2(hL)

        hT = mid_hidden

        h6 = torch.cat((self.emb_tokens(torch.tensor([0]*b).view(-1, 1).cuda()), hL.view(-1, 30, self.d_model), self.emb_tokens(torch.tensor([1]*b).view(-1, 1).cuda()), hN.view(-1, 40, self.d_model), self.emb_tokens(torch.tensor([2]*b).view(-1, 1).cuda()), hE.view(-1, 30, self.d_model)), 1)
        h6 = self.transformer_encoder1(h6, None)
        h6 = h6.view(b, 103, self.d_model)
        hNL = torch.concat((h6[:, 0].unsqueeze(1), h6[:, 31].unsqueeze(1), h6[:, 72].unsqueeze(1)), axis = 1)
        hNL = self.LN(hNL).view(-1, self.d_model * 3)
        hNL = self.fcHln(hNL) 

        h1 = F.relu(self.fc1(hT + hNL))
        h2 = self.fc2(h1)
        return self.act(h2), logp, hT, hNL

class MILNCELoss(torch.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, video_embd, text_embd):
        x = torch.matmul(video_embd, text_embd.t())
        nominator = x * torch.eye(x.shape[0])[:,:].cuda()    
        nominator = nominator.sum(dim=1) #--> [B]

        #cat([B, B] [B, B])--> [B, 2B] 
        denominator = torch.cat((x, x.permute(1,0)), dim=1) #--> [B, 2B]
        denominator = torch.logsumexp(denominator, dim=1) #--> [B]
        return torch.mean(denominator - nominator)

