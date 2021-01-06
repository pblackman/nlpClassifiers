import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_
from transformers import AdamW, BertModel
import numpy as np
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, bert_path, criterion, batch_size, num_layers, hidden_layers, num_labels):
        super(LSTM, self).__init__()
        self.output_size = num_labels
        self.embedding_dim = 3072
        self.criterion = criterion
        self.batch_size = batch_size
        self.n_layers = num_layers
        self.hidden_dim = hidden_layers
        self.bert_model = BertModel.from_pretrained(bert_path, output_hidden_states=True)
        self.lstm = nn.LSTM(self.embedding_dim,
                            self.hidden_dim,
                            self.n_layers,
                            dropout=0.2,
                            batch_first=True)

        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, bert_ids, hidden, bert_mask, Y):
        batch_size = bert_ids.size(0)
        outputs = self.bert_model(input_ids=bert_ids, attention_mask=bert_mask)
        self.bert_model.eval()
        hidden_states = outputs[2][1:]
        outputs = torch.cat(tuple([hidden_states[i] for i in [-1, -2, -3, -4]]), dim=-1)
        bert_mask = bert_mask.unsqueeze(2)
        # Multiply output with mask to only retain non-paddding tokens
        #outputs = torch.mul(outputs, bert_mask)
        # First item ['CLS'] is sentence representation
        #outputs = outputs[:, 0, :]
        #packed_input = pack_padded_sequence(outputs, 30, batch_first=True, enforce_sorted=False)
        lstm_out, hidden = self.lstm(outputs, hidden)
         #lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        #out = out.view(batch_size, -1)
        out = out[:,-1]
        loss = self.criterion(out.squeeze(), Y)
        return loss, out, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

class BertSentenceFeaturesModel(nn.Module):
    def __init__(self, bert_path, criterion, num_labels):
        super(BertSentenceFeaturesModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_path, output_hidden_states=True)
        self.pre_classifier = nn.Linear(3072, 512)
        self.criterion = criterion
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(512, num_labels)

    def forward(self, bert_ids, bert_mask, Y):
        outputs = self.bert_model(input_ids=bert_ids, attention_mask=bert_mask)
        self.bert_model.eval()
        hidden_states = outputs[2][1:]
        outputs = torch.cat(tuple([hidden_states[i] for i in [-1, -2, -3, -4]]), dim=-1)
        bert_mask = bert_mask.unsqueeze(2)
        # Multiply output with mask to only retain non-paddding tokens
        outputs = torch.mul(outputs, bert_mask)
        # First item ['CLS'] is sentence representation
        outputs = outputs[:, 0, :]
        outputs = self.pre_classifier(outputs)
        outputs = self.dropout(nn.ReLU()(outputs))
        outputs = nn.Softmax(dim=0)(self.classifier(outputs))
        loss = self.criterion(outputs, Y)
        return loss, outputs


class BOWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size, criterion):
        super(BOWClassifier, self).__init__()
        self.hidden = nn.Linear(vocab_size, 1000)
        self.act1 = nn.ReLU()
        kaiming_uniform_(self.hidden.weight, nonlinearity='relu')
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(1000, num_labels) 
        kaiming_uniform_(self.output.weight, nonlinearity='sigmoid')
        self.act2 = nn.Softmax()
        self.criterion = criterion
        
    def forward(self, x, y):
        x = self.hidden(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.output(x)
        #x = self.act2(x)
        loss = self.criterion(x, y)
        return loss, x

# loosely based onhttps://github.com/gaussic/text-classification/blob/master/cnn_pytorch.py
class CNNClassifier(nn.Module):
    def __init__(self, num_labels, seq_max_len, vocab_size, criterion, embedding_dim, embedding_weights=None):
        super(CNNClassifier, self).__init__()
        self.criterion = criterion
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if(embedding_weights is not None):
            self.embeddings.from_pretrained(embedding_weights, freeze=True)
        self.conv = nn.Conv1d(embedding_dim, 256, 4)
        self.dropout = nn.Dropout(0.5)  # a dropout layer
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.fc1 = nn.Linear(256, 512)  # a dense layer for classification
        self.act3 = nn.Softmax(dim=1)
        self.fc2 = nn.Linear(512, num_labels)  # a dense layer for classification


    @staticmethod
    def global_max_pool(x):
        """Convolution and global max pooling layer"""
        return x.permute(0, 2, 1).max(1)[0]

    def forward(self, x, y):
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        embedded = self.embeddings(x).permute(0, 2, 1)
        x = self.conv(embedded)
        x = self.global_max_pool(x)
        x = self.act1(x)
        #x = self.dropout(x)
        x = self.fc1(x) 
        x = self.act2(x)
        x = self.fc2(x) 
        #x = self.act3(x)
        loss = self.criterion(x, y)
        return loss, x
