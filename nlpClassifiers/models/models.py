import torch
import torch.nn as nn
from transformers import AdamW, BertModel
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.init import kaiming_uniform_
from torch.optim import SGD
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.optim import Adam


class LSTMNet(nn.Module):
    def __init__(self, device, num_classes, num_features, criterion, embedding_dim, bidirectional, embedding_weights=None):
        super(LSTMNet, self).__init__()
        self.criterion = criterion
        self.embedding_dim = embedding_dim
        self.lstm_units = 300
        self.lstm_act = nn.Tanh()
        self.num_directions = 2 if bidirectional else 1
        self.device = device
        self.n_layers = 1
        self.embedding = nn.Embedding(num_features, embedding_dim)
        if(embedding_weights is not None):
            print("Embedding layer Weights won't be updated.")
            self.embedding.from_pretrained(embedding_weights, freeze=True)
            #self.embedding.weight.requires_grad = False
        else:
            self.embedding.weight.requires_grad = True
            self.embedding.weight.data.uniform_(-1, 1)
        
       
        self.lstm = nn.LSTM(embedding_dim, self.lstm_units, num_layers=self.n_layers, bidirectional=bidirectional, batch_first=True)
        self.num_directions = 2 if bidirectional else 1

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        self.dropout = nn.Dropout(0.2)       
        
        self.linear = nn.Linear(self.num_directions * self.lstm_units, num_classes)
        self.init_weights()
    def forward(self, x, y):

        # lstm step => then ONLY take the sequence's final timetep to pass into the linear/dense layer
        # Note: lstm_out contains outputs for every step of the sequence we are looping over (for BPTT)
        # but we just need the output of the last step of the sequence, aka lstm_out[-1]
        #print("x input:", x.size())
        x = self.embedding(x)
        #print("embeddings:", x.size())
        #lstm_out, (ht, ct) = self.lstm(x)
        x_, (h_n, c_n) = self.lstm(x)
        if self.num_directions == 2:
            h = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h = h_n[-1, :, :]

        #print("h before activation:", h.size())
        h = self.lstm_act(h)
        #print("h after activation:", h.size())
        output = self.dropout(h)
        output = self.linear(output)
        #print("output after linear:", output.size())
        loss = self.criterion(output.squeeze(), y)
        #print("loss:", loss.size())
        return loss,output
   
    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        nn.init.uniform(self.embedding.weight.data, a=-0.5, b=0.5)
        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

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