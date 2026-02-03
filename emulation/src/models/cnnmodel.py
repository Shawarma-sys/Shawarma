import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor

class TextCNN1(nn.Module):

    def __init__(self,input_size, num_classes,
                 len_vocab,ipd_vocab,
                 len_embedding_bits,ipd_embedding_bits,
                 nk, ebdin,
                 device):
        super(TextCNN1, self).__init__()
        self.ebdin = ebdin
        self.input_size = input_size
        self.len_embedding_bits = len_embedding_bits
        self.ipd_embedding_bits = ipd_embedding_bits
        ebdbit = len_embedding_bits + ipd_embedding_bits
        self.len_vocab = len_vocab
        self.ipd_vocab = ipd_vocab
        self.num_classes = num_classes
        self.device = device
        self.nk = nk
        self.len_embedding = nn.Embedding(self.len_vocab, self.len_embedding_bits).to(device)
        self.ipd_embedding = nn.Embedding(self.ipd_vocab, self.ipd_embedding_bits).to(device)
        self.bn_logits = nn.BatchNorm1d(num_classes).to(device)
        
        self.fc1 = nn.Linear(ebdbit,ebdin).to(device)
        self.conv3 = nn.Conv2d(1, nk, (3, ebdin)).to(device)
        self.conv4 = nn.Conv2d(1, nk, (4, ebdin)).to(device)
        self.conv5 = nn.Conv2d(1, nk, (5, ebdin)).to(device)
        self.fc2 = nn.Linear(18*nk, num_classes).to(device)


    def forward(self, x):
        len_x = x[:,:,0].to(self.device)
        ipd_x = x[:,:,1].to(self.device)
        len_x = len_x.to(self.device).long()
        ipd_x = ipd_x.to(self.device).long()
        len_x = self.len_embedding(len_x)
        ipd_x = self.ipd_embedding(ipd_x)
        batch = x.shape[0]
        x = torch.cat((len_x, ipd_x), dim=-1)
        x = self.fc1(x)
        x = x .view(x.shape[0],1,x.shape[1],x.shape[2])
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))
        x1 = x1.view(batch,-1)
        x2 = x2.view(batch,-1)
        x3 = x3.view(batch,-1)
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, 1, -1)
        x = self.fc2(x)
        x = x.squeeze(1) 
        x = self.bn_logits(x)
        x = x.view(-1, self.num_classes)
        return x

class TextCNN2(nn.Module):

    def __init__(self,input_size, num_classes,
                 len_vocab,ipd_vocab,
                 len_embedding_bits,ipd_embedding_bits,
                 nk, ebdin,
                 device):
        super(TextCNN2, self).__init__()
        self.ebdin = ebdin
        self.input_size = input_size
        self.len_embedding_bits = len_embedding_bits
        self.ipd_embedding_bits = ipd_embedding_bits
        ebdbit = len_embedding_bits + ipd_embedding_bits
        self.len_vocab = len_vocab
        self.ipd_vocab = ipd_vocab
        self.num_classes = num_classes
        self.device = device
        self.nk = nk
        
        self.len_embedding = nn.Embedding(self.len_vocab, self.len_embedding_bits).to(device)
        self.ipd_embedding = nn.Embedding(self.ipd_vocab, self.ipd_embedding_bits).to(device)
        self.bn_logits = nn.BatchNorm1d(num_classes).to(device)
        
        self.fc1 = nn.Linear(ebdbit, ebdin).to(device)
        
        # 严格链状结构：限制通道数和特征图大小
        # 最终特征图尺寸限制在 nk * 7 * 1 (当 nk=8 时即为 8 * 7 * 1，符合 2 * 4 * 7 * 1 的限制)
        self.features = nn.Sequential(
            nn.Conv2d(1, nk, (3, ebdin)).to(device),
            nn.BatchNorm2d(nk).to(device),
            nn.ReLU(),
            nn.Conv2d(nk, nk, (3, 1), padding=(1, 0)).to(device),
            nn.BatchNorm2d(nk).to(device),
            nn.ReLU(),
            nn.Conv2d(nk, nk, (3, 1), padding=(1, 0)).to(device),
            nn.BatchNorm2d(nk).to(device),
            nn.ReLU()
        )
        
        # 假设 window_size=9:
        # 第一层 (3, ebdin) 后高度为 7
        # 后续层 (3, 1) 带 padding=1 高度维持为 7
        # 特征总数: 7 * nk (当 nk=8 时为 56，低于 TextCNN1 的 18 * nk)
        self.fc2 = nn.Linear(7 * nk, num_classes).to(device)

    def forward(self, x):
        len_x = x[:,:,0].to(self.device).long()
        ipd_x = x[:,:,1].to(self.device).long()
        len_x = self.len_embedding(len_x)
        ipd_x = self.ipd_embedding(ipd_x)
        batch = x.shape[0]
        
        x = torch.cat((len_x, ipd_x), dim=-1)
        x = self.fc1(x)
        x = x.view(batch, 1, x.shape[1], x.shape[2])
        
        x = self.features(x)
        
        x = x.view(batch, -1)
        x = self.fc2(x)
        x = self.bn_logits(x)
        x = x.view(-1, self.num_classes)
        return x
