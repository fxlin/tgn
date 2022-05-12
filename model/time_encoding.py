import torch
import numpy as np
import traceback


# xzl: a learnable model... used by temporal encodings 
#     t is projected through a linear layer, then simply cos()... quantization??

class TimeEncode(torch.nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(1, dimension)

    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                       .float().reshape(dimension, -1))
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    #       xzl: @seq_len: # of recent interactions (from temporal neighborhood).
    #           e.g. [200,1] source node's own time encoding. (200 -- batch size)
    #            [600,10] time encodings for edges with 1-hop neighbors. 600 -- 200x3 (source, dest, neg_dest). 
    #                   10 (seq_len):num of recent interactions (i.e. # of 1-hop neighbors)
    t = t.unsqueeze(dim=2)

    #print("xzl: time encode:t size", t.size())    
    # output has shape [batch_size, seq_len, dimension]
    #     xzl: e.g. 600x10x172
    '''
    if t.size(dim=1) == 10: 
      print(t.cpu())
    
      print("------------------ xzl: time encode:t size", t.size())
      for line in traceback.format_stack():
        print(line.strip())
    '''

    output = torch.cos(self.w(t))   # xzl: element-wise cos

    '''
    if t.size(dim=1) == 10: 
      # print(output.cpu())
      print("xzl: time encode output size", output.size())    
    '''
    
    return output
