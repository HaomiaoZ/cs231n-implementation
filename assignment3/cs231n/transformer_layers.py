import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # should be able to vectorize
        for i in range(max_len):
          for j in range(embed_dim):
              if j%2==0:
                pe[:,i,j] = math.sin(i*10000**(-j/embed_dim))
              else:
                pe[:,i,j] = math.cos(i*10000**(-(j-1)/embed_dim))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # one number from output in the matrix is different from expected, so likely should be correct?
        pe = self.get_buffer("pe")
        output = self.dropout(x+pe[:, :S, :D])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # l->S, d->E from notebook to .py, T = S when self atten (Source dim = Target dim)
        # online code does get same error rate as mine implementation (for non masked version)
        
        # from the version generate nan to working version, two things changed:
        # 1. attn_mask==0 rather than using attn_mask directly
        # 2. use temp val to add intermediate steps rather than put everyhing into y calculation 

        H = self.n_head
        
        q = self.query(query) # Q (N, S, E) XQ
        k = self.key(key) # K (N, T, E) XK
        v = self.value(value) # V (N, T, E) XV

        q_reshaped = torch.reshape(q, (N, S, H, E//H)).swapaxes(1,2) # (N, S, E) ->(N, S, H, E/H)->(N, H, S, E/H)
        k_reshaped = torch.reshape(k, (N, T, H, E//H)).swapaxes(1,2) # (N, T, E) ->(N, T, H, E/H)->(N, H, T, E/H)
        v_reshaped = torch.reshape(v, (N, T, H, E//H)).swapaxes(1,2) # (N, T, E) ->(N, T, H, E/H)->(N, H, T, E/H)
        
        qk_norm = torch.matmul(q_reshaped, k_reshaped.swapaxes(2,3))/torch.sqrt(torch.tensor(E/H)) #(N,H,S,E/H)*(N,H,E/H,T)->(N, H, S, T)
        
        if attn_mask is not None:
          qk_norm = torch.masked_fill(qk_norm, attn_mask==0, float('-inf')) # -inf so after softmax it is close to 0, using a really small number works

        temp_val = F.softmax(qk_norm, dim = 3) #softmax along target dimension (along key and value dimension), so along T, which is the last dim (dim 3)

        temp_val = self.attn_drop(temp_val)

        y = torch.matmul(temp_val, v_reshaped) # (N, H, S, T)(softmax along S slice)* (N,H,T,E/H) = (N,H,S,E/H)
        output = self.proj(y.swapaxes(1, 2).reshape((N, S, E))) #(N, H, S, E/H) -> (N, S, H, E/H) -> (N,S,E)
        
        '''
        # code from csdn so that the grad are not nan

        # 投影QKV
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # 获取投影个数
        H = self.n_head
        # 获取投影维度
        D = self.head_dim
        # 矩阵分割 与 转置 这里需要结合QKV的形状来理解
        Q = Q.reshape(N, S, H, D).transpose(1, 2)  # (N H S D)
        K = K.reshape(N, T, H, D).transpose(1, 2)  # (N H T D)
        V = V.reshape(N, T, H, D).transpose(1, 2)  # (N H T D)

        # 矩阵乘法算权重  (N, H, S, K) * (N, H, K, T) -> N, H, S, T
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # (N H S T)

        # 判断是否需要mask
        if attn_mask is not None:
            energy = energy.masked_fill(attn_mask == 0, float('-inf'))

        # softmax计算
        A = torch.softmax(energy, dim=3)  # 对第四维度进行softmax

        # 使用dropout
        A = self.attn_drop(A)

        # 计算加权和  (N, H, S, T) * (N, H, T, K) -> (N, H, S, K)
        Y = A.matmul(V)

        # 再投影回去
        Y = Y.transpose(1, 2).reshape(N, S, E)  # (N, S, E)
        output = self.proj(Y)  # (N, S, E)
        '''



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


