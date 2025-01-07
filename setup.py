import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Self Attention"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.d_k = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
    
    def scale_dot_product_attention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weight = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weight, V)
        return output
    
    def forward(self, X):
        batch_size, seq_len, _ = X.size()

        Q = self.q_linear(X)
        K = self.k_linear(X)
        V = self.v_linear(X)

        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        output = self.scale_dot_product_attention(Q, K, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.output_linear(output)


if __name__ == "__main__":
    batch_size, seq_len, d_model, n_heads = 32, 10, 512, 8
    X = torch.randn(batch_size, seq_len, d_model)
    mha = MultiHeadAttention(d_model, n_heads)
    output = mha(X)
    print(output.size())
