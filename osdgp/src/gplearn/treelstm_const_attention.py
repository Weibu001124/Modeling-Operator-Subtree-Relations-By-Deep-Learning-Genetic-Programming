import torch
import torch.nn as nn
import numpy as np
import random
import math

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class TreeNode:
    def __init__(self, token, left=None, right=None):
        self.token = token  # int id
        self.left = left
        self.right = right
    def __str__(self):
        if self.left is None and self.right is None:
            if isinstance(self.token, int):
                return f'X{self.token}'
            return str(self.token)
        if self.right is None:
            return f'{self.token}({self.left})'
        return f'{self.token}({self.left}, {self.right})'

class AttentiveMixedArityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim

        # 2019 Self-Attention
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # gates for i, f, o, u
        self.W_i = nn.Linear(embedding_dim, hidden_dim)
        self.U_i = nn.Linear(2 * hidden_dim, hidden_dim)

        self.W_f = nn.Linear(embedding_dim, hidden_dim)
        self.U_f_l = nn.Linear(2 * hidden_dim, hidden_dim)
        self.U_f_r = nn.Linear(2 * hidden_dim, hidden_dim)

        self.W_o = nn.Linear(embedding_dim, hidden_dim)
        self.U_o = nn.Linear(2 * hidden_dim, hidden_dim)

        self.W_u = nn.Linear(embedding_dim, hidden_dim)
        self.U_u = nn.Linear(2 * hidden_dim, hidden_dim)

        # For unary node (only one child), reduce dimensions
        self.U_i_unary = nn.Linear(hidden_dim, hidden_dim)
        self.U_f_unary = nn.Linear(hidden_dim, hidden_dim)
        self.U_o_unary = nn.Linear(hidden_dim, hidden_dim)
        self.U_u_unary = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node):
        if isinstance(node.token, float):
            token_id = self.vocab_size - 1
        else:
            token_id = node.token
        x = self.embedding(torch.tensor([token_id], dtype=torch.long))

        # Leaf node
        if node.left is None and node.right is None:
            h = torch.tanh(self.W_u(x))
            c = torch.zeros_like(h)
            return h, c

        # Unary node (only one child)
        if node.left is not None and node.right is None:
            h_child, c_child = self.forward(node.left)
            i = torch.sigmoid(self.W_i(x) + self.U_i_unary(h_child))
            f = torch.sigmoid(self.W_f(x) + self.U_f_unary(h_child))
            o = torch.sigmoid(self.W_o(x) + self.U_o_unary(h_child))
            u = torch.tanh(self.W_u(x) + self.U_u_unary(h_child))

            c = i * u + f * c_child
            h = o * torch.tanh(c)
            return h, c

        # Binary node
        h_l, c_l = self.forward(node.left)
        h_r, c_r = self.forward(node.right)
        
        # 2019 Self-Attention (Model 1)
        # Stack the hidden states of the left and right child nodes
        M = torch.stack([h_l, h_r], dim=1) 

        # Generate Query, Key, and Value matrices through linear transformations
        Q = self.W_q(M)
        K = self.W_k(M)
        V = self.W_v(M)
        
        # Calculate the attention alignment scores
        d_k = self.hidden_dim
        align = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
        
        # Apply Softmax to obtain the attention probability distribution (alpha)
        alpha = torch.softmax(align, dim=-1)
        
        # Generate the new attention-encoded hidden states
        h_tilde = torch.bmm(alpha, V)
        
        # Split the updated states back into left and right child representations
        h_l_tilde = h_tilde[:, 0, :]
        h_r_tilde = h_tilde[:, 1, :]
        
        # Original: h_cat = torch.cat([h_l, h_r], dim=1)
        h_cat = torch.cat([h_l_tilde, h_r_tilde], dim=1)

        i = torch.sigmoid(self.W_i(x) + self.U_i(h_cat))
        f_l = torch.sigmoid(self.W_f(x) + self.U_f_l(h_cat))
        f_r = torch.sigmoid(self.W_f(x) + self.U_f_r(h_cat))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_cat))
        u = torch.tanh(self.W_u(x) + self.U_u(h_cat))

        c = i * u + f_l * c_l + f_r * c_r
        h = o * torch.tanh(c)
        return h, c