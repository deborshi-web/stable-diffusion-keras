import torch
from torch import nn
from torch.nn import functional as F
from attention import selfattention
import tensorflow as tf


class CLIPEmbedding(keras.layers.Layer):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        self.token_embedding=tf.keras.layers.Embedding(n_vocab, n_embd)
        tf.Variable(tf.zeros((n_token, n_embd)))
        #self.position_value = nn.Parameter(torch.zeros((n_token, n_embd)))

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_value
        return x


class CLIPLayer(keras.layers.Layer):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.layernorm_1 =  tf.keras.layers.LayerNormalization(n_embd) #nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = tf.keras.layers.LayerNormalization(n_embd)
        self.linear_1 = tf.keras.layers.Dense(4 * n_embd)
        self.linear_2 = tf.keras.layers.Dense(4 * n_embd)

    def forward(self, x):
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * tf.sigmoid(1.702 * x) # QuickGELU activation function
        x = self.linear_2(x)
        x += residue

        return x

class CLIP(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers =CLIPLayer(12, 768)

        self.layernorm = tf.keras.layers.LayerNormalization(768)

    def forward(self, tokens):
        #tokens = tokens.type(tf.int64)
        tokens =tf.int64
        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)
        return output
