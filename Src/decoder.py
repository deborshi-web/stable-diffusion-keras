import torch
from torch import nn
from torch.nn import functional as F
import tensorflow as tf


class AttentionBlock(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = tf.keras.layers.GroupNormalization(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        residue = x
        x = self.groupnorm(x)

        n, c, h, w = x.shape
        x = tf.transpose((n, c, h * w))
        x = self.attention(x)
        x=tf.transpose(-1, -2)
        x=tf.transpose((n, c, h, w))
        x += residue
        return x

class ResidualBlock(keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupnorm_1 = tf.keras.layers.GroupNormalization(32, in_channels)

        self.conv_1 = tf.keras.layers.Conv2D(in_channels, out_channels,padding='valid')

        self.groupnorm_2 = tf.keras.layers.GroupNormalization(32, in_channels)

        self.conv_2 = tf.keras.layers.Conv2D(in_channels, out_channels,padding='valid')

        if in_channels == out_channels:
            self.residual_layer = tf.identity(x)
        else:
            self.residual_layer = tf.keras.layers.Conv2D(in_channels, out_channels,padding='valid')

    def forward(self, x):
        residue = x

        x = self.groupnorm_1(x)
        silu = silu = tf.keras.layers.Activation('silu')
        silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_1(x)
        silu = silu = tf.keras.layers.Activation('silu')
        silu(x)
        x = self.conv_1(x)
        residual_layer+=residue

              #x + self.residual_layer(residue)


class Decoder(keras.layers.Layer):
    def __init__(self):
      super().__init__()
      tf.keras.layers.Conv2D(4, 4, padding='valid'),
      tf.keras.layers.Conv2D(4, 512, padding='valid'),
      ResidualBlock(512, 512),
      AttentionBlock(512),
      ResidualBlock(512, 512),
      ResidualBlock(512, 512),
      ResidualBlock(512, 512),
      ResidualBlock(512, 512),
      tf.keras.layers.UpSampling2D(size=2),
      tf.keras.layers.Conv2D(512, 512,padding='valid'),
      ResidualBlock(512, 512),
      ResidualBlock(512, 512),
      ResidualBlock(512, 512),
      tf.keras.layers.UpSampling2D(size=2),
      tf.keras.layers.Conv2D(512, 512,padding='valid'),
      ResidualBlock(512, 256),
      ResidualBlock(256, 256),
      ResidualBlock(256, 256),
      tf.keras.layers.UpSampling2D(size=2),
      tf.keras.layers.Conv2D(256, 256,padding='valid'),
      ResidualBlock(256, 128),
      ResidualBlock(128, 128),
      ResidualBlock(128, 128),
      tf.keras.layers.GroupNormalization(32, 128),
      tf.keras.layers.Activation('silu'),
      tf.keras.layers.Conv2D(128, 3,padding='valid')

    def forward(self, x):
        x /= 0.18215
        return x