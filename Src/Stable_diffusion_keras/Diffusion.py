import torch
from torch import nn
from torch.nn import functional as F
from attention import selfattention, cross_attention
import tensorflow as tf


class TimeEmbedding(keras.layers.Layer):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = tf.keras.layers.Dense(4 * n_embd)
        self.linear_2 = tf.keras.layers.Dense(4 * n_embd)

    def forward(self, x):
        x = self.linear_1(x)
        x = tf.keras.activations.silu(x)
        x = self.linear_2(x)
        return x

class ResidualBlock(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = tf.keras.layers.GroupNormalization(32, in_channels)
        self.conv_feature = tf.keras.layers.Conv2D(in_channels, out_channels, padding='valid')
        self.linear_time = tf.keras.layers.Dense(n_time)

        self.groupnorm_merged = tf.keras.layers.GroupNormalization(32, out_channels)
        self.conv_merged = tf.keras.layers.Conv2D(out_channels, out_channels, padding='valid')

        if in_channels == out_channels:
            self.residual_layer = tf.keras.layers.Identity()
        else:
            self.residual_layer = tf.keras.layers.Conv2D(in_channels, out_channels, padding='valid')

    def forward(self, feature, time):
        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = tf.keras.activations.silu(feature)
        feature = self.conv_feature(feature)

        time = tf.keras.activations.silu(time)
        time = self.linear_time(time)

        merged = feature + time
        merged = self.groupnorm_merged(merged)
        merged = tf.keras.activations.silu(feature)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)

class AttentionBlock(keras.layers.Layer):
    def __init__(self, n_head: int, n_embd: int, d_context=1):
        super().__init__()

        channels = n_head * n_embd
        self.groupnorm = tf.keras.layers.GroupNormalization(32, channels, epsilon=0.01)
        self.conv_input = tf.keras.layers.Conv2D(channels, channels,padding='valid')

        self.layernorm_1 = tf.keras.layers.LayerNormalization(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = tf.keras.layers.LayerNormalization(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = tf.keras.layers.LayerNormalization(channels)
        self.linear_geglu_1  = tf.keras.layers.Dense(channels)   #(channels, 4 * channels * 2)
        self.linear_geglu_2 = tf.keras.layers.Dense(channels)

        self.conv_output = tf.keras.layers.Conv2D(channels, channels,padding='valid')

    def forward(self, x, context):
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape
        x = x = tf.reshape(x,(n, c, h * w))   # (n, c, hw)
        x = tf.transpose(x,perm=[0,2,1])  # (n, hw, c)

        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1.forward(tf.keras.random.normal((3,2,1)))
        x = residue_short

        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2.forward(x=tf.keras.random.normal((1,1,1)),y=tf.keras.random.normal((1,1,1)))
        x += residue_short

        residue_short = x
        x = self.layernorm_3(x)
        x= self.linear_geglu_1(x)
        gate= tf.split(x,num_or_size_splits=2,axis=1)
        x=tf.keras.activations.gelu(gate)
        x = self.linear_geglu_2(x)
        x = residue_short

        x = tf.reshape(x,(n, c, h * w))  # (n, c, hw)
        x = tf.transpose(x,perm=[0,2,1])    # (n, c, h, w)



class Upsample(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(channels, channels, padding='valid')

    def forward(self, x):
      x=tf.image.resize(x,size=(2,2),method='nearest')
      return self.conv(x)

class SwitchSequential(keras.layers.Layer):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNet(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.encoders =([
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
        ])
        self.bottleneck = SwitchSequential(
            )
        self.decoders =([
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
            SwitchSequential(),
        ])

    def forward(self, x, context, time):
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            x = tf.concat(2,axis=0)
            x = layers(x, context, time)

        return x


class FinalLayer(keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = tf.keras.layers.GroupNormalization(32, in_channels)
        self.conv =tf.keras.layers.Conv2D(in_channels, out_channels, padding='valid')

    def forward(self, x):
        x = self.groupnorm(x)
        x = x = tf.keras.activations.silu(x)
        x = self.conv(x)
        return x

class Diffusion(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = FinalLayer(320, 4)

    def forward(self, latent, context, time):
        time = self.time_embedding(time)
        output = self.unet(latent, context, time)
        output = self.final(output)
        return output
