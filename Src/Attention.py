class SelfAttention(keras.layers.Layer):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = tf.keras.layers.Dense(d_embed)
        self.out_proj = tf.keras.layers.Dense(d_embed)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        q, k, v=tf.split(self.in_proj(x), 3)

        k=tf.transpose(k,perm=[0,2,1])

        weight=q@k

        if causal_mask:
          mask =tf.ones_like(weight, dtype=tf.bool).triu(1)
          weight.masked_fill_(mask)
        weight /= math.sqrt(self.d_head)
        active = tf.keras.layers.Activation('softmax')
        weight = active(weight)

        output = weight @ v
        output=tf.transpose(output,perm=[1,0,2])
        #output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output.shape
