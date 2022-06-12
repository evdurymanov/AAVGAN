import tensorflow as tf
from gan.protein.protein import NUM_AMINO_ACIDS
from gan.sngan.generator import Generator
from common.model import ops
from common.model.generator_ops import get_dimentions_factors, get_kernel, sn_block, sn_block_conditional
from tensorflow_probability.python.distributions import RelaxedOneHotCategorical


class GumbelGeneratorCond(Generator):
    def __init__(self, config, shape, num_classes=None, scope_name=None):
        super(GumbelGeneratorCond, self).__init__(config, shape, num_classes, scope_name)
        self.strides = self.get_strides()
        self.number_of_layers = len(self.strides)
        self.starting_dim = self.dim * (2 ** self.number_of_layers)
        self.get_initial_shape(config)
        self.final_bn = ops.BatchNorm(name='g_bn')

    def get_strides(self):
        strides = [(1, 2), (1, 2), (1, 2), (1, 2)]
        if self.length == 512:
            strides.extend([(1, 2), (1, 2)])
        return strides

    def network(self, z, labels, reuse):

        # Fully connected
        i_shape = self.initial_shape
        h = ops.snlinear(z, i_shape[1] * i_shape[2] * i_shape[3], name='noise_linear')
        h = tf.reshape(h, i_shape)
        # Resnet architecture
        hidden_dim = self.starting_dim
        for layer_id in range(self.number_of_layers):
            self.log(h.shape)
            block_name, dilation_rate, hidden_dim, stride = self.get_block_params(hidden_dim, layer_id)
            h = self.add_cond_sn_block(h, hidden_dim, block_name, dilation_rate, stride, labels)
            if layer_id == self.number_of_layers - 2:
                h = self.add_attention(h, hidden_dim, reuse)
                hidden_dim = hidden_dim*2

        # Final conv
        h_act = self.act(self.final_bn(h), name="h_act")
        last = ops.snconv2d(h_act, NUM_AMINO_ACIDS, (1, 1), name='last_conv')

        # Gumbel max trick
        out = RelaxedOneHotCategorical(temperature=self.get_temperature(True), logits=last).sample()
        return out


    def add_cond_sn_block(self, x, out_dim, block_name, dilation_rate, strides, labels):
        kernel = get_kernel(x, self.kernel)
        h = sn_block_conditional(x, labels[0], out_dim, self.num_classes[0], block_name, kernel, strides, dilation_rate, self.act, self.pooling, 'VALID')
        tf.summary.histogram(block_name, h, family=self.scope_name)
        return h