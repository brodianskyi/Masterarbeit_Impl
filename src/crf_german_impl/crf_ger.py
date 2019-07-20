import tensorflow as tf
from keras import backend as K
from keras.engine import Layer, InputSpec

def crf_decode(potentials, transition_params, sequence_length):
    """
    Decode the highest scoring sequence of tags in TensorFlow.
    potentials: <Tensor "embedding" shape=(2,2,11), dtype=float32>
                [batch_size, max_seq_len, num_tags] - Matrix of unary potentials

    transition_params: <Tensor> shape=(11,11)
                       [n_classes, n_classes] - matrix of binary potentials
    sequence_length: <Tensor> shape=(2,1) -> Flatten -> shape(2,)
                    [batch_size] - containing sequence length
    Return:
      decode_tags: [batch_size, max_seq_len], with dtype tf.int32 - the highest scoring tag indicies
      best_score:  [batch_size] -  the score of decode_tags
    """
    # B - is the batch_size
    # T - max_seq_len
    # num_tags by "0" (output)

    # num_tags = 2
    num_tags = potentials.get_shape()[2].value
    # computes



class CRFLayer(Layer):

    def __init__(self, transition_params=None, **kwargs):
        super(CRFLayer, self).__init__(**kwargs)
        self.transition_params = transition_params
        # InputSpec - specifies the input_dim of every input to a layer
        # Param for InputSpec - ndim -  Integer, expected rank(dim) of the input
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=2)]
        self.supports_masking = True

    def build(self, input_shape):
        """
        Create the layer weights.
        Args:
            crf([word_embeddings, sequence_length])
            input_shape (list(tuple, tuple)): [(batch_size, n_step, n_classes), (batch_size, 1)]
            input_shape{list}:[(2, 2, 11),(2, 1)]

        """
        # test if the condition true if false then exception
        assert len(input_shape) == 2
        # (batch_size, n_step, n_classes)
        assert len(input_shape[0]) == 3
        # batch_size, 1)
        assert len(input_shape[1]) == 2
        n_steps = input_shape[0][1]
        n_classes = input_shape[0][2]
        # check n_steps, condition must be true
        assert n_steps is None or n_steps >= 2
        # Adds a weight variable to the layer.
        # @interfaces.legacy_add_weight_support
        # def add_weight(self, shape, initializer, name)
        self.transition_params = self.add_weight(shape=(n_classes, n_classes),
                                                 initializer="uniform",
                                                 name="transition")
        # InputSpec - specifies the input_dim of every input to a layer
        # Param for InputSpec - dtype - expected DataType (K.float = 'float32') of the input;
        # shape =  Shape tuple, expected shape of the input (may include None for unchecked axes)
        # self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=2)] (from __init__ method)
        # [InputSpec(dtype=float32, shape=(None, 2, 11), ndim=3), InputSpec(dtype=int32, shape=(None, 1), ndim=2]
        self.input_spec = [InputSpec(dtype=K.floatx(), shape=(None, n_steps, n_classes)),
                           InputSpec(dtype="int32", shape=(None, 1))]
        # This method must set self.built = True at the end
        self.built = True
        # then go to the method call

    def call(self, inputs, mask=None, **kwargs):
        # layer logic implementation
        # call(inputs = <list>: [<Tensor "embedding" shape=(2,2,11), dtype=float32>, <Tensor> shape=(2,1), dtype=int32]) and after
        # inputs = <Tensor "embedding" shape=(2,2,11), dtype=float32>
        # sequence_length = <Tensor> shape=(2,1), dtype=int32
        inputs, sequence_length = inputs
        #  Flatten a tensor, reshape into 1-D; from shape=(2,1) in shape(2,)
        # definition of sequence_length outside __init__ method
        self.sequence_length = K.flatten(sequence_length)
        y_pred = self.viterbi_decode(inputs, self.sequence_length)

    def viterby_decode(self, potentials, sequence_length):
        '''
        Decode the highest scoring sequence of tags in TensorFlow
        Arg:
           potentials: inputs = <Tensor "embedding" shape=(2,2,11), dtype=float32>
                       [batch_size, max_seq_len, num_tags] - Matrix of unary potentials
           sequence_length: <Tensor> shape=(2,1) -> Flatten -> shape(2,)
                           [batch_size] - containing sequence length
        Return:
            decode_tags: [batch_size, max_seq_len], with dtype tf.int32
            contains the highest scoring tag indicies
        '''
        decode_tags, best_score = crf_decode(potentials, self.transition_params, sequence_length)

        return decode_tags






    def compute_output_shape(self, input_shape):
        # modify the shape of input
        return input_shape[0]
