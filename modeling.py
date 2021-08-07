# coding=utf-8


import math

import paddle
from paddle import nn
from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from paddlenlp.transformers import PretrainedModel, register_base_model

ACT2FN = {'gelu': nn.GELU()}


def _convert_attention_mask(attention_mask, inputs):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask.unsqueeze(1)
    elif attention_mask.dim() == 2:
        # extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
    extended_attention_mask = paddle.cast(extended_attention_mask, inputs.dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def create_config(kwargs):
    class Config:
        def __init__(self, kwargs):
            for k, v in kwargs.items():
                self.__setattr__(k, v)

    return Config(kwargs)


class SqueezeBertEmbeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", paddle.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64, )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MatMulWrapper(nn.Layer):
    """
    Wrapper for paddle.matmul(). This makes flop-counting easier to implement. Note that if you directly call
    paddle.matmul() in your code, the flop counter will typically ignore the flops of the matmul.
    """

    def __init__(self):
        super().__init__()

    def forward(self, mat1, mat2):
        """
        :param inputs: two paddle tensors :return: matmul of these tensors
        Here are the typical dimensions found in BERT (the B is optional) mat1.shape: [B, <optional extra dims>, M, K]
        mat2.shape: [B, <optional extra dims>, K, N] output shape: [B, <optional extra dims>, M, N]
        """
        return paddle.matmul(mat1, mat2)


class SqueezeBertLayerNorm(nn.LayerNorm):
    """
    This is a nn.LayerNorm subclass that accepts NCW data layout and performs normalization in the C dimension.
    N = batch C = channels W = sequence length
    """

    def __init__(self, hidden_size, epsilon=1e-12):
        nn.LayerNorm.__init__(self, normalized_shape=hidden_size,
                              epsilon=epsilon)  # instantiates self.{weight, bias, eps}

    def forward(self, x):
        x = x.transpose((0, 2, 1))
        x = nn.LayerNorm.forward(self, x)
        return x.transpose((0, 2, 1))


class ConvDropoutLayerNorm(nn.Layer):
    """
    ConvDropoutLayerNorm: Conv, Dropout, LayerNorm
    """

    def __init__(self, cin, cout, groups, dropout_prob):
        super().__init__()

        self.conv1d = nn.Conv1D(in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)
        self.layernorm = SqueezeBertLayerNorm(cout)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        x = self.conv1d(hidden_states)
        x = self.dropout(x)
        x = x + input_tensor
        x = self.layernorm(x)
        return x


class ConvActivation(nn.Layer):
    """
    ConvActivation: Conv, Activation
    """

    def __init__(self, cin, cout, groups, act):
        super().__init__()
        self.conv1d = nn.Conv1D(in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)
        self.act = ACT2FN[act]

    def forward(self, x):
        output = self.conv1d(x)
        return self.act(output)


class SqueezeBertSelfAttention(nn.Layer):
    def __init__(self, config, cin, q_groups=1, k_groups=1, v_groups=1):
        """
        config = used for some things; ignored for others (work in progress...) cin = input channels = output channels
        groups = number of groups to use in conv1d layers
        """
        super().__init__()
        if cin % config.num_attention_heads != 0:
            raise ValueError(
                f"cin ({cin}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(cin / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Conv1D(in_channels=cin, out_channels=cin, kernel_size=1, groups=q_groups)
        self.key = nn.Conv1D(in_channels=cin, out_channels=cin, kernel_size=1, groups=k_groups)
        self.value = nn.Conv1D(in_channels=cin, out_channels=cin, kernel_size=1, groups=v_groups)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(axis=-1)

        self.matmul_qk = MatMulWrapper()
        self.matmul_qkv = MatMulWrapper()

    def transpose_for_scores(self, x):
        """
        - input: [N, C, W]
        - output: [N, C1, W, C2] where C1 is the head index, and C2 is one head's contents
        """
        new_x_shape = (x.shape[0], self.num_attention_heads, self.attention_head_size, x.shape[-1])  # [N, C1, C2, W]
        x = x.reshape(new_x_shape)
        return x.transpose((0, 1, 3, 2))  # [N, C1, C2, W] --> [N, C1, W, C2]

    def transpose_key_for_scores(self, x):
        """
        - input: [N, C, W]
        - output: [N, C1, C2, W] where C1 is the head index, and C2 is one head's contents
        """
        new_x_shape = (x.shape[0], self.num_attention_heads, self.attention_head_size, x.shape[-1])  # [N, C1, C2, W]
        x = x.reshape(new_x_shape)
        # no `permute` needed
        return x

    def transpose_output(self, x):
        """
        - input: [N, C1, W, C2]
        - output: [N, C, W]
        """
        x = x.transpose((0, 1, 3, 2))  # [N, C1, C2, W]
        new_x_shape = (x.shape[0], self.all_head_size, x.shape[3])  # [N, C, W]
        x = x.reshape(new_x_shape)
        return x

    def forward(self, hidden_states, attention_mask, output_attentions):
        """
        expects hidden_states in [N, C, W] data layout.
        The attention_mask data layout is [N, W], and it does not need to be transposed.
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_score = self.matmul_qk(query_layer, key_layer)
        attention_score = attention_score / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_score = attention_score + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_score)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = self.matmul_qkv(attention_probs, value_layer)
        context_layer = self.transpose_output(context_layer)

        result = {"context_layer": context_layer}
        if output_attentions:
            result["attention_score"] = attention_score
        return result


class SqueezeBertLayer(nn.Layer):
    def __init__(self, config):
        """
        - hidden_size = input chans = output chans for Q, K, V (they are all the same ... for now) = output chans for
          the module
        - intermediate_size = output chans for intermediate layer
        - groups = number of groups for all layers in the BertLayer. (eventually we could change the interface to
          allow different groups for different layers)
        """
        super().__init__()

        c0 = config.hidden_size
        c1 = config.hidden_size
        c2 = config.intermediate_size
        c3 = config.hidden_size

        self.attention = SqueezeBertSelfAttention(
            config=config, cin=c0, q_groups=config.q_groups, k_groups=config.k_groups, v_groups=config.v_groups
        )
        self.post_attention = ConvDropoutLayerNorm(
            cin=c0, cout=c1, groups=config.post_attention_groups, dropout_prob=config.hidden_dropout_prob
        )
        self.intermediate = ConvActivation(cin=c1, cout=c2, groups=config.intermediate_groups, act=config.hidden_act)
        self.output = ConvDropoutLayerNorm(
            cin=c2, cout=c3, groups=config.output_groups, dropout_prob=config.hidden_dropout_prob
        )

    def forward(self, hidden_states, attention_mask, output_attentions):
        att = self.attention(hidden_states, attention_mask, output_attentions)
        attention_output = att["context_layer"]

        post_attention_output = self.post_attention(attention_output, hidden_states)
        intermediate_output = self.intermediate(post_attention_output)
        layer_output = self.output(intermediate_output, post_attention_output)

        output_dict = {"feature_map": layer_output}
        if output_attentions:
            output_dict["attention_score"] = att["attention_score"]

        return output_dict


class SqueezeBertEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()

        assert config.embedding_size == config.hidden_size, (
            "If you want embedding_size != intermediate hidden_size,"
            "please insert a Conv1D layer to adjust the number of channels "
            "before the first SqueezeBertLayer."
        )

        self.layers = nn.LayerList(SqueezeBertLayer(config) for _ in range(config.num_hidden_layers))

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):

        if head_mask is None:
            head_mask_is_all_none = True
        elif head_mask.count(None) == len(head_mask):
            head_mask_is_all_none = True
        else:
            head_mask_is_all_none = False
        assert head_mask_is_all_none is True, "head_mask is not yet supported in the SqueezeBert implementation."

        # [batch_size, sequence_length, hidden_size] --> [batch_size, hidden_size, sequence_length]
        hidden_states = hidden_states.transpose((0, 2, 1))

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:

            if output_hidden_states:
                hidden_states = hidden_states.transpose((0, 2, 1))
                all_hidden_states += (hidden_states,)
                hidden_states = hidden_states.transpose((0, 2, 1))

            layer_output = layer.forward(hidden_states, attention_mask, output_attentions)

            hidden_states = layer_output["feature_map"]

            if output_attentions:
                all_attentions += (layer_output["attention_score"],)

        # [batch_size, hidden_size, sequence_length] --> [batch_size, sequence_length, hidden_size]
        hidden_states = hidden_states.transpose((0, 2, 1))

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)


class SqueezeBertPooler(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SqueezeBertPredictionHeadTransform(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class SqueezeBertLMPredictionHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.transform = SqueezeBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)

        # self.bias = nn.Parameter(paddle.zeros(config.vocab_size))
        self.bias = paddle.create_parameter(config.vocab_size, is_bias=True)

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class SqueezeBertOnlyMLMHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.predictions = SqueezeBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class SqueezeBertPreTrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    base_model_prefix = "squeezebert"
    model__file = "model_json"

    # pretrained general uration
    gen_weight = 1.0
    disc_weight = 50.0
    tie_word_embeddings = True
    untied_generator_embeddings = False
    use_softmax_sample = True

    # model init uration
    pretrained_init_uration = {
        "mobilebert-uncased": {
            "attention_probs_dropout_prob": 0.1,
            "classifier_activation": False,
            "embedding_size": 128,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 512,
            "intra_bottleneck_size": 128,
            "key_query_shared_bottleneck": True,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "mobilebert",
            "normalization_type": "no_norm",
            "num_attention_heads": 4,
            "num_feedforward_networks": 4,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "transformers_version": "4.6.0.dev0",
            "trigram_input": True,
            "true_hidden_size": 128,
            "type_vocab_size": 2,
            "use_bottleneck": True,
            "use_bottleneck_attention": False,
            "vocab_size": 30522
        },

    }
    resource_files_names = {"model_state": "model_state.pdparams"}

    # pretrained_resource_files_map = {
    #     "model_state": {
    #         "convbert-base":
    #             "http://paddlenlp.bj.bcebos.com/models/transformers/convbert/convbert-base/model_state.pdparams",
    #         "convbert-medium-small":
    #             "http://paddlenlp.bj.bcebos.com/models/transformers/convbert/convbert-medium-small/model_state.pdparams",
    #         "convbert-small":
    #             "http://paddlenlp.bj.bcebos.com/models/transformers/convbert/convbert-small/model_state.pdparams",
    #     }
    # }

    def init_weights(self):
        """
        Initializes and tie weights if needed.
        """
        # Initialize weights
        self.apply(self._init_weights)
        # Tie weights if needed
        self.tie_weights()

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        if hasattr(self, "get_output_embeddings") and hasattr(
                self, "get_input_embeddings"):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings,
                                           self.get_input_embeddings())

    def _init_weights(self, layer):
        """Initialize the weights"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.squeezebert.cofing["initializer_range"],
                    shape=layer.weight.shape, ))
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.full_like(layer.weight, 1.0))
            layer._epsilon = 1e-12

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone layer weights"""
        if output_embeddings.weight.shape == input_embeddings.weight.shape:
            output_embeddings.weight = input_embeddings.weight
        elif output_embeddings.weight.shape == input_embeddings.weight.t(
        ).shape:
            output_embeddings.weight.set_value(input_embeddings.weight.t())
        else:
            raise ValueError(
                "when tie input/output embeddings, the shape of output embeddings: {}"
                "should be equal to shape of input embeddings: {}"
                "or should be equal to the shape of transpose input embeddings: {}".
                    format(
                    output_embeddings.weight.shape,
                    input_embeddings.weight.shape,
                    input_embeddings.weight.t().shape, ))
        if getattr(output_embeddings, "bias", None) is not None:
            if output_embeddings.weight.shape[
                -1] != output_embeddings.bias.shape[0]:
                raise ValueError(
                    "the weight lase shape: {} of output_embeddings is not equal to the bias shape: {}"
                    "please check output_embeddings uration".format(
                        output_embeddings.weight.shape[-1],
                        output_embeddings.bias.shape[0], ))


# class SqueezeBertPreTrainedModel(PreTrainedModel):
#     """
#     An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
#     models.
#     """
#
#     base_model_prefix = "transformer"
#     _keys_to_ignore_on_load_missing = [r"position_ids"]
#
#     def _init_weights(self, module):
#         """Initialize the weights"""
#         if isinstance(module, (nn.Linear, nn.Conv1D)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pypaddle/pypaddle/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, SqueezeBertLayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)


SQUEEZEBERT_START_DOCSTRING = r"""
    The SqueezeBERT model was proposed in `SqueezeBERT: What can computer vision teach NLP about efficient neural
    networks? <https://arxiv.org/abs/2006.11316>`__ by Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, and Kurt W.
    Keutzer
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
    This model is also a PyTorch `paddle.nn.Layer <https://pypaddle.org/docs/stable/nn.html#paddle.nn.Layer>`__
    subclass. Use it as a regular PyTorch Layer and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    For best results finetuning SqueezeBERT on text classification tasks, it is recommended to use the
    `squeezebert/squeezebert-mnli-headless` checkpoint as a starting point.
    Parameters:
        config (:class:`~transformers.SqueezeBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
    Hierarchy::
        Internal class hierarchy:
            SqueezeBertModel
                SqueezeBertEncoder
                    SqueezeBertLayer
                    SqueezeBertSelfAttention
                        ConvActivation
                        ConvDropoutLayerNorm
    Data layouts::
        Input data is in [batch, sequence_length, hidden_size] format.
        Data inside the encoder is in [batch, hidden_size, sequence_length] format. But, if :obj:`output_hidden_states
        == True`, the data from inside the encoder is returned in [batch, sequence_length, hidden_size] format.
        The final output of the encoder is in [batch, sequence_length, hidden_size] format.
"""

SQUEEZEBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`paddle.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`~transformers.SqueezeBertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`paddle.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`paddle.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:
            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`paddle.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`paddle.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (:obj:`paddle.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


class SqueezeBertModel(SqueezeBertPreTrainedModel):
    def __init__(self, **kwargs):
        super().__init__()
        config = self.config = create_config(kwargs)
        self.embeddings = SqueezeBertEmbeddings(config)
        self.encoder = SqueezeBertEncoder(config)
        self.pooler = SqueezeBertPooler(config)

        # self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = paddle.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        extended_attention_mask = _convert_attention_mask(attention_mask, embedding_output)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        return (sequence_output, pooled_output) + encoder_outputs[1:]


class SqueezeBertForMaskedLM(SqueezeBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.transformer = SqueezeBertModel(config)
        self.cls = SqueezeBertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`paddle.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.reshape(-1, self.config.vocab_size), labels.reshape(-1))

        output = (prediction_scores,) + outputs[2:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output


class SqueezeBertForSequenceClassification(SqueezeBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.transformer = SqueezeBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`paddle.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, self.num_labels), labels.reshape(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class SqueezeBertForMultipleChoice(SqueezeBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.transformer = SqueezeBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`paddle.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where `num_choices` is the size of the second dimension of the input tensors. (see
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.reshape(-1, input_ids.shape(-1)) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape(-1)) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.reshape(-1, inputs_embeds.shape(-2), inputs_embeds.shape(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        output = (reshaped_logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class SqueezeBertForTokenClassification(SqueezeBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = SqueezeBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`paddle.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.reshape(-1) == 1
                active_logits = logits.reshape(-1, self.num_labels)
                active_labels = paddle.where(
                    active_loss, labels.reshape(-1), paddle.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.reshape(-1, self.num_labels), labels.reshape(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class SqueezeBertForQuestionAnswering(SqueezeBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = SqueezeBertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        start_positions (:obj:`paddle.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (:obj:`paddle.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output


if __name__ == '__main__':
    import transformers as tfm
    import torch
    import numpy as np
    import time

    configs = [{
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "model_type": "squeezebert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size": 30528,
        "q_groups": 4,
        "k_groups": 4,
        "v_groups": 4,
        "post_attention_groups": 1,
        "intermediate_groups": 4,
        "output_groups": 4,
        "pad_token_id": 0,
        'layer_norm_eps': 1e-12
    },
        {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 768,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "model_type": "squeezebert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 30528,
            "q_groups": 4,
            "k_groups": 4,
            "v_groups": 4,
            "post_attention_groups": 1,
            "intermediate_groups": 4,
            "output_groups": 4,
            "num_labels": 3,
            "pad_token_id": 0,
            'layer_norm_eps': 1e-12
        },
        {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 768,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "model_type": "squeezebert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 30528,
            "q_groups": 4,
            "k_groups": 4,
            "v_groups": 4,
            "post_attention_groups": 1,
            "intermediate_groups": 4,
            "output_groups": 4,
            "pad_token_id": 0,
            'layer_norm_eps': 1e-12
        }
    ]
    names = ['squeezebert-uncased', 'squeezebert-mnli', 'squeezebert-mnli-headless']
    for i in range(3):
        config = configs[i]
        name = names[i]
        paddle_model = SqueezeBertModel(**config)
    # paddle_parameters = list(paddle_model.named_parameters())
    # torch_parameters = list(torch.load('../models/squeezebert-uncased/pytorch_model.bin').items())
    #
    # for (p_name, pw), (t_name, tw) in zip(paddle_parameters, torch_parameters):
    #     if t_name.startswith('transformer.'):
    #         t_name = t_name.replace('transformer.', '')
    #     assert t_name == p_name, (p_name, t_name)
    #     p_shape = pw.shape
    #     t_shape = list(tw.shape)
    #     if p_shape != t_shape:
    #         print(p_name)

    # for i in paddle_model.named_parameters():
    #     print(i[0], i[1].shape)

        states = paddle.load('C:/Users/xly/Desktop/paddle复现/models/{}/model_state.pdparams'.format(name))
        paddle_model.set_state_dict(states)
        paddle_model.eval()
        input = 'it is a nice day today!'
        tokenizer = tfm.SqueezeBertTokenizer.from_pretrained('C:/Users/xly/Desktop/paddle复现/models/{}/'.format(name))
        #
        input_ids = tokenizer.encode(input, return_tensors='np')
        paddle_inp = paddle.Tensor(input_ids)
        t= time.time()
        with paddle.no_grad():
            paddle_out = paddle_model(paddle_inp)
        t_squeeze = time.time() - t
        torch_model = tfm.models.squeezebert.SqueezeBertModel.from_pretrained(
            'C:/Users/xly/Desktop/paddle复现/models/{}/'.format(name))
        torch_model.eval()
        torch_model.float()
        torch_inp = torch.from_numpy(np.array(paddle_inp)).long()
        with torch.no_grad():
            torch_out = torch_model(torch_inp)

        for a, b in zip(paddle_model.named_parameters(), torch_model.named_parameters()):
            t1 = a[1].numpy()
            t2 = b[1].detach().numpy()
            shape1 = t1.shape
            shape2 = t2.shape
            if shape1 != shape2:
                t2 = t2.T
            if (t1 - t2).sum() > 0.1:
                print(a[0], a[0] == b[0])
                print(t1.shape)
                print((t1 - t2).sum())

    # # print((paddle_out[0].numpy() - torch_out[0].numpy()).mean())
    # print(paddle_out[0].numpy().reshape(-1)[0], torch_out[0].numpy().reshape(-1)[0])
    # # print(f"huggingface {model[0]} vs paddle {model[1]}")
        print('model_name:', name)
        print("min difference:", (np.abs(paddle_out[0].numpy() - torch_out[0].numpy())).min())
        print("max difference:", (np.abs(paddle_out[0].numpy() - torch_out[0].numpy())).max())
        if i == 0:
            from paddlenlp.transformers import BertModel
            bert = BertModel.from_pretrained('bert-base-uncased')
            t = time.time()
            with paddle.no_grad():
                out = bert(paddle_inp)
            t_bert = time.time() - t
            print('squeeze cost {}, bert cost {}'.format(t_squeeze*1000, t_bert*1000))