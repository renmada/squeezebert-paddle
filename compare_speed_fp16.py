import transformers as tfm
import torch
import numpy as np
import time
from modeling import *
import paddle
paddle.set_default_dtype('float16')

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
names = ['squeezebert-uncased']
for i in range(1):
    config = configs[i]
    name = names[i]
    paddle_model = SqueezeBertModel(**config)
    states = paddle.load('C:/Users/xly/Desktop/paddle复现/models/{}/model_state.pdparams'.format(name))
    paddle_model.set_state_dict(states)
    paddle_model.eval()
    input = 'it is a nice day today!'
    tokenizer = tfm.SqueezeBertTokenizer.from_pretrained('C:/Users/xly/Desktop/paddle复现/models/{}/'.format(name))
    #
    input_ids = tokenizer.encode(input, return_tensors='np')
    paddle_inp = paddle.Tensor(input_ids.astype('int64'))
    t = time.time()
    with paddle.no_grad():
        paddle_out = paddle_model(paddle_inp)
    print('here')

    t_squeeze = time.time() - t
    torch_model = tfm.models.squeezebert.SqueezeBertModel.from_pretrained(
        'C:/Users/xly/Desktop/paddle复现/models/{}/'.format(name))
    torch_model.eval()
    # torch_model.float()
    torch_inp = torch.from_numpy(np.array(paddle_inp)).long()
    t = time.time()
    with torch.no_grad():
        torch_out = torch_model(torch_inp)
    t_torch = time.time() - t
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

    print('model_name:', name)
    print("mean difference:", (np.abs(paddle_out[0].numpy() - torch_out[0].numpy())).mean())
    print("max difference:", (np.abs(paddle_out[0].numpy() - torch_out[0].numpy())).max())
    if i == 0:
        from paddlenlp.transformers import BertModel
        bert = BertModel.from_pretrained('bert-base-uncased')
        t = time.time()
        with paddle.no_grad():
            out = bert(paddle_inp)
        t_bert = time.time() - t
        print('squeeze paddle  cost {},  squeeze torch cotst {}, bert cost {}'.format(t_squeeze * 1000,
                                                                                      t_torch * 1000,
                                                                                      t_bert * 1000))



from transformers.models.squeezebert import SqueezeBertModel