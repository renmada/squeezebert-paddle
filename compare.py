import transformers as tfm
import torch
import numpy as np
import time
from modeling import *
import paddle
from tokenizer import *
from utils import *

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
inputs = ['it is a nice day today!', 'hello_moto']
# inputs = ['it is a nice day today!']

names = ['squeezebert-uncased', 'squeezebert-mnli', 'squeezebert-mnli-headless']

for i in range(3):
    config = configs[i]
    name = names[i]
    model_path = './models/{}/'.format(name)
    paddle_model = SqueezeBertModel(**config)
    states = paddle.load('./models/{}/model_state.pdparams'.format(name))
    paddle_model.set_state_dict(states)
    paddle_model.eval()
    paddle_tokenizer = SqueezeBertTokenizer.from_pretrained(model_path)
    inp = paddle_tokenizer.batch_encode(inputs, return_attention_mask=True)
    input_ids = sequence_padding([x['input_ids'] for x in inp])
    mask = sequence_padding([x['attention_mask'] for x in inp])
    #
    paddle_inp = paddle.to_tensor(input_ids)
    mask = paddle.to_tensor(mask)
    t = time.time()
    with paddle.no_grad():
        paddle_out = paddle_model(paddle_inp, attention_mask=mask)
        # paddle_out = paddle_model(paddle_inp)
    t_squeeze = time.time() - t

    tokenizer = tfm.SqueezeBertTokenizer.from_pretrained(model_path)
    torch_model = tfm.models.squeezebert.SqueezeBertModel.from_pretrained(
        model_path)
    torch_model.eval()
    torch_model.float()
    torch_inp = tokenizer.batch_encode_plus(inputs, return_attention_mask=True, padding=True,
                                            return_tensors='pt', return_token_type_ids=False)
    t = time.time()
    with torch.no_grad():
        torch_out = torch_model(**torch_inp)
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
        print('squeeze paddle  cost {},  squeeze torch cost {}, bert cost {}'.format(t_squeeze * 1000,
                                                                                      t_torch * 1000,
                                                                                      t_bert * 1000))