import torch
import torch.nn as nn
import copy
from torch.nn import CrossEntropyLoss, MSELoss
from torch import Tensor, device, dtype, nn
from typing import Callable, Dict, List, Optional, Tuple
from collections import OrderedDict
import pickle
import os
class MetaBertEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(30522, 768, padding_idx=0)
        self.position_embeddings = nn.Embedding(512, 768)
        self.token_type_embeddings = nn.Embedding(2, 768)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(768)#,ln_bs = config.ln_bs,max_seq_len = config.max_seq_len, train_mode = config.train_mode)
        self.dropout = nn.Dropout(0)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None,params =None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)#,self.get_subdict(params,'word_embeddings'))

        position_embeddings = self.position_embeddings(position_ids)#, self.get_subdict(params,'position_embeddings'))
        token_type_embeddings = self.token_type_embeddings(token_type_ids)#, self.get_subdict(params, 'token_type_embeddings'))

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        #print(embeddings)
        embeddings = self.LayerNorm(embeddings)#, self.get_subdict(params, 'LayerNorm'))
        embeddings = self.dropout(embeddings)
        return embeddings

class BertEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = MetaBertEmbeddings()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple, device: device) -> Tensor:
        extended_attention_mask = attention_mask[:, None, None, :]
        self.dtype = list(self.parameters())[0].dtype
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    def get_head_mask(self, head_mask: Tensor, num_hidden_layers: int, is_attention_chunked: bool = False) -> Tensor:
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to fload if need + fp16 compatibility
        return head_mask
    def forward(self,input_ids=None):

        input_shape = input_ids.size()
        device = input_ids.device

        attention_mask = torch.ones(input_shape, device=device)
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        if False and encoder_hidden_states is not None: #TODO #self.config.deceoder
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(None, 768)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=None)
        return embedding_output

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 =  nn.Linear(768, 4)
        self.num_labels = 4

    def forward(self,x):
        #pooled_output = self.dropout(x)
        pooled_output = x
        logits = self.linear_1(pooled_output)#, self.get_subdict(params, 'linear_1'))

        return logits.view(-1,self.num_labels)


def get_pre_trained(path):
    model = BertEmbeddingModel().cuda()
    decoder_model = Classifier().cuda()
    encoder_save_model_path = os.path.join(path, 'pre_trained_encoder_checkpoint-{}'.format(0))
    decoder_save_model_path = os.path.join(path, 'pre_trained_decoder_checkpoint-{}'.format(0))
    model_state_dict = torch.load(encoder_save_model_path)
    decoder_state_dict = torch.load(decoder_save_model_path)

    model.load_state_dict(model_state_dict)
    decoder_model.load_state_dict(decoder_state_dict)
    return  model, decoder_model


def forward(input_ids,labels, model, decoder_model):
    loss_fct = nn.CrossEntropyLoss().cuda()
    #input_ids, masks, labels = load_container(args)
    outputs =  model(input_ids)
    outputs = outputs.sum(axis =1).float()
    logits  = decoder_model(outputs)
    loss = loss_fct(logits, labels.view(-1))
    return loss

def compute_grad(input_ids,labels ,model, decoder_model, create_graph= False):
    loss = forward(input_ids,labels,model,decoder_model)
    loss.backward(create_graph = create_graph)
    model_params = OrderedDict(model.named_parameters())
    decoder_params = OrderedDict(decoder_model.named_parameters())
    if create_graph:
        print("when create_graph is True")
        print('loss                     :{}'.format(loss))
        print("classifier grad norm     :{}".format(torch.norm(decoder_params['linear_1.weight'].grad)))
        print("bert embedding grad norm :{}".format(torch.norm(model_params['embeddings.word_embeddings.weight'].grad)))
    else:
        print("When create_graph is False")
        print('loss                     :{}'.format(loss))
        print("classifier grad norm     :{}".format(torch.norm(decoder_params['linear_1.weight'].grad)))
        print("bert embedding grad norm :{}".format(torch.norm(model_params['embeddings.word_embeddings.weight'].grad)))

def main():
    base_path = "models/"
    model, decoder_model = get_pre_trained(base_path)
    input_ids = torch.randint(0, 30000, (40,113)).cuda()
    labels = torch.randint(0,3,(40,)).cuda()

    compute_grad(input_ids,labels, copy.deepcopy(model), copy.deepcopy(decoder_model), create_graph = True)
    compute_grad(input_ids,labels, copy.deepcopy(model), copy.deepcopy(decoder_model), create_graph = False)
main() 


