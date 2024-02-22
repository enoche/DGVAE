import enum
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch import Tensor


def mlp_layers(layer_dims):
    mlp_modules = []
    for i, (d_in, d_out) in enumerate(zip(layer_dims[: -1], layer_dims[1:])):
        layer = nn.Linear(d_in, d_out)
        nn.init.xavier_normal_(layer.weight)
        nn.init.normal_(layer.bias, std=0.01)
        mlp_modules.append(layer)
        del layer
        if i != len(layer_dims[:-1]) - 1:
            mlp_modules.append(nn.Tanh())
    return nn.Sequential(*mlp_modules)


class MacridVAE(nn.Module):
    def __init__(self,
                 item_adj,
                 num_items: int,
                 layers: list,
                 emb_size: int,
                 dropout: float,
                 num_prototypes: int,
                 tau: float = 1.,
                 nogb: bool = False,
                 anneal_cap: float = 0.2,
                 std: float = 1.,
                 reg_weights: List[float] = None,
                 total_anneal_steps: int = 20000):
        super(MacridVAE, self).__init__()
        if reg_weights is None:
            reg_weights = [0.0, 0.0]
        self.item_adj = item_adj
        self.num_items = num_items
        self.layers = layers
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_prototypes = num_prototypes
        self.tau = tau
        self.nogb = nogb
        self.anneal_cap = anneal_cap
        self.std = std
        self.reg_weights = reg_weights
        self.update = 0
        self.total_anneal_steps = total_anneal_steps

        self.encoder_layer_dims = [self.num_items] + self.layers + [self.emb_size * 2]
        #self.encoder = mlp_layers(self.encoder_layer_dims)
        self.encoder = nn.Linear(self.num_items, self.emb_size*2)
        nn.init.xavier_normal_(self.encoder.weight)
        nn.init.normal_(self.encoder.bias, std=0.01)

        self.item_embeddings = nn.Embedding(self.num_items, self.emb_size)
        nn.init.xavier_normal_(self.item_embeddings.weight)

        self.prototype_embeddings = nn.Embedding(self.num_prototypes, self.emb_size)
        nn.init.xavier_normal_(self.prototype_embeddings.weight)

        self.dropout_layer = nn.Dropout(self.dropout)

        self.loss = nn.MSELoss(reduction='sum')

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=self.std)
            return mu + epsilon * std
        return mu

    def encode(self, input_rating):
        prototypes = fn.normalize(self.prototype_embeddings.weight, dim=1)
        items = fn.normalize(self.item_embeddings.weight, dim=1)
        # # GNN
        # h = self.item_embeddings.weight
        # if self.item_adj is not None:
        #     for i in range(1):
        #         h = torch.sparse.mm(self.item_adj, h)
        # items = fn.normalize(h, dim=1)

        cates_logits = items.matmul(prototypes.transpose(0, 1)) / self.tau

        if self.nogb:
            cates = cates_logits.softmax(dim=1)
        else:
            cates_sample = fn.gumbel_softmax(cates_logits, tau=1, hard=False, dim=-1)
            cates_mode = cates_logits.softmax(dim=-1)
            cates = self.training * cates_sample + (1 - self.training) * cates_mode  # [num_items, num_prototypes]

        mu_list = []
        logvar_list = []
        z_list = []
        cates_k_list = []

        for k in range(self.num_prototypes):
            cates_k = cates[:, k].reshape(1, -1)  # [1, num_items]
            cates_k_list.append(cates_k)
            # encoder
            x_k = input_rating * cates_k  # [batch_size, num_items]
            x_k = fn.normalize(x_k, p=2)
            tmp_h = x_k
            if self.item_adj is not None:
                for i in range(2):
                    tmp_h = torch.mm(tmp_h, self.item_adj.to_dense())
            x_k = self.dropout_layer((tmp_h+x_k)/2)
            #x_k = self.dropout_layer(x_k)
            h = self.encoder(x_k)

            mu = h[:, : self.emb_size]
            mu = fn.normalize(mu, dim=1)
            logvar = -h[:, self.emb_size:]

            mu_list.append(mu)
            logvar_list.append(logvar)

            z = self.reparameterize(mu, logvar)
            z = fn.normalize(z, dim=1)
            z_list.append(z)

        return z_list, mu_list, logvar_list, cates_k_list, items, prototypes

    def decode(self, z_list, cates_k_list, items, need_prob_k=False):
        probs = None
        prob_k_list = []

        for k in range(self.num_prototypes):
            # decoder
            z_k = z_list[k]
            cates_k = cates_k_list[k]
            logits_k = z_k.matmul(items.transpose(0, 1)) / self.tau
            probs_k = torch.exp(logits_k)
            probs_k = probs_k * cates_k
            if need_prob_k:
                prob_k_list.append(probs_k)
            probs = (probs_k if probs is None else (probs + probs_k))

        return probs, (prob_k_list if need_prob_k else None)

    def forward(self, input_rating, need_prob_k=False, need_z=False):
        z_list, mu_list, logvar_list, cates_k_list, items, prototypes = self.encode(input_rating)

        probs, prob_k_list = self.decode(z_list, cates_k_list, items, need_prob_k)
        logits = torch.log(probs)

        outputs = [logits, mu_list, logvar_list]
        if need_prob_k:
            outputs.append(prob_k_list)
        if need_z:
            outputs.append(z_list)
        return tuple(outputs)

    def calculate_kl_loss(self, logvar):
        kl_loss = None
        for i in range(self.num_prototypes):
            kl_ = -0.5 * torch.mean(torch.sum(1 + logvar[i] - logvar[i].exp(), dim=1))
            kl_loss = kl_ if kl_loss is None else kl_loss + kl_
        return kl_loss

    @staticmethod
    def calculate_ce_loss(input_rating, logits):
        ce_loss = -(fn.log_softmax(logits, 1) * input_rating).sum(dim=1).mean()
        return ce_loss

    def calculate_loss(self, input_rating, need_z=False):
        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1. * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        logits, mu, logvar, z_list = self.forward(input_rating, need_z=True)
        kl_loss = self.calculate_kl_loss(logvar)
        ce_loss = self.calculate_ce_loss(input_rating, logits)

        if self.reg_weights[0] != 0 or self.reg_weights[1] != 0:
            if need_z:
                return ce_loss + kl_loss * anneal + self.reg_loss(), z_list
            return ce_loss + kl_loss * anneal + self.reg_loss()
        if need_z:
            return ce_loss + kl_loss * anneal, z_list
        return ce_loss + kl_loss * anneal

    def predict(self, input_rating):
        prediction, _, _, = self.forward(input_rating=input_rating,
                                         need_prob_k=False)
        prediction[input_rating.nonzero(as_tuple=True)] = -np.inf
        return prediction

    def reg_loss(self):
        reg_1, reg_2 = self.reg_weights[: 2]
        loss_1 = reg_1 * self.item_embeddings.weight.norm(2)
        loss_2 = reg_2 * self.prototype_embeddings.weight.norm(2)
        loss_3 = 0

        for name, param in self.encoder.named_parameters():
            if name.endswith('weight'):
                loss_3 = loss_3 + reg_2 * param.norm(2)

        return loss_1 + loss_2 + loss_3

    def predict_per_prototype_outputs(self, input_rating):
        _, _, _, prob_k_list = self.forward(input_rating,
                                            need_prob_k=True)
        return prob_k_list

    def get_param_names(self):
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def get_params(self):
        return self.parameters()

    def save(self, out_file):
        config_dict = {
            'item_adj': self.item_adj,
            'num_items': self.num_items,
            'layers': self.layers,
            'emb_size': self.emb_size,
            'dropout': self.dropout,
            'num_prototypes': self.num_prototypes,
            'tau': self.tau,
            'nogb': self.nogb,
            'anneal_cap': self.anneal_cap,
            'std': self.std,
            'reg_weights': self.reg_weights,
            'total_anneal_steps': self.total_anneal_steps,
            'state_dict': self.state_dict()
        }
        torch.save(config_dict, out_file)
        print(f'Save model to {out_file}')

    @classmethod
    def load(cls, model_file):
        config_dict = torch.load(model_file)
        model = MacridVAE(item_adj=config_dict['item_adj'],
                          num_items=config_dict['num_items'],
                          layers=config_dict['layers'],
                          emb_size=config_dict['emb_size'],
                          dropout=config_dict['dropout'],
                          num_prototypes=config_dict['num_prototypes'],
                          tau=config_dict['tau'],
                          nogb=config_dict['nogb'],
                          anneal_cap=config_dict['anneal_cap'],
                          std=config_dict['std'],
                          reg_weights=config_dict['reg_weights'],
                          total_anneal_steps=config_dict['total_anneal_steps'])
        model.load_state_dict(config_dict['state_dict'])
        return model


class DGVAE(nn.Module):
    def __init__(self,
                 item_adj,
                 num_items: int,
                 num_words: int,
                 layers: list,
                 emb_size: int,
                 dropout: float,
                 num_prototypes: int,
                 tau: float = 1.,
                 nogb: bool = False,
                 rating_anneal_cap: float = 0.2,
                 text_anneal_cap: float = 2,
                 std: float = 1.,
                 text_std: float = 1.,
                 reg_weights: List[float] = None,
                 rating_total_anneal_steps: int = 20000,
                 text_total_anneal_steps: int = 20000,
                 lambda_text: float = 1.,
                 lambda_reg: float = 0.1):
        super(DGVAE, self).__init__()
        if reg_weights is None:
            reg_weights = [0.0, 0.0]
        self.item_adj = item_adj
        self.num_items = num_items
        self.num_words = num_words
        self.layers = layers
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_prototypes = num_prototypes
        self.tau = tau
        self.nogb = nogb
        self.rating_anneal_cap = rating_anneal_cap
        self.text_anneal_cap = text_anneal_cap
        self.std = std
        self.text_std = text_std
        self.reg_weights = reg_weights
        self.update = 0
        self.rating_total_anneal_steps = rating_total_anneal_steps
        self.text_total_anneal_steps = text_total_anneal_steps
        self.lambda_text = lambda_text
        self.lambda_reg = lambda_reg

        self.cf_network = MacridVAE(item_adj=self.item_adj,
                                    num_items=self.num_items,
                                    layers=self.layers,
                                    emb_size=self.emb_size,
                                    dropout=self.dropout,
                                    num_prototypes=self.num_prototypes,
                                    tau=self.tau,
                                    nogb=self.nogb,
                                    anneal_cap=self.rating_anneal_cap,
                                    std=self.std,
                                    reg_weights=self.reg_weights,
                                    total_anneal_steps=self.rating_total_anneal_steps)

        self.text_network = MacridVAE(item_adj=None,
                                      num_items=self.num_words,
                                      layers=self.layers,
                                      emb_size=self.emb_size,
                                      dropout=self.dropout,
                                      num_prototypes=self.num_prototypes,
                                      tau=self.tau,
                                      nogb=self.nogb,
                                      anneal_cap=self.text_anneal_cap,
                                      std=self.text_std,
                                      reg_weights=self.reg_weights,
                                      total_anneal_steps=self.text_total_anneal_steps)

    def forward(self, input_rating, input_text, need_prob_k=False):
        cf_z_list, cf_mu_list, cf_logvar_list, cf_cates_k_list, cf_items, cf_prototypes = \
            self.cf_network.encode(input_rating)
        text_z_list, text_mu_list, text_logvar_list, text_cates_k_list, text_items, text_prototypes = \
            self.text_network.encode(input_text)
        assert len(cf_z_list) == len(text_z_list)
        cf_z_tensor = torch.stack(cf_z_list, dim=0).permute(1, 0, 2)
        text_z_tensor = torch.stack(text_z_list, dim=0).permute(1, 0, 2)
        text_z_tensor_given_cf = self.attention_layer(q_mat=cf_z_tensor,
                                                      k_mat=text_z_tensor,
                                                      v_mat=text_z_tensor)
        text_z_list_given_cf = self.split(text_z_tensor_given_cf)
        cf_z_tensor_given_text = self.attention_layer(q_mat=text_z_tensor,
                                                      k_mat=cf_z_tensor,
                                                      v_mat=cf_z_tensor)

        cf_z_list_given_text = self.split(cf_z_tensor_given_text)

        reg_term = self.mutual_information_reg(cf_z_list_given_text, text_z_list_given_cf)
        z_list = [cf_z_list_given_text[i] + text_z_list_given_cf[i] for i in range(self.num_prototypes)]
        cf_probs, cf_prob_k_list = \
            self.cf_network.decode(z_list, cf_cates_k_list, cf_items, need_prob_k)
        cf_logits = torch.log(cf_probs)

        text_probs, text_prob_k_list = \
            self.text_network.decode(z_list, text_cates_k_list, text_items, need_prob_k)
        text_logits = torch.log(text_probs)
        if need_prob_k:
            return cf_logits, text_logits, reg_term, cf_logvar_list, text_logvar_list, \
                   cf_prob_k_list, text_prob_k_list
        return cf_logits, text_logits, reg_term, cf_logvar_list, text_logvar_list

    def mutual_information_reg(self, list_a: List[Tensor], list_b: List[Tensor]):
        assert len(list_a) == len(list_b)
        assert len(list_a) == self.num_prototypes
        mi_reg = None
        for i in range(self.num_prototypes):
            x_pos = list_a[i]  # [bs, dim]
            y_pos = list_b[i]
            f_values = [(x_pos * list_b[j]).sum(dim=-1) for j in range(self.num_prototypes) if j != i]
            soft_plus_f_values = [fn.softplus(value).unsqueeze(dim=-1) for value in f_values]
            e_n = torch.cat(soft_plus_f_values, dim=-1).mean(dim=-1)
            e_p = -fn.softplus(-(x_pos * y_pos).sum(dim=-1))
            loss_i = (e_n - e_p).mean()
            mi_reg = loss_i if mi_reg is None else loss_i + mi_reg
        return mi_reg / self.num_prototypes

    def attention_layer(self, q_mat, k_mat, v_mat):
        p_1 = (q_mat.bmm(k_mat.transpose(1, 2)))
        p_1 = p_1.tanh()
        p_2 = -(q_mat.unsqueeze(dim=2) - k_mat.unsqueeze(dim=1)).abs().sum(dim=-1)
        p_2 = (p_2 - p_2.mean(dim=-1, keepdim=True)).sigmoid()
        return (p_1 * p_2).bmm(v_mat)

    @staticmethod
    def split(x: Tensor):
        """
        :param x: Tensor of size [m, n, p]
        :return: list of n elements, each element is a tensor of size [m, p]
        """
        x = x.permute(1, 0, 2)
        tensor_list = x.split(split_size=1, dim=0)
        return [t.squeeze(dim=0) for t in tensor_list]

    def calculate_loss(self, input_rating, input_text):
        cf_logits, text_logits, reg_term, cf_logvar_list, text_logvar_list = \
            self.forward(input_rating, input_text)

        self.update += 1

        if self.rating_total_anneal_steps > 0:
            rating_anneal = min(self.rating_anneal_cap, 1. * self.update / self.rating_total_anneal_steps)
        else:
            rating_anneal = self.rating_anneal_cap

        if self.text_total_anneal_steps > 0:
            text_anneal = min(self.text_anneal_cap, 1. * self.update / self.text_total_anneal_steps)
        else:
            text_anneal = self.text_anneal_cap

        cf_kl_loss = self.cf_network.calculate_kl_loss(cf_logvar_list)
        cf_ce_loss = self.cf_network.calculate_ce_loss(input_rating, cf_logits)
        if self.cf_network.reg_weights[0] != 0 or self.cf_network.reg_weights[1] != 0:
            cf_loss = cf_ce_loss + cf_kl_loss * rating_anneal + self.cf_network.reg_loss()
        else:
            cf_loss = cf_ce_loss + cf_kl_loss * rating_anneal

        text_kl_loss = self.text_network.calculate_kl_loss(text_logvar_list)
        text_ce_loss = self.text_network.calculate_ce_loss(input_text, text_logits)

        if self.text_network.reg_weights[0] != 0 or self.text_network.reg_weights[1] != 0:
            text_loss = text_ce_loss + text_kl_loss * text_anneal + self.text_network.reg_loss()
        else:
            text_loss = text_ce_loss + text_kl_loss * text_anneal

        final_loss = cf_loss + self.lambda_text * text_loss
        if self.lambda_reg > 0:
            final_loss = final_loss + self.lambda_reg * reg_term
        return final_loss

    def predict(self, input_rating, input_text):
        rating_prediction, text_prediction, _, _, _ = self.forward(input_rating, input_text)
        rating_prediction[input_rating.nonzero(as_tuple=True)] = -np.Inf
        return rating_prediction, text_prediction

    def predict_per_prototype_output(self, input_rating, input_text):
        _, _, _, _, _, rating_prob_k_list, text_prob_k_list = self.forward(input_rating, input_text, need_prob_k=True)
        return rating_prob_k_list, text_prob_k_list

    def get_param_names(self):
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def get_params(self):
        return self.parameters()

    def save(self, out_file):
        config_dict = {
            'item_adj': self.item_adj,
            'num_items': self.num_items,
            'num_words': self.num_words,
            'layers': self.layers,
            'emb_size': self.emb_size,
            'dropout': self.dropout,
            'num_prototypes': self.num_prototypes,
            'tau': self.tau,
            'nogb': self.nogb,
            'text_anneal_cap': self.text_anneal_cap,
            'rating_anneal_cap': self.rating_anneal_cap,
            'std': self.std,
            'text_std': self.text_std,
            'reg_weights': self.reg_weights,
            'rating_total_anneal_steps': self.rating_total_anneal_steps,
            'text_total_anneal_steps': self.text_total_anneal_steps,
            'lambda_text': self.lambda_text,
            'lambda_reg': self.lambda_reg,
            'state_dict': self.state_dict()
        }
        torch.save(config_dict, out_file)
        print(f'Save model to {out_file}')

    @classmethod
    def load(cls, model_file):
        config_dict = torch.load(model_file)
        model = DGVAE(item_adj=config_dict['item_adj'],
                       num_items=config_dict['num_items'],
                       num_words=config_dict['num_words'],
                       layers=config_dict['layers'],
                       emb_size=config_dict['emb_size'],
                       dropout=config_dict['dropout'],
                       num_prototypes=config_dict['num_prototypes'],
                       tau=config_dict['tau'],
                       nogb=config_dict['nogb'],
                       rating_anneal_cap=config_dict['rating_anneal_cap'],
                       text_anneal_cap=config_dict['text_anneal_cap'],
                       std=config_dict['std'],
                       text_std=config_dict['text_std'],
                       reg_weights=config_dict['reg_weights'],
                       rating_total_anneal_steps=config_dict['rating_total_anneal_steps'],
                       text_total_anneal_steps=config_dict['text_total_anneal_steps'],
                       lambda_text=config_dict['lambda_text'],
                       lambda_reg=config_dict['lambda_reg'])
        model.load_state_dict(config_dict['state_dict'])
        return model
