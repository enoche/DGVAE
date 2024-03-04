from typing import Union

import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from torch.utils.data import DataLoader, Dataset
from utils import load_json


class CSRDataset(Dataset):
    def __init__(self, x: csr_matrix, y: Union[csr_matrix, np.array] = None, z: Union[csr_matrix, np.array] = None,
                 t: np.array = None):
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        if self.y is not None:
            assert self.x.shape[0] == self.y.shape[0], print(self.x.shape, self.y.shape)
        if self.z is not None:
            assert self.x.shape[0] == self.z.shape[0], print(self.x.shape, self.z.shape)
        if self.t is not None:
            assert self.x.shape[0] == self.t.shape[0], print(self.x.shape, self.t.shape)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        batch_x = np.array(self.x[idx].todense())[0]
        output = [idx, batch_x]
        if self.y is not None:
            if isinstance(self.y, csr_matrix):
                batch_y = np.array(self.y[idx].todense())[0]
            else:
                batch_y = self.y[idx]
            output.append(batch_y)
        if self.z is not None:
            if isinstance(self.z, csr_matrix):
                batch_z = np.array(self.z[idx].todense())[0]
            else:
                batch_z = self.z[idx]
            output.append(batch_z)
        if self.t is not None:
            batch_t = self.t[idx]
            output.append(batch_t)
        return tuple(output)


def read_rating_file(file_path, binary=False):
    data = pd.read_csv(file_path).values
    inputs = data[:, 0: 2].astype(int)  # user_id, item_id
    targets = data[:, 2].astype(int)  # rating
    if binary:
        targets = (targets > 0).astype(int)
    return inputs, targets


def get_npz_file(file_path):
    matrix = load_npz(file=file_path)
    return matrix


def get_sparse_word_matrix(args):
    dataset = args.dataset
    ddir = args.dataset_dir
    file_path = f'{ddir}/{dataset}/{dataset}.user.words.tfidf.npz'
    return get_npz_file(file_path)


def get_data(args, binary=True, num_workers=1, train_shuffle=True):
    dataset = args.dataset
    ddir = args.dataset_dir
    batch_size = args.batch_size

    train_path = f'{ddir}/{dataset}/{dataset}.train.ratings'
    val_path = f'{ddir}/{dataset}/{dataset}.valid.ratings'
    test_path = f'{ddir}/{dataset}/{dataset}.test.ratings'

    train_inputs, train_targets = read_rating_file(train_path, binary=binary)
    val_inputs, val_targets = read_rating_file(val_path, binary=binary)
    test_inputs, test_targets = read_rating_file(test_path, binary=binary)

    word_vocab = load_json(f'{ddir}/{dataset}/{dataset}.vocab')
    user_vocab = np.load(f'{ddir}/{dataset}/{dataset}.uid.npy')
    item_vocab = np.load(f'{ddir}/{dataset}/{dataset}.iid.npy')

    data_stats = {
        'num_users': len(user_vocab),
        'num_items': len(item_vocab),
        'num_ratings': train_targets.shape[0],
        'num_valid_users': len(set(val_inputs[:, 0])),
        'num_valid_items': len(set(val_inputs[:, 1])),
        'num_valid_ratings': val_targets.shape[0],
        'num_test_users': len(set(test_inputs[:, 0])),
        'num_test_items': len(set(test_inputs[:, 1])),
        'num_test_ratings': test_targets.shape[0],
        'word_vocab': word_vocab,
        'num_words': len(word_vocab)
    }

    text_csr_matrix = get_sparse_word_matrix(args)

    train_uids = train_inputs[:, 0]
    train_iids = train_inputs[:, 1]
    train_csr_matrix = csr_matrix((train_targets, (train_uids, train_iids)),
                                  shape=(len(user_vocab), len(item_vocab)))

    train_loader = DataLoader(CSRDataset(x=train_csr_matrix,
                                         y=text_csr_matrix),
                              batch_size=batch_size,
                              shuffle=train_shuffle,
                              drop_last=False,
                              pin_memory=True,
                              num_workers=num_workers)

    valid_uids = val_inputs[:, 0]
    valid_iids = val_inputs[:, 1]
    valid_csr_matrix = csr_matrix((val_targets, (valid_uids, valid_iids)),
                                  shape=(len(user_vocab), len(item_vocab)))
    train_valid_loader = DataLoader(CSRDataset(x=train_csr_matrix,
                                               y=valid_csr_matrix,
                                               z=text_csr_matrix),
                                    batch_size=batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    pin_memory=True,
                                    num_workers=num_workers)

    test_uids = test_inputs[:, 0]
    test_iids = test_inputs[:, 1]
    test_csr_matrix = csr_matrix((test_targets, (test_uids, test_iids)),
                                 shape=(len(user_vocab), len(item_vocab)))
    train_test_loader = DataLoader(CSRDataset(x=train_csr_matrix,
                                              y=test_csr_matrix,
                                              z=text_csr_matrix),
                                   batch_size=batch_size,
                                   shuffle=False,
                                   drop_last=False,
                                   pin_memory=True,
                                   num_workers=num_workers)
    return (train_loader, train_valid_loader, train_test_loader), data_stats


def compute_normalized_laplacian(indices, adj_size):
    adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
    row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
    r_inv_sqrt = torch.pow(row_sum, -0.5)
    rows_inv_sqrt = r_inv_sqrt[indices[0]]
    cols_inv_sqrt = r_inv_sqrt[indices[1]]
    values = rows_inv_sqrt * cols_inv_sqrt
    return torch.sparse.FloatTensor(indices, values, adj_size)


def get_knn_adj_mat(knn_k, mm_embeddings):
    context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    _, knn_ind = torch.topk(sim, knn_k, dim=-1)
    adj_size = sim.size()
    del sim
    # construct sparse adj
    indices0 = torch.arange(knn_ind.shape[0])
    indices0 = torch.unsqueeze(indices0, 1)
    indices0 = indices0.expand(-1, knn_k)
    indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
    # norm
    return compute_normalized_laplacian(indices, adj_size)


def build_item_adj(args):
    dataset = args.dataset
    ddir = args.dataset_dir
    knn_k = args.knn_k
    txt_weight = args.txt_weight
    t_feat_file_path = f'{ddir}/{dataset}/text_feat.npy'
    t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor)
    v_feat_file_path = f'{ddir}/{dataset}/image_feat.npy'
    v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor)
    # sim matrix
    text_adj = get_knn_adj_mat(knn_k, t_feat)
    image_adj = get_knn_adj_mat(knn_k, v_feat)
    # fusion
    return (1.0 - txt_weight) * image_adj + txt_weight * text_adj

