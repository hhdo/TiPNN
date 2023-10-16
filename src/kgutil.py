""" Knowledge graph dataset for Relational-GCN
Code adapted from authors' implementation of Relational-GCN
https://github.com/tkipf/relational-gcn
https://github.com/MichSchli/RelationPrediction
"""

from __future__ import absolute_import
from __future__ import print_function

import gzip
import os
from collections import Counter

import numpy as np
import scipy.sparse as sp
from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url
import sys

_downlaod_prefix = _get_dgl_url('dataset/')

class RGCNLinkDataset(object):
    """RGCN link prediction dataset

    The dataset contains a graph depicting the connectivity of a knowledge
    base. Currently, the knowledge bases from the
    `RGCN paper <https://arxiv.org/pdf/1703.06103.pdf>`_ supported are
    FB15k-237, FB15k, wn18

    The original knowledge base is stored as an RDF file, and this class will
    download and parse the RDF file, and performs preprocessing.

    An object of this class has 5 member attributes needed for link
    prediction:

    num_nodes: int
        number of entities of knowledge base
    num_rels: int
        number of relations (including reverse relation) of knowledge base
    train: numpy.array
        all relation triplets (src, rel, dst) for training
    valid: numpy.array
        all relation triplets (src, rel, dst) for validation
    test: numpy.array
        all relation triplets (src, rel, dst) for testing

    Usually, user don't need to directly use this class. Instead, DGL provides
    wrapper function to load data (see example below).

    Examples
    --------
    Load FB15k-237 dataset

    >>> from dgl.contrib.data import load_data
    >>> data = load_data(dataset='FB15k-237')

    """
    def __init__(self, name, dir=None):
        self.name = name
        if dir:
            self.dir = dir
            self.dir = os.path.join(self.dir, self.name)

        else:
            self.dir = get_download_dir()
            tgz_path = os.path.join(self.dir, '{}.tar.gz'.format(self.name))
            download(_downlaod_prefix + '{}.tgz'.format(self.name), tgz_path)
            self.dir = os.path.join(self.dir, self.name)
            extract_archive(tgz_path, self.dir)
        # print(self.dir)


    def load(self, load_time=True):
        entity_path = os.path.join(self.dir, 'entity2id.txt')
        relation_path = os.path.join(self.dir, 'relation2id.txt')
        train_path = os.path.join(self.dir, 'train.txt')
        valid_path = os.path.join(self.dir, 'valid.txt')
        test_path = os.path.join(self.dir, 'test.txt')
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        self.train = _read_triplets_as_list(train_path, entity_dict, relation_dict, load_time)
        self.valid = _read_triplets_as_list(valid_path, entity_dict, relation_dict, load_time)
        self.test = _read_triplets_as_list(test_path, entity_dict, relation_dict, load_time)
        self.num_nodes = len(entity_dict)
        self.num_rels = len(relation_dict)
        self.relation_dict = relation_dict
        self.entity_dict = entity_dict
        # print("# Sanity Check:  entities: {}".format(self.num_nodes))
        # print("# Sanity Check:  relations: {}".format(self.num_rels))
        # print("# Sanity Check:  edges: {}".format(len(self.train)))


def load_link(dataset):
    data = RGCNLinkDataset(dataset)
    data.load()
    return data


def load_from_local(dir, dataset):
    data = RGCNLinkDataset(dataset, dir)
    # if "-d" in dataset:
    #     data.load(load_time=True)
    # else:
    #
    data.load()
    return data


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors."""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(sp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _bfs_relational(adj, roots):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        yield next_lvl

        current_lvl = set.union(next_lvl)


def to_unicode(input):
    # FIXME (lingfan): not sure about python 2 and 3 str compatibility
    return str(input)
    """ lingfan: comment out for now
    if isinstance(input, unicode):
        return input
    elif isinstance(input, str):
        return input.decode('utf-8', errors='replace')
    return str(input).decode('utf-8', errors='replace')
    """


def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[int(line[1])] = line[0]
    return d


def _read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line


def _read_triplets_as_list(filename, entity_dict, relation_dict, load_time):
    l = []
    for triplet in _read_triplets(filename):
        s = int(triplet[0])
        r = int(triplet[1])
        o = int(triplet[2])
        if load_time:
            st = int(triplet[3])
            # et = int(triplet[4])
            # l.append([s, r, o, st, et])
            l.append([s, r, o, st])
        else:
            l.append([s, r, o])
    return l
