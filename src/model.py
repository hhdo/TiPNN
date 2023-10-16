from collections.abc import Sequence
import torch
from torch import nn, autograd
from torch.nn import functional as F
import layers
from torch_scatter import scatter_add

class TiPNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_nodes, num_relation, message_func="distmult", aggregate_func="pna",
                 short_cut=False, layer_norm=False, activation="relu", num_mlp_layer=2,
                 history_len=10, time_encoding=True, time_encoding_independent=True):
        super(TiPNN,self).__init__()

        self.dims = [input_dim] + list(hidden_dims)
        self.num_nodes = num_nodes
        self.num_relation = num_relation *2 # reverse rel type should be added
        self.short_cut = short_cut  # whether to use residual connections between layers
        
        self.history_len = history_len

        # Learnable Relation Representation
        self.query = nn.Embedding(self.num_relation, input_dim)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1): # num of hidden layers
            self.layers.append(layers.TemporalPathAgg(self.dims[i], self.dims[i + 1], self.num_relation,
                                                                self.dims[0], message_func, aggregate_func, layer_norm,
                                                                activation, time_encoding, time_encoding_independent))

        self.feature_dim = hidden_dims[-1] + input_dim

        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(self.feature_dim, self.feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(self.feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation // 2)
        return new_h_index, new_t_index, new_r_index

    def pathProc(self, history_graph, h_index, query, separate_grad=False):
        batch_size = len(h_index)
        
        index = h_index.unsqueeze(-1).expand_as(query)

        # initialize all pairs states as zeros in memory
        initial_stat = torch.zeros(batch_size, history_graph.num_nodes(), self.dims[0], device=h_index.device)
        
        # Temporal Path Initialization
        initial_stat.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        
        size = (history_graph.num_nodes(), history_graph.num_nodes())
        edge_weight = torch.ones(history_graph.num_edges(), device=h_index.device)

        edge_weights = []
        layer_input = initial_stat
        
        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()
            # layers iteration
            hidden = layer(layer_input, query, initial_stat, 
                           torch.stack(history_graph.edges()), history_graph.edata['type'], history_graph.edata['time'],
                           size, edge_weight = edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # shortcut setting
                hidden = hidden + layer_input
            edge_weights.append(edge_weight)
            layer_input = hidden

        return {
            "node_feature": hidden,
            "edge_weights": edge_weights,
        }

    def forward(self, history_graph, query_triple):
        h_index, r_index, t_index = query_triple.unbind(-1)
        shape = h_index.shape
        batch_size = shape[0]

        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # initialize queries (relation types of the given triples)
        query = self.query(r_index[:, 0])

        # Query-aware Temporal Path Processing
        output = self.pathProc(history_graph, h_index[:, 0], query)
        feature = output["node_feature"]

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, history_graph.num_nodes(), -1)
        final_feature = torch.cat([feature, node_query], dim=-1)
     
        index = t_index.unsqueeze(-1).expand(-1, -1, final_feature.shape[-1])
        # extract representations of tail entities
        feature_t = final_feature.gather(1, index)

        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature_t).squeeze(-1)

        return score.view(shape)


    def get_loss(self, args, pred):
        
        target = torch.zeros_like(pred)
        target[:, 0] = 1
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        neg_weight = torch.ones_like(pred)
        if args.adversarial_temperature > 0:
            with torch.no_grad():
                neg_weight[:, 1:] = F.softmax(pred[:, 1:] / args.adversarial_temperature, dim=-1)
        else:
            neg_weight[:, 1:] = 1 / args.negative_num
        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss = loss.mean()

        tmp = torch.mm(self.query.weight, self.query.weight.permute(1, 0))
        orthogonal_regularizer = torch.norm(tmp - 1 * torch.diag(torch.ones(self.num_relation, device=pred.device)), 2)

        loss = loss + orthogonal_regularizer
        return loss
    
    def visualize(self, data, batch):
        # data: history graph
        # batch: the triplet to be visualized
        assert batch.shape == (1, 3)
        h_index, r_index, t_index = batch.unbind(-1)

        query = self.query(r_index)

        output = self.pathProc(data, h_index, query, separate_grad=True)
        feature = output["node_feature"]
        edge_weights = output["edge_weights"]

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes(), -1)
        final_feature = torch.cat([feature, node_query], dim=-1)

        index = t_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, final_feature.shape[-1])
        feature_t = final_feature.gather(1, index).squeeze(0)
        score = self.mlp(feature_t).squeeze(-1)

        edge_grads = autograd.grad(score, edge_weights)
        graph_simple_data = {
                'num_nodes': data.num_nodes(),
                'edge_index': torch.stack(data.edges()),
                'edge_type': data.edata['type'],
                'edge_time': data.edata['time']
        }
        distances, back_edges = self.beam_search_distance(graph_simple_data, edge_grads, h_index, t_index, num_beam=10)
        paths, weights = self.topk_average_length(distances, back_edges, t_index, k=8)

        return paths, weights
    
    @torch.no_grad()
    def beam_search_distance(self, data, edge_grads, h_index, t_index, num_beam=10):
        num_nodes = data['num_nodes']
        input = torch.full((num_nodes, num_beam), float("-inf"), device=h_index.device)
        input[h_index, 0] = 0
        edge_mask = data['edge_index'][0, :] != t_index

        distances = []
        back_edges = []

        for edge_grad in edge_grads: 
            node_in, node_out = data['edge_index'][:, edge_mask] 
            relation = data['edge_type'][edge_mask]
            rel_ts = data['edge_time'][edge_mask]
            edge_grad = edge_grad[edge_mask]
            message = input[node_in] + edge_grad.unsqueeze(-1) # (num_edges, num_beam)
            
            msg_source = torch.stack([node_in, node_out, relation, rel_ts], dim=-1).unsqueeze(1).expand(-1, num_beam, -1)
            is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & \
                           (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            is_duplicate = is_duplicate.float() - \
                           torch.arange(num_beam, dtype=torch.float, device=message.device) / (num_beam + 1)

            prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
            msg_source = torch.cat([msg_source, prev_rank], dim=-1) # (num_edges, num_beam, 4)

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)
            message = message[order].flatten() # (num_edges * num_beam)
            msg_source = msg_source[order].flatten(0, -2) # (num_edges * num_beam, 4)
            size = node_out.bincount(minlength=num_nodes)
            msg2out = size_to_index(size[node_out_set] * num_beam)
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool, device=message.device), is_duplicate])
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = msg2out.bincount(minlength=len(node_out_set))

            if not torch.isinf(message).all():
                distance, rel_index = scatter_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                back_edge = msg_source[abs_index]
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 5)
                distance = scatter_add(distance, node_out_set, dim=0, dim_size=num_nodes)
                back_edge = scatter_add(back_edge, node_out_set, dim=0, dim_size=num_nodes)
            else:
                distance = torch.full((num_nodes, num_beam), float("-inf"), device=message.device)
                back_edge = torch.zeros(num_nodes, num_beam, 5, dtype=torch.long, device=message.device)

            distances.append(distance)
            back_edges.append(back_edge)
            input = distance

        return distances, back_edges
    
    def topk_average_length(self, distances, back_edges, t_index, k=10):
        paths = []
        average_lengths = []

        for i in range(len(distances)):
            distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
            back_edge = back_edges[i][t_index].flatten(0, -2)[order]
            for d, (h, t, r, ts, prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
                if d == float("-inf"):
                    break
                path = [(h, t, r, ts)]
                for j in range(i - 1, -1, -1):
                    h, t, r, ts, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r, ts))
                paths.append(path[::-1])
                average_lengths.append(d / len(path))

        if paths:
            average_lengths, paths = zip(*sorted(zip(average_lengths, paths), reverse=True)[:k])

        return paths, average_lengths
    

def size_to_index(size):
    range = torch.arange(len(size), device=size.device)
    index2sample = range.repeat_interleave(size)
    return index2sample


def multi_slice_mask(starts, ends, length):
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    slices = torch.cat([starts, ends])
    mask = scatter_add(values, slices, dim=0, dim_size=length + 1)[:-1]
    mask = mask.cumsum(0).bool()
    return mask


def scatter_extend(data, size, input, input_size):
    new_size = size + input_size
    new_cum_size = new_size.cumsum(0)
    new_data = torch.zeros(new_cum_size[-1], *data.shape[1:], dtype=data.dtype, device=data.device)
    starts = new_cum_size - new_size
    ends = starts + size
    index = multi_slice_mask(starts, ends, new_cum_size[-1])
    new_data[index] = data
    new_data[~index] = input
    return new_data, new_size


def scatter_topk(input, size, k, largest=True):
    index2graph = size_to_index(size)
    index2graph = index2graph.view([-1] + [1] * (input.ndim - 1))

    mask = ~torch.isinf(input)
    max = input[mask].max().item()
    min = input[mask].min().item()
    safe_input = input.clamp(2 * min - max, 2 * max - min)
    offset = (max - min) * 4
    if largest:
        offset = -offset
    input_ext = safe_input + offset * index2graph
    index_ext = input_ext.argsort(dim=0, descending=largest)
    num_actual = size.clamp(max=k)
    num_padding = k - num_actual
    starts = size.cumsum(0) - size
    ends = starts + num_actual
    mask = multi_slice_mask(starts, ends, len(index_ext)).nonzero().flatten()

    if (num_padding > 0).any():
        padding = ends - 1
        padding2graph = size_to_index(num_padding)
        mask = scatter_extend(mask, num_actual, padding[padding2graph], num_padding)[0]

    index = index_ext[mask] # (N * k, ...)
    value = input.gather(0, index)
    if isinstance(k, torch.Tensor) and k.shape == size.shape:
        value = value.view(-1, *input.shape[1:])
        index = index.view(-1, *input.shape[1:])
        index = index - (size.cumsum(0) - size).repeat_interleave(k).view([-1] + [1] * (index.ndim - 1))
    else:
        value = value.view(-1, k, *input.shape[1:])
        index = index.view(-1, k, *input.shape[1:])
        index = index - (size.cumsum(0) - size).view([-1] + [1] * (index.ndim - 1))

    return value, index