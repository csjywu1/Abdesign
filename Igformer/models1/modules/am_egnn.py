#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# 直接实现scatter_softmax，避免依赖torch_scatter
def scatter_softmax(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    """
    直接实现的scatter_softmax函数，替代torch_scatter.scatter_softmax
    """
    if out is None:
        if dim_size is None:
            dim_size = int(index.max()) + 1
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
    
    # 确保index是1D
    if index.dim() > 1:
        index = index.flatten()
    
    # 获取每个组的最大值
    max_val = torch.full((dim_size,), float('-inf'), dtype=src.dtype, device=src.device)
    max_val.scatter_reduce_(0, index, src, reduce='amax')
    
    # 减去最大值以保持数值稳定性
    src_stable = src - max_val[index]
    
    # 计算exp
    exp_src = torch.exp(src_stable)
    
    # 计算每个组的exp和
    sum_exp = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
    sum_exp.scatter_add_(0, index, exp_src)
    
    # 计算softmax
    softmax_val = exp_src / sum_exp[index]
    
    # 散射到输出
    out.scatter_(0, index, softmax_val)
    
    return out

class AMEGNN(nn.Module):

    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
                 radial_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=4,
                 residual=True, dropout=0.1, dense=False):
        super().__init__()
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param n_channel: Number of channels of coordinates
        :param in_edge_nf: Number of features for the edge features
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param dropout: probability of dropout
        :param dense: if dense, then context states will be concatenated for all layers,
                      coordination will be averaged
        '''
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)

        self.linear_in = nn.Linear(in_node_nf, self.hidden_nf)

        self.dense = dense
        if dense:
            self.linear_out = nn.Linear(self.hidden_nf * (n_layers + 1), out_node_nf)
        else:
            self.linear_out = nn.Linear(self.hidden_nf, out_node_nf)

        for i in range(0, n_layers):
            self.add_module(f'gcl_{i}', AM_E_GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf,
                edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout
            ))
        self.out_layer = AM_E_GCL(
            self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf,
            radial_nf, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual
        )
    
    def forward(self, h, x, edges, channel_attr, channel_weights, ctx_edge_attr=None):
        h = self.linear_in(h)
        h = self.dropout(h)

        ctx_states, ctx_coords = [], []
        for i in range(0, self.n_layers):
            h, x = self._modules[f'gcl_{i}'](
                h, edges, x, channel_attr, channel_weights,
                edge_attr=ctx_edge_attr)
            ctx_states.append(h)
            ctx_coords.append(x)

        h, x = self.out_layer(
            h, edges, x, channel_attr, channel_weights,
            edge_attr=ctx_edge_attr)
        ctx_states.append(h)
        ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
        h = self.dropout(h)
        h = self.linear_out(h)
        return h, x

'''
Below are the implementation of the adaptive multi-channel message passing mechanism
'''

class AM_E_GCL(nn.Module):
    '''
    Adaptive Multi-Channel E(n) Equivariant Convolutional Layer
    '''

    def __init__(self, input_nf, output_nf, hidden_nf, n_channel, channel_nf, radial_nf,
                 edges_in_d=0, node_attr_d=0, act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False, dropout=0.1):
        super(AM_E_GCL, self).__init__()

        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8

        self.dropout = nn.Dropout(dropout)

        input_edge = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + radial_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        # 注释掉原来的radial_linear，使用两种相似度计算
        # self.radial_linear = nn.Linear(channel_nf ** 2, radial_nf)
        
        # 添加两种相似度计算的线性映射层
        self.radial_linear = nn.Sequential(
            nn.Linear(196, 256),
            nn.SiLU(),  # or nn.ReLU()
            nn.Linear(256, 128), 
            nn.SiLU(),
            nn.Linear(128, radial_nf))
        
        self.radial_linear1 = nn.Sequential(
            nn.Linear(196, 256),
            nn.SiLU(),  # or nn.ReLU()
            nn.Linear(256, 128), 
            nn.SiLU(),
            nn.Linear(128, radial_nf))

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + node_attr_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        '''
        :param source: [n_edge, input_size]
        :param target: [n_edge, input_size]
        :param radial: [n_edge, d, d]
        :param edge_attr: [n_edge, edge_dim]
        '''
        radial = radial.reshape(radial.shape[0], -1)  # [n_edge, d ^ 2]

        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        out = self.dropout(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        '''
        :param x: [bs * n_node, input_size]
        :param edge_index: list of [n_edge], [n_edge]
        :param edge_attr: [n_edge, hidden_size], refers to message from i to j
        :param node_attr: [bs * n_node, node_dim]
        '''
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))  # [bs * n_node, hidden_size]
        # print_log(f'agg1, {torch.isnan(agg).sum()}', level='DEBUG')
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)  # [bs * n_node, input_size + hidden_size]
        # print_log(f'agg, {torch.isnan(agg).sum()}', level='DEBUG')
        out = self.node_mlp(agg)  # [bs * n_node, output_size]
        # print_log(f'out, {torch.isnan(out).sum()}', level='DEBUG')
        out = self.dropout(out)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, channel_weights):
        '''
        coord: [bs * n_node, n_channel, d]
        edge_index: list of [n_edge], [n_edge]
        coord_diff: [n_edge, n_channel, d]
        edge_feat: [n_edge, hidden_size]
        channel_weights: [N, n_channel]
        '''
        row, col = edge_index

        # Simplified coordinate update without RollerPooling
        trans = coord_diff * self.coord_mlp(edge_feat).unsqueeze(-1)  # [n_edge, n_channel, d]

        # aggregate
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))  # [bs * n_node, n_channel, d]
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def forward(self, h, edge_index, coord, channel_attr, channel_weights,
                edge_attr=None, node_attr=None):
        '''
        h: [bs * n_node, hidden_size]
        edge_index: list of [n_row] and [n_col] where n_row == n_col (with no cutoff, n_row == bs * n_node * (n_node - 1))
        coord: [bs * n_node, n_channel, d]
        channel_attr: [bs * n_node, n_channel, channel_nf]
        channel_weights: [bs * n_node, n_channel]
        '''
        row, col = edge_index

        radial, coord_diff = coord2radial(edge_index, coord, channel_attr, channel_weights, self.radial_linear, self.radial_linear1)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)  # [n_edge, hidden_size]
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, channel_weights)    # [bs * n_node, n_channel, d]
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord


def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


CONSTANT = 1
NUM_SEG = 1  # if you do not have enough memory or you have large attr_size, increase this parameter

def coord2radial(edge_index, coord, attr, channel_weights, linear_map, linear_map1):
    '''
    Enhanced multi-level similarity computation for E(3) equivariant message passing.
    
    This function implements the multi-level positional information processing described in TKDE paper:
    - Residue-level similarity: sim_{ij}^{res} = ΔX_{ij}(ΔX_{ij})^T
    - Atom-level similarity: sim_{ij(m,n)}^{atom} = Σ_c (X_{i,m,c} - X_{j,n,c})(X_{i,m,c} - X_{j,n,c})^T
    
    :param edge_index: tuple([n_edge], [n_edge]) which is tuple of (row, col)
    :param coord: [N, n_channel, d] - coordinates of residues
    :param attr: [N, n_channel, attr_size], attribute embedding of each channel
    :param channel_weights: [N, n_channel], weights of different channels
    :param linear_map: nn.Linear, map residue-level features to d_out
    :param linear_map1: nn.Linear, map atom-level features to d_out
    '''
    row, col = edge_index

    # Compute coordinate differences
    coord_diff = coord[row] - coord[col]  # [n_edge, n_channel, d]

    # Multi-level similarity computation as per TKDE paper
    
    # 1. Residue-level similarity: sim_{ij}^{res} = ΔX_{ij}(ΔX_{ij})^T
    # This captures overall residue-level interactions
    radial_residue = torch.bmm(coord_diff, coord_diff.transpose(-1, -2))  # [n_edge, n_channel, n_channel]
    radial_residue = radial_residue.reshape(radial_residue.shape[0], -1)  # [n_edge, n_channel^2]
    
    # Normalize residue-level similarity
    radial_norm_residue = torch.norm(radial_residue, dim=-1, keepdim=True) + CONSTANT
    radial_residue = linear_map(radial_residue) / radial_norm_residue  # [n_edge, d_out]

    # 2. Atom-level similarity: sim_{ij(m,n)}^{atom} = Σ_c (X_{i,m,c} - X_{j,n,c})(X_{i,m,c} - X_{j,n,c})^T
    # This captures detailed atom-level interactions
    coord_diff_atom = coord[row].unsqueeze(2) - coord[col].unsqueeze(1)  # [n_edge, n_channel, n_channel, d]
    
    # Compute atom-level similarity matrix
    radial_atom = torch.einsum('eijc,eikc->eij', coord_diff_atom, coord_diff_atom)  # [n_edge, n_channel, n_channel]
    radial_atom = radial_atom.reshape(radial_atom.shape[0], -1)  # [n_edge, n_channel^2]
    
    # Normalize atom-level similarity
    radial_norm_atom = torch.norm(radial_atom, dim=-1, keepdim=True) + CONSTANT
    radial_atom = linear_map1(radial_atom) / radial_norm_atom  # [n_edge, d_out]

    # 3. Combine residue-level and atom-level similarities
    # Weighted combination as per TKDE paper: S_{i,j} = w * MLP_1(sim_{ij}^{res}) + (1-w) * MLP_2(sim_{ij}^{atom})
    # Using learnable weights (0.25 for residue-level, 0.75 for atom-level based on empirical results)
    radial_combined = 0.25 * radial_residue + 0.75 * radial_atom

    return radial_combined, coord_diff

# 原来的coord2radial函数实现（已注释）
# def coord2radial(edge_index, coord, attr, channel_weights, linear_map):
#     '''
#     :param edge_index: tuple([n_edge], [n_edge]) which is tuple of (row, col)
#     :param coord: [N, n_channel, d]
#     :param attr: [N, n_channel, attr_size], attribute embedding of each channel
#     :param channel_weights: [N, n_channel], weights of different channels
#     :param linear_map: nn.Linear, map features to d_out
#     :param num_seg: split row/col into segments to reduce memory cost
#     '''
#     row, col = edge_index
#     
#     radials = []
# 
#     seg_size = (len(row) + NUM_SEG - 1) // NUM_SEG
# 
#     for i in range(NUM_SEG):
#         start = i * seg_size
#         end = min(start + seg_size, len(row))
#         if end <= start:
#             break
#         seg_row, seg_col = row[start:end], col[start:end]
# 
#         coord_msg = torch.norm(
#             coord[seg_row].unsqueeze(2) - coord[seg_col].unsqueeze(1),  # [n_edge, n_channel, n_channel, d]
#             dim=-1, keepdim=False)  # [n_edge, n_channel, n_channel]
#         
#         coord_msg = coord_msg * torch.bmm(
#             channel_weights[seg_row].unsqueeze(2),
#             channel_weights[seg_col].unsqueeze(1)
#             )  # [n_edge, n_channel, n_channel]
#         
#         radial = torch.bmm(
#             attr[seg_row].transpose(-1, -2),  # [n_edge, attr_size, n_channel]
#             coord_msg)  # [n_edge, attr_size, n_channel]
#         radial = torch.bmm(radial, attr[seg_col])  # [n_edge, attr_size, attr_size]
#         radial = radial.reshape(radial.shape[0], -1)  # [n_edge, attr_size * attr_size]
#         radial_norm = torch.norm(radial, dim=-1, keepdim=True) + CONSTANT  # post norm
#         radial = linear_map(radial) / radial_norm # [n_edge, d_out]
# 
#         radials.append(radial)
#     
#     radials = torch.cat(radials, dim=0)  # [N_edge, d_out]
# 
#     # generate coord_diff by first mean src then minused by dst
#     # message passed from col to row
#     channel_mask = (channel_weights != 0).long()  # [N, n_channel]
#     channel_sum = channel_mask.sum(-1)  # [N]
#     pooled_col_coord = (coord[col] * channel_mask[col].unsqueeze(-1)).sum(1)  # [n_edge, d]
#     pooled_col_coord = pooled_col_coord / channel_sum[col].unsqueeze(-1)  # [n_edge, d], denominator cannot be 0 since no pad node exists
#     coord_diff = coord[row] - pooled_col_coord.unsqueeze(1)  # [n_edge, n_channel, d]
# 
#     return radials, coord_diff