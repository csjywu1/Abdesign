#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

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

from .am_egnn import AM_E_GCL
from .unified_alignment import UnifiedAlignmentModule


class AMEncoder(nn.Module):

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
            self.add_module(f'ctx_gcl_{i}', AM_E_GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf,
                edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout
            ))
            self.add_module(f'inter_gcl_{i}', AM_E_GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf,
                edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout
            ))
        self.out_layer = AM_E_GCL(
            self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf,
            radial_nf, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual
        )
        
        # Graph alignment module
        self.graph_alignment = UnifiedAlignmentModule(hidden_nf)
    
    def forward(self, h, x, ctx_edges, inter_mask, inter_x, inter_edges, update_mask, inter_update_mask, channel_attr, channel_weights,
                ctx_edge_attr=None):
        h = self.linear_in(h)
        h = self.dropout(h)
        inter_h = h[inter_mask]
        inter_channel_attr = channel_attr[inter_mask]
        inter_channel_weights = channel_weights[inter_mask]

        ctx_states, ctx_coords, inter_coords = [], [], []
        for i in range(0, self.n_layers):
            h, x = self._modules[f'ctx_gcl_{i}'](
                h, ctx_edges, x, channel_attr, channel_weights,
                edge_attr=ctx_edge_attr)
            # synchronization of the shadow paratope (native -> shadow)
            inter_h = inter_h.clone()
            inter_h[inter_update_mask] = h[update_mask]
            inter_h, inter_x = self._modules[f'inter_gcl_{i}'](
                inter_h, inter_edges, inter_x, inter_channel_attr, inter_channel_weights
            )
            
            # Apply graph alignment between large and small graphs
            if inter_h.size(0) > 0 and h.size(0) > 0:
                h_large_corresponding = h[inter_mask]
                inter_h_aligned = self.graph_alignment(inter_h, h_large_corresponding)
                inter_h = inter_h_aligned
            
            # synchronization of the shadow paratope (shadow -> native)
            h = h.clone()
            h[inter_mask] = inter_h
            ctx_states.append(h)
            ctx_coords.append(x)
            inter_coords.append(inter_x)

        h, x = self.out_layer(
            h, ctx_edges, x, channel_attr, channel_weights,
            edge_attr=ctx_edge_attr)
        ctx_states.append(h)
        ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
            inter_x = torch.mean(torch.stack(inter_coords), dim=0)
        h = self.dropout(h)
        h = self.linear_out(h)
        return h, x, inter_x
