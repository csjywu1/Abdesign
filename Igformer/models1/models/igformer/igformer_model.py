#!/usr/bin/python
# -*- coding:utf-8 -*-
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加abdesign路径到sys.path
import sys
sys.path.insert(0, '../../..')  # 从models1/models/dyMEAN/回到abdesign根目录

from data.pdb_utils import VOCAB
from utils.nn_utils import SeparatedAminoAcidFeature, ProteinFeature
from utils.nn_utils import GMEdgeConstructor, SeperatedCoordNormalizer
from utils.nn_utils import _knn_edges
from evaluation.rmsd import kabsch_torch

# 使用igformer的模块
from modules.am_enc import AMEncoder
from modules.am_egnn import AMEGNN


def full_attention_conv(qs, ks, vs, output_attn=False):
    # normalize input
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    # compute attention for visualization if needed
    if output_attn:
        attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
        normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
        attention = attention / normalizer

    if output_attn:
        return attn_output, attention
    else:
        return attn_output


class TransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        query = self.Wq(query_input).reshape(-1,
                                             self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1,
                                            self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1,
                                                  self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(
                query, key, value, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(
                query, key, value)  # [N, H, D]

        final_output = attention_output
        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, data):
        x = data
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class SGFormer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=False, use_act=True,
                 graph_weight=0.8, gnn=None, aggregate='add'):
        super().__init__()
        self.trans_conv = TransConv(in_channels, hidden_channels, num_layers, num_heads, alpha, dropout, use_bn,
                                    use_residual, use_weight)
        self.gnn = gnn
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.use_act = use_act

        self.aggregate = aggregate

        if aggregate == 'add':
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')

    def forward(self, data):
        x1 = self.trans_conv(data)
        if self.use_graph:
            x2 = self.gnn(data)
            if self.aggregate == 'add':
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
        else:
            x = x1
        x = self.fc(x)
        return x

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]
        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.gnn.reset_parameters()


class CustomSeparatedAminoAcidFeature(SeparatedAminoAcidFeature):
    """
    Custom SeparatedAminoAcidFeature that integrates APP, SGFormer, and EGNN processing
    保持APP和SGFormer模块，但简化实现以避免NaN问题
    """
    def __init__(self, embed_size, atom_embed_size, hidden_size, n_channel, 
                 app_alpha, app_projection, sgformer_W, final_projection, init_egnn,
                 relative_position=True, edge_constructor=GMEdgeConstructor, 
                 fix_atom_weights=False, backbone_only=False):
        super().__init__(embed_size, atom_embed_size, relative_position=relative_position, 
                        edge_constructor=edge_constructor, fix_atom_weights=fix_atom_weights, 
                        backbone_only=backbone_only)
        
        # Store references to the modules
        self.hidden_size = hidden_size  # 添加hidden_size属性
        self.app_alpha = app_alpha
        self.app_projection = app_projection
        # 创建新的SGFormer实例
        self.sgformer = SGFormer(
            in_channels=hidden_size,
            hidden_channels=hidden_size, 
            out_channels=hidden_size,
            num_layers=2,
            num_heads=1,
            alpha=0.5,
            dropout=0.1,
            use_bn=True,
            use_residual=True,
            use_weight=True,
            use_graph=False,
            use_act=True
        )
        # 保持向后兼容
        self.sgformer_W = sgformer_W
        self.final_projection = final_projection
        self.init_egnn = init_egnn
    
    def _apply_app(self, H, ctx_edges):
        """
        Approximate Personalized Propagation (APP) - 数值稳定版本
        保持APP功能但增加数值稳定性
        """
        # Create adjacency matrix
        num_nodes = H.size(0)
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=H.device, dtype=H.dtype)
        adj_matrix[ctx_edges[0], ctx_edges[1]] = 1.0
        adj_matrix[ctx_edges[1], ctx_edges[0]] = 1.0  # Make symmetric
        
        # Add self-loops for numerical stability
        adj_matrix = adj_matrix + torch.eye(num_nodes, device=H.device, dtype=H.dtype)
        
        # Compute degree matrix
        degree = adj_matrix.sum(dim=1)
        
        # Normalized transition matrix with better numerical stability
        degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)  # 增加epsilon避免除零
        degree_inv_sqrt_matrix = torch.diag(degree_inv_sqrt)
        P = torch.matmul(torch.matmul(degree_inv_sqrt_matrix, adj_matrix), degree_inv_sqrt_matrix)
        
        # Apply APP
        H_proj = self.app_projection(H)  # [N, hidden_size]
        
        # 确保app_alpha在合理范围内
        app_alpha = torch.sigmoid(self.app_alpha)  # 使用sigmoid确保在[0,1]范围内
        H_app = app_alpha * torch.matmul(P, H_proj) + (1 - app_alpha) * H_proj
        
        return H_app
    
    def _apply_sgformer(self, H):
        """
        Simplified Graph Transformer (SGFormer) - 使用完整的SGFormer实现
        """
        # 使用新的SGFormer实现
        H_sgformer = self.sgformer(H)  # [N, hidden_size]
        
        return H_sgformer
    
    def forward(self, X, S, batch_id, k_neighbors, residue_pos=None, smooth_prob=None, smooth_mask=None):
        # Get base embeddings from parent class
        H_0, (ctx_edges, inter_edges), (atom_embeddings, atom_weights) = super().forward(
            X, S, batch_id, k_neighbors, residue_pos, smooth_prob, smooth_mask)
        
        # Apply APP and SGFormer
        H_local = self._apply_app(H_0, ctx_edges)  # [N, hidden_size]
        H_global = self._apply_sgformer(H_0)       # [N, hidden_size]
        H_combined = H_local + H_global            # [N, hidden_size]
        
        # Apply EGNN
        H_processed, X_processed = self.init_egnn(H_combined, X, ctx_edges,
                                                 channel_attr=atom_embeddings,
                                                 channel_weights=atom_weights)
        
        # Return processed embeddings (already 128 dimensions)
        H_final = H_processed  # [N, hidden_size]
        
        return H_final, (ctx_edges, inter_edges), (atom_embeddings, atom_weights)


class TriangleAttention(nn.Module):
    """
    Triangle Attention module for paratope-epitope interactions
    回退到原始简单版本
    """
    def __init__(self, hidden_dim, n_channel=14):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_channel = n_channel
        
        # 原始简单的MLP网络
        self.residue_similarity_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.atom_similarity_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Learnable weight for combining residue and atom level similarities
        self.similarity_weight = nn.Parameter(torch.tensor(0.5))
        
        # Interaction MLP
        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # H_p + H_e + S_ij
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Triangle multiplicative module components
        self.W_l = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        # Axial attention components
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, H, X, paratope_mask, epitope_mask):
        """
        H: [N, hidden_dim] - node features
        X: [N, n_channel, 3] - coordinates
        paratope_mask: [N] - boolean mask for paratope nodes
        epitope_mask: [N] - boolean mask for epitope nodes
        """
        # Extract paratope and epitope features and coordinates
        H_p = H[paratope_mask]  # [N_p, hidden_dim]
        H_e = H[epitope_mask]  # [N_e, hidden_dim]
        X_p = X[paratope_mask]  # [N_p, n_channel, 3]
        X_e = X[epitope_mask]  # [N_e, n_channel, 3]
        
        # Get CA atom coordinates (index 1)
        X_p_ca = X_p[:, 1]  # [N_p, 3]
        X_e_ca = X_e[:, 1]  # [N_e, 3]
        
        # Compute multi-level geometric similarity following EGNN approach
        # 1. Residue-level similarity: S_ij^res = ΔX_ij(ΔX_ij)^T
        delta_X = X_p_ca.unsqueeze(1) - X_e_ca.unsqueeze(0)  # [N_p, N_e, 3]
        S_residue = torch.sum(delta_X * delta_X, dim=-1, keepdim=True)  # [N_p, N_e, 1]
        
        # 2. Atom-level similarity: S_ij^atom = Σ_c (X_p,i,CA,c - X_e,j,CA,c)²
        S_atom = torch.sum((X_p_ca.unsqueeze(1) - X_e_ca.unsqueeze(0))**2, dim=-1, keepdim=True)  # [N_p, N_e, 1]
        
        # 3. Process similarities through MLPs
        S_residue_feat = self.residue_similarity_mlp(S_residue)  # [N_p, N_e, hidden_dim]
        S_atom_feat = self.atom_similarity_mlp(S_atom)  # [N_p, N_e, hidden_dim]
        
        # 4. Weighted combination: S_ij = w * MLP_1(S_ij^res) + (1-w) * MLP_2(S_ij^atom)
        w = torch.sigmoid(self.similarity_weight)  # ensure weight is in [0, 1]
        S_combined = w * S_residue_feat + (1 - w) * S_atom_feat  # [N_p, N_e, hidden_dim]
        
        # 5. Compute interaction features Z = MLP(H_p ⊕ H_e ⊕ S_ij)
        H_p_expanded = H_p.unsqueeze(1).expand(-1, H_e.size(0), -1)  # [N_p, N_e, hidden_dim]
        H_e_expanded = H_e.unsqueeze(0).expand(H_p.size(0), -1, -1)  # [N_p, N_e, hidden_dim]
        Z_input = torch.cat([H_p_expanded, H_e_expanded, S_combined], dim=-1)  # [N_p, N_e, 3*hidden_dim]
        
        # Reshape for MLP processing
        Z_flat = Z_input.view(-1, Z_input.size(-1))  # [N_p * N_e, 3*hidden_dim]
        Z_flat = self.interaction_mlp(Z_flat)  # [N_p * N_e, hidden_dim]
        Z = Z_flat.view(Z_input.size(0), Z_input.size(1), -1)  # [N_p, N_e, hidden_dim]
        
        # Triangle multiplicative module
        H_l = self.W_l(H_p)  # [N_p, hidden_dim]
        H_r = self.W_r(H_e)  # [N_e, hidden_dim]
        
        # Compute triangle multiplicative update
        H_l_expanded = H_l.unsqueeze(1).expand(-1, H_e.size(0), -1)  # [N_p, N_e, hidden_dim]
        H_r_expanded = H_r.unsqueeze(0).expand(H_p.size(0), -1, -1)  # [N_p, N_e, hidden_dim]
        I = H_l_expanded * H_r_expanded  # [N_p, N_e, hidden_dim]
        
        # Gating mechanism
        gate = torch.sigmoid(self.W_g(Z))  # [N_p, N_e, hidden_dim]
        f_triangle = gate * I  # [N_p, N_e, hidden_dim]
        
        # Axial attention
        Q = self.W_q(Z)  # [N_p, N_e, hidden_dim]
        K = self.W_k(Z)  # [N_p, N_e, hidden_dim]
        V = self.W_v(Z)  # [N_p, N_e, hidden_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # [N_p, N_e, N_e]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [N_p, N_e, N_e]
        
        # Apply attention
        f_att = torch.matmul(attention_weights, V)  # [N_p, N_e, hidden_dim]
        
        # Combine triangle multiplicative and attention
        H_triangle = f_triangle + f_att  # [N_p, N_e, hidden_dim]
        
        # Aggregate back to paratope nodes
        H_updated = H.clone()
        H_updated[paratope_mask] = H_triangle.mean(dim=1)  # [N_p, hidden_dim]
        
        return H_updated


class IgformerModel(nn.Module):
    def __init__(self, embed_size, hidden_size, n_channel, num_classes,
                 mask_id=VOCAB.get_mask_idx(), k_neighbors=9, bind_dist_cutoff=6,
                 n_layers=3, iter_round=3, dropout=0.1, struct_only=False,
                 backbone_only=False, fix_channel_weights=False, pred_edge_dist=True,
                 keep_memory=True, cdr_type='H3', paratope='H3', relative_position=False) -> None:
        super().__init__()
        self.mask_id = mask_id
        self.num_classes = num_classes
        self.bind_dist_cutoff = bind_dist_cutoff
        self.k_neighbors = k_neighbors
        self.round = iter_round
        self.struct_only = struct_only

        # options
        self.backbone_only = backbone_only
        self.fix_channel_weights = fix_channel_weights
        self.pred_edge_dist = pred_edge_dist
        self.keep_memory = keep_memory
        if self.backbone_only:
            n_channel = 4
        self.cdr_type = cdr_type
        self.paratope = paratope

        atom_embed_size = embed_size // 4
        self.protein_feature = ProteinFeature(backbone_only=backbone_only)
        if keep_memory:
            self.memory_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size)  # 128 -> 128
            )
        if self.pred_edge_dist:  # use predicted dist for KNN-graph at the interface
            if self.keep_memory:  # this ffn acts on the memory
                self.edge_H_ffn = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size)
                )
            self.edge_dist_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(2 * hidden_size, hidden_size),  # 256 -> 128
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
            # this GNN encodes the initial hidden states for initial edge distance prediction
            self.init_gnn = AMEGNN(
                hidden_size, hidden_size, hidden_size, n_channel,
                channel_nf=atom_embed_size, radial_nf=hidden_size,
                in_edge_nf=0, n_layers=n_layers+1, residual=True,  # 增加层数: n_layers -> n_layers+1
                dropout=dropout, dense=False)
        if not struct_only:
            self.ffn_residue = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, self.num_classes)
            )
        else:
            self.prmsd_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
        self.gnn = AMEncoder(
            hidden_size, hidden_size, hidden_size, n_channel,  # 128 -> 128
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=0, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False)
        
        # Triangle Attention module
        self.triangle_attention = TriangleAttention(hidden_size, n_channel)
        
        # Projection layer for Triangle Attention input
        self.triangle_projection = nn.Linear(hidden_size, hidden_size)  # 128 -> 128
        
        # APP and SGFormer modules - 保持功能但增加数值稳定性
        self.app_alpha = nn.Parameter(torch.tensor(0.5))  # APP权重
        self.app_projection = nn.Linear(hidden_size, hidden_size)  # 128 -> 128
        self.sgformer_W = nn.Linear(hidden_size, hidden_size)      # 128 -> 128
        self.final_projection = nn.Linear(hidden_size, hidden_size) # 128 -> 128
        
        # EGNN for initial processing
        self.init_egnn = AMEGNN(
            hidden_size, hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=0, n_layers=2, residual=True,  # 增加层数: 1 -> 2
            dropout=dropout, dense=False)
        
        # Initialize custom aa_feature with integrated APP, SGFormer, and EGNN
        self.aa_feature = CustomSeparatedAminoAcidFeature(
            embed_size, atom_embed_size, hidden_size, n_channel,
            self.app_alpha, self.app_projection, self.sgformer_W, 
            self.final_projection, self.init_egnn,
            relative_position=relative_position,
            edge_constructor=GMEdgeConstructor,
            fix_atom_weights=fix_channel_weights,
            backbone_only=backbone_only
        )
        
        self.normalizer = SeperatedCoordNormalizer()

        # training related cache
        self.batch_constants = {}

    def init_mask(self, X, S, cmask, smask, template):
        if not self.struct_only:
            S[smask] = self.mask_id
        X[cmask] = template
        return X, S

    def message_passing(self, X, S, residue_pos, interface_X, paratope_mask, batch_id, t, memory_H=None, smooth_prob=None, smooth_mask=None):
        # embeddings - now includes APP, SGFormer, and EGNN processing
        H_0, (ctx_edges, inter_edges), (atom_embeddings, atom_weights) = self.aa_feature(X, S, batch_id, self.k_neighbors, residue_pos, smooth_prob=smooth_prob, smooth_mask=smooth_mask)

        if not self.keep_memory:
            memory_H = None

        if memory_H is not None:
            H_0 = H_0 + self.memory_ffn(memory_H)

        if self.pred_edge_dist:
            if memory_H is not None:
                edge_H = self.edge_H_ffn(memory_H)
            else:
                # replace the MLP with gnn for initial edge distance prediction
                edge_H, dumb_X = self.init_gnn(H_0, X, ctx_edges,
                                       channel_attr=atom_embeddings,
                                       channel_weights=atom_weights)
                X = X + dumb_X * 0  # to cheat the autograd check

        # update coordination of the global node
        X = self.aa_feature.update_globel_coordinates(X, S)

        # prepare local complex
        local_mask = self.batch_constants['local_mask']
        local_is_ab = self.batch_constants['local_is_ab']
        local_batch_id = self.batch_constants['local_batch_id']
        local_X = X[local_mask].clone()
        local_H = H_0[local_mask].clone()
        
        # Apply Triangle Attention for paratope-epitope interactions
        local_paratope_mask = local_is_ab  # paratope nodes in local complex
        local_epitope_mask = ~local_is_ab  # epitope nodes in local complex
        
        # Project local_H to hidden_size for Triangle Attention
        local_H_projected = self.triangle_projection(local_H)  # [N_local, hidden_size]
        local_H_projected = self.triangle_attention(local_H_projected, local_X, local_paratope_mask, local_epitope_mask)
        
        # Project back to hidden_size
        local_H = self.final_projection(local_H_projected)  # [N_local, hidden_size]
        
        # prepare local complex edges
        local_ctx_edges = self.batch_constants['local_ctx_edges']  # [2, Ec]
        local_inter_edges = self.batch_constants['local_inter_edges']  # [2, Ei]
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        offsets, max_n, gni2lni = self.batch_constants['local_edge_infos']
        # for context edges, use edges in the native paratope
        local_ctx_edges = _knn_edges(
            local_X, atom_pos, local_ctx_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors,
            (offsets, local_batch_id, max_n, gni2lni))
        # for interative edges, use edges derived from the predicted distance
        local_X[local_is_ab] = interface_X
        if self.pred_edge_dist:
            src_H, dst_H = local_H[local_inter_edges[0]], local_H[local_inter_edges[1]]
            p_edge_dist = self.edge_dist_ffn(torch.cat([src_H, dst_H], dim=-1)) +\
                          self.edge_dist_ffn(torch.cat([dst_H, src_H], dim=-1))  # perm-invariant
            p_edge_dist = p_edge_dist.squeeze()
        else:
            p_edge_dist = None
        local_inter_edges = _knn_edges(
            local_X, atom_pos, local_inter_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors,
            (offsets, local_batch_id, max_n, gni2lni), given_dist=p_edge_dist)
        local_edges = torch.cat([local_ctx_edges, local_inter_edges], dim=1)

        # Update H_0 with the processed local_H (avoid inplace operation)
        H_0 = H_0.clone()
        H_0[local_mask] = local_H
        
        # message passing
        H, pred_X, pred_local_X = self.gnn(H_0, X, ctx_edges,
                                           local_mask, local_X, local_edges,
                                           paratope_mask, local_is_ab,
                                           channel_attr=atom_embeddings,
                                           channel_weights=atom_weights)
        interface_X = pred_local_X[local_is_ab]
        pred_logits = None if self.struct_only else self.ffn_residue(H)

        return pred_logits, pred_X, interface_X, H, p_edge_dist  # [N, num_classes], [N, n_channel, 3], [Ncdr, n_channel, 3], [N, hidden_size]
    
    @torch.no_grad()
    def init_interface(self, X, S, paratope_mask, batch_id, init_noise=None):
        ag_centers = X[S == self.aa_feature.boa_idx][:, 0]  # [bs, 3]
        init_local_X = torch.zeros_like(X[paratope_mask])
        init_local_X = init_local_X + ag_centers[batch_id[paratope_mask]].unsqueeze(1)
        noise = torch.randn_like(init_local_X) if init_noise is None else init_noise
        ca_noise = noise[:, 1]
        noise = noise / 10  + ca_noise.unsqueeze(1) # scale other atoms
        noise[:, 1] = ca_noise
        init_local_X = init_local_X + noise
        return init_local_X

    @torch.no_grad()
    def _prepare_batch_constants(self, S, paratope_mask, lengths):
        # generate batch id
        batch_id = torch.zeros_like(S)  # [N]
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)  # [N], item idx in the batch
        self.batch_constants['batch_id'] = batch_id
        self.batch_constants['batch_size'] = torch.max(batch_id) + 1

        segment_ids = self.aa_feature._construct_segment_ids(S)
        self.batch_constants['segment_ids'] = segment_ids

        # interface relatd
        is_ag = segment_ids == self.aa_feature.ag_seg_id
        not_ag_global = S != self.aa_feature.boa_idx
        local_mask = torch.logical_or(
            paratope_mask, torch.logical_and(is_ag, not_ag_global)
        )
        local_segment_ids = segment_ids[local_mask]
        local_is_ab = local_segment_ids != self.aa_feature.ag_seg_id
        local_batch_id = batch_id[local_mask]
        self.batch_constants['is_ag'] = is_ag
        self.batch_constants['local_mask'] = local_mask
        self.batch_constants['local_is_ab'] = local_is_ab
        self.batch_constants['local_batch_id'] = local_batch_id
        self.batch_constants['local_segment_ids'] = local_segment_ids
        # interface local edges
        (row, col), (offsets, max_n, gni2lni) = self.aa_feature.edge_constructor.get_batch_edges(local_batch_id)
        row_segment_ids, col_segment_ids = local_segment_ids[row], local_segment_ids[col]
        is_ctx = row_segment_ids == col_segment_ids
        is_inter = torch.logical_not(is_ctx)

        self.batch_constants['local_ctx_edges'] = torch.stack([row[is_ctx], col[is_ctx]])  # [2, Ec]
        self.batch_constants['local_inter_edges'] = torch.stack([row[is_inter], col[is_inter]])  # [2, Ei]
        self.batch_constants['local_edge_infos'] = (offsets, max_n, gni2lni)

        interface_batch_id = batch_id[paratope_mask]
        self.batch_constants['interface_batch_id'] = interface_batch_id
    
    def _clean_batch_constants(self):
        self.batch_constants = {}

    @torch.no_grad()
    def _get_inter_edge_dist(self, X, S):
        local_mask = self.batch_constants['local_mask']
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        src_dst = self.batch_constants['local_inter_edges'].T
        dist = X[local_mask][src_dst]  # [Ef, 2, n_channel, 3]
        dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)  # [Ef, n_channel, n_channel, 3]
        dist = torch.norm(dist, dim=-1)  # [Ef, n_channel, n_channel]
        pos_pad = atom_pos[src_dst] == self.aa_feature.atom_pos_pad_idx # [Ef, 2, n_channel]
        pos_pad = torch.logical_or(pos_pad[:, 0].unsqueeze(2), pos_pad[:, 1].unsqueeze(1))  # [Ef, n_channel, n_channel]
        dist = dist + pos_pad * 1e10  # [Ef, n_channel, n_channel]
        dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]  # [Ef]
        return dist
        is_binding = dist <= self.bind_dist_cutoff
        return is_binding

    def _forward(self, X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, init_noise=None):
        batch_id = self.batch_constants['batch_id']

        # mask sequence and initialize coordinates with template
        X, S = self.init_mask(X, S, cmask, smask, template)

        # normalize
        X = self.normalizer.centering(X, S, batch_id, self.aa_feature)
        X = self.normalizer.normalize(X)

        # update center
        X = self.aa_feature.update_globel_coordinates(X, S)

        # prepare initial interface
        interface_X = self.init_interface(X, S, paratope_mask, batch_id, init_noise)

        # sequence and structure loss
        r_pred_S_logits, pred_S_dist, = [], None
        r_interface_X = [interface_X.clone()]  # init
        r_edge_dist = []
        memory_H = None
        # message passing
        for t in range(self.round):
            pred_S_logits, pred_X, interface_X, H, edge_dist = self.message_passing(X, S, residue_pos, interface_X, paratope_mask, batch_id, t, memory_H, pred_S_dist, smask)
            memory_H = H
            r_interface_X.append(interface_X.clone())
            r_pred_S_logits.append((pred_S_logits, smask))
            r_edge_dist.append(edge_dist)
            # 1. update X
            X = X.clone()
            X[cmask] = pred_X[cmask]
            X = self.aa_feature.update_globel_coordinates(X, S)

            if not self.struct_only:
                # 2. update S
                S = S.clone()
                if t == self.round - 1:
                    S[smask] = torch.argmax(pred_S_logits[smask], dim=-1)
                else:
                    pred_S_dist = torch.softmax(pred_S_logits[smask], dim=-1)

        interface_batch_id = self.batch_constants['interface_batch_id']

        if self.struct_only:
            # predicted rmsd
            prmsd = self.prmsd_ffn(H[cmask]).squeeze()  # [N_ab]
        else:
            prmsd = None

        # uncentering and unnormalize
        pred_X = self.normalizer.unnormalize(pred_X)
        pred_X = self.normalizer.uncentering(pred_X, batch_id)
        for i, interface_X in enumerate(r_interface_X):
            interface_X = self.normalizer.unnormalize(interface_X)
            interface_X = self.normalizer.uncentering(interface_X, interface_batch_id, _type=4)
            r_interface_X[i] = interface_X
        self.normalizer.clear_cache()

        return H, S, r_pred_S_logits, pred_X, r_interface_X,  r_edge_dist, prmsd

    def forward(self, X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, xloss_mask, context_ratio=0):
        '''
        :param X: [N, n_channel, 3], Cartesian coordinates
        :param context_ratio: float, rate of context provided in masked sequence, should be [0, 1) and anneal to 0 in training
        '''
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
            xloss_mask = xloss_mask[:, :4]
        # clone ground truth coordinates, sequence
        true_X, true_S = X.clone(), S.clone()

        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)
        batch_id = self.batch_constants['batch_id']

        # provide some ground truth for annealing sequence training
        if context_ratio > 0:
            not_ctx_mask = torch.rand_like(smask, dtype=torch.float) >= context_ratio
            smask = torch.logical_and(smask, not_ctx_mask)

        # get results
        H, pred_S, r_pred_S_logits, pred_X, r_interface_X, r_edge_dist, prmsd = self._forward(X, S, cmask, smask, paratope_mask, residue_pos, template, lengths)

        # sequence negtive log likelihood
        snll, total = 0, 0
        if not self.struct_only:
            for logits, mask in r_pred_S_logits:
                snll = snll + F.cross_entropy(logits[mask], true_S[mask], reduction='sum')
                total = total + mask.sum()
            snll = snll / total

        # structure loss
        struct_loss, struct_loss_details, bb_rmsd, ops = self.protein_feature.structure_loss(pred_X, true_X, true_S, cmask, batch_id, xloss_mask, self.aa_feature)

        # docking loss
        gt_interface_X = true_X[paratope_mask]
        # 1. interface loss (shadow paratope)
        interface_atom_pos = self.aa_feature._construct_atom_pos(true_S[paratope_mask])
        interface_atom_mask = interface_atom_pos != self.aa_feature.atom_pos_pad_idx
        interface_loss = F.smooth_l1_loss(
            r_interface_X[-1][interface_atom_mask],
            gt_interface_X[interface_atom_mask])
        # 2. edge dist loss
        if self.pred_edge_dist:
            gt_edge_dist = self._get_inter_edge_dist(self.normalizer.normalize(true_X), true_S)
            ed_loss, r_ed_losses = 0, []
            for edge_dist in r_edge_dist:
                r_ed_loss = F.smooth_l1_loss(edge_dist, gt_edge_dist)
                ed_loss = ed_loss + r_ed_loss
                r_ed_losses.append(r_ed_loss)
        else:
            r_ed_losses = [0 for _ in range(self.round)]
            ed_loss = 0
        dock_loss = interface_loss + ed_loss

        if self.struct_only:
            # predicted rmsd
            prmsd_loss = F.smooth_l1_loss(prmsd, bb_rmsd)
            pdev_loss = prmsd_loss
        else:
            pdev_loss, prmsd_loss = None, None

        # comprehensive loss
        loss = snll + struct_loss + dock_loss + (0 if pdev_loss is None else pdev_loss)

        self._clean_batch_constants()

        # AAR
        with torch.no_grad():
            aa_hit = pred_S[smask] == true_S[smask]
            aar = aa_hit.long().sum() / aa_hit.shape[0]

        return loss, (snll, aar), (struct_loss, *struct_loss_details), (dock_loss, interface_loss, ed_loss, r_ed_losses), (pdev_loss, prmsd_loss)

    def sample(self, X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, init_noise=None, return_hidden=False):
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
        gen_X, gen_S = X.clone(), S.clone()
        
        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)

        batch_id = self.batch_constants['batch_id']
        batch_size = self.batch_constants['batch_size']
        segment_ids = self.batch_constants['segment_ids']
        interface_batch_id = self.batch_constants['interface_batch_id']
        is_ab = segment_ids != self.aa_feature.ag_seg_id
        s_batch_id = batch_id[smask]

        best_metric = torch.ones(batch_size, dtype=torch.float, device=X.device) * 1e10
        interface_cmask = paratope_mask[cmask]

        n_tries = 10 if self.struct_only else 1
        for i in range(n_tries):
        
            # generate
            H, pred_S, r_pred_S_logits, pred_X, r_interface_X, _, prmsd = self._forward(X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, init_noise)

            # PPL or PRMSD
            if not self.struct_only:
                S_logits = r_pred_S_logits[-1][0][smask]
                S_probs = torch.max(torch.softmax(S_logits, dim=-1), dim=-1)[0]
                nlls = -torch.log(S_probs)
                
                # Custom implementation of scatter_mean instead of using scatter_ops
                batch_size = s_batch_id.max() + 1
                metric = torch.zeros(batch_size, dtype=nlls.dtype, device=nlls.device)
                
                # Get unique batch IDs and compute mean for each
                unique_batch_ids = torch.unique(s_batch_id)
                for batch_idx in unique_batch_ids:
                    if batch_idx < 0:  # Skip negative indices
                        continue
                    mask = (s_batch_id == batch_idx)
                    if mask.sum() > 0:
                        # Calculate mean for this batch
                        mean_val = nlls[mask].mean()
                        metric[batch_idx] = mean_val
            else:
                # Custom implementation of scatter_mean instead of using scatter_ops
                batch_size = interface_batch_id.max() + 1
                metric = torch.zeros(batch_size, dtype=prmsd.dtype, device=prmsd.device)
                
                # Get unique batch IDs and compute mean for each
                unique_batch_ids = torch.unique(interface_batch_id)
                for batch_idx in unique_batch_ids:
                    if batch_idx < 0:  # Skip negative indices
                        continue
                    mask = (interface_batch_id == batch_idx)
                    if mask.sum() > 0:
                        # Calculate mean for this batch
                        mean_val = prmsd[interface_cmask][mask].mean()
                        metric[batch_idx] = mean_val

            update = metric < best_metric
            cupdate = cmask & update[batch_id]
            supdate = smask & update[batch_id]
            # update metric history
            best_metric[update] = metric[update]

            # 1. set generated part
            gen_X[cupdate] = pred_X[cupdate]
            if not self.struct_only:
                gen_S[supdate] = pred_S[supdate]
        
            interface_X = r_interface_X[-1]
            # 2. align by cdr
            for i in range(batch_size):
                if not update[i]:
                    continue
                # 1. align CDRH3
                is_cur_graph = batch_id == i
                cdrh3_cur_graph = torch.logical_and(is_cur_graph, paratope_mask)
                ori_cdr = gen_X[cdrh3_cur_graph][:, :4]  # backbone
                pred_cdr = interface_X[interface_batch_id == i][:, :4]
                _, R, t = kabsch_torch(ori_cdr.reshape(-1, 3), pred_cdr.reshape(-1, 3))

                # 2. tranform antibody
                is_cur_ab = is_cur_graph & is_ab
                ab_X = torch.matmul(gen_X[is_cur_ab], R.T) + t
                gen_X[is_cur_ab] = ab_X

        self._clean_batch_constants()

        if return_hidden:
            return gen_X, gen_S, metric, H
        return gen_X, gen_S, metric

if __name__ == '__main__':
    torch.random.manual_seed(0)
    # equivariance test
    embed_size, hidden_size = 64, 128
    n_channel, d = 14, 3
    n_aa_type = 20
    scale = 10
    dtype = torch.float
    device = torch.device('cuda:0')
    model = dyMEANModel(embed_size, hidden_size, n_channel,
                   VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
                   k_neighbors=9, bind_dist_cutoff=6.6, n_layers=3)
    model.to(device)
    model.eval()

    ag_len, h_len, l_len = 48, 120, 110
    center_x = torch.randn(3, 1, n_channel, d, device=device, dtype=dtype) * scale
    ag_X = torch.randn(ag_len, n_channel, d, device=device, dtype=dtype) * scale
    h_X = torch.randn(h_len, n_channel, d, device=device, dtype=dtype) * scale
    l_X = torch.randn(l_len, n_channel, d, device=device, dtype=dtype) * scale

    X = torch.cat([center_x[0], ag_X, center_x[1], h_X, center_x[2], l_X], dim=0)
    S = torch.cat([torch.tensor([model.aa_feature.boa_idx], device=device),
                   torch.randint(low=0, high=20, size=(ag_len,), device=device),
                   torch.tensor([model.aa_feature.boh_idx], device=device),
                   torch.randint(low=0, high=20, size=(h_len,), device=device),
                   torch.tensor([model.aa_feature.bol_idx], device=device),
                   torch.randint(low=0, high=20, size=(l_len,), device=device)], dim=0)
    cmask = torch.tensor([0] + [0 for _ in range(ag_len)] + [0] + [1 for _ in range(h_len)] + [0] + [1 for _ in range(l_len)], device=device).bool()
    smask = torch.zeros_like(cmask)
    smask[ag_len+10:ag_len+20] = 1
    paratope_mask = smask
    residue_pos = torch.tensor([0] + [0 for _ in range(ag_len)] + [0] + list(range(1, h_len + 1)) + [0] + list(range(1, l_len + 1)), device=device)
    template = torch.randn_like(X[cmask]) * scale
    lengths = torch.tensor([ag_len + h_len + l_len + 3], device=device)
    noise = torch.randn_like(X[paratope_mask])

    torch.random.manual_seed(1)
    gen_X, _, _ = model.sample(X, S, cmask, smask, smask, residue_pos, template, lengths, init_noise=noise)
    # tmpx1 = model.tmpx

    # random rotaion matrix
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q_ag, t_ag = U.mm(V), torch.randn(3, device=device)

    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q_ab, t_ab = U.mm(V), torch.randn(3, device=device)

    ag_X = torch.matmul(ag_X, Q_ag) + t_ag
    X = torch.cat([center_x[0], ag_X, center_x[1], h_X, center_x[2], l_X], dim=0)
    template = torch.matmul(template, Q_ab) + t_ab

    # this is f([Q1x1+t1, Q2x2+t2])
    torch.random.manual_seed(1)
    noise = torch.matmul(noise, Q_ag)
    gen_op_X, _, _ = model.sample(X, S, cmask, smask, smask, residue_pos, template, lengths, init_noise=noise)
    # tmpx2 = model.tmpx
    # gt_tmpx = torch.matmul(tmpx1, Q_ag) + t_ag
    # error = torch.abs(gt_tmpx[:, :4] - tmpx2[:, :4]).sum(-1).flatten().mean()
    # # error = torch.abs(tmpx2 - tmpx1).sum(-1).flatten().mean()
    # print(error.item())
    # assert error < 1e-3
    # print('independent equivariance check passed')


    gt_op_X = torch.matmul(gen_X, Q_ag) + t_ag

    error = torch.abs(gt_op_X[cmask][:, :4] - gen_op_X[cmask][:, :4]).sum(-1).flatten().mean()
    print(error.item())
    assert error < 1e-3
    print('independent equivariance check passed')
