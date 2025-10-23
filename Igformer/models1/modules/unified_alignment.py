#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedAlignmentModule(nn.Module):
    """
    Unified Alignment Module for large-small graph alignment
    Based on Igformer paper implementation using HSIC and MMD
    """
    
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Alignment projection layers
        self.align_proj_large = nn.Linear(hidden_size, hidden_size)
        self.align_proj_small = nn.Linear(hidden_size, hidden_size)
        
        # Kernel parameters for HSIC and MMD
        self.sigma = nn.Parameter(torch.tensor(1.0))
        
    def gaussian_kernel(self, X, Y=None):
        """
        Compute Gaussian kernel matrix
        """
        if Y is None:
            Y = X
            
        # Compute pairwise distances
        X_norm = (X ** 2).sum(1).view(-1, 1)
        Y_norm = (Y ** 2).sum(1).view(1, -1)
        dist = X_norm + Y_norm - 2.0 * torch.mm(X, Y.t())
        
        # Apply Gaussian kernel
        kernel = torch.exp(-dist / (2 * self.sigma ** 2))
        return kernel
    
    def compute_hsic(self, X, Y):
        """
        Compute Hilbert-Schmidt Independence Criterion (HSIC)
        """
        n = X.size(0)
        
        # Compute kernel matrices
        K_X = self.gaussian_kernel(X)
        K_Y = self.gaussian_kernel(Y)
        
        # Center the kernel matrices
        H = torch.eye(n, device=X.device) - 1.0 / n
        K_X_centered = torch.mm(torch.mm(H, K_X), H)
        K_Y_centered = torch.mm(torch.mm(H, K_Y), H)
        
        # Compute HSIC
        hsic = torch.trace(torch.mm(K_X_centered, K_Y_centered)) / (n * n)
        return hsic
    
    def compute_mmd(self, X, Y):
        """
        Compute Maximum Mean Discrepancy (MMD)
        """
        # Compute kernel matrices
        K_XX = self.gaussian_kernel(X, X)
        K_YY = self.gaussian_kernel(Y, Y)
        K_XY = self.gaussian_kernel(X, Y)
        
        # Compute MMD
        mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        return mmd
    
    def forward(self, small_graph_features, large_graph_features):
        """
        Align small graph features with large graph features
        
        Args:
            small_graph_features: [N_small, hidden_size] - features from small graph
            large_graph_features: [N_large, hidden_size] - features from large graph
            
        Returns:
            aligned_features: [N_small, hidden_size] - aligned small graph features
        """
        # Project features to alignment space
        small_proj = self.align_proj_small(small_graph_features)
        large_proj = self.align_proj_large(large_graph_features)
        
        # Compute alignment loss (HSIC + MMD)
        hsic_loss = self.compute_hsic(small_proj, large_proj)
        mmd_loss = self.compute_mmd(small_proj, large_proj)
        
        # Alignment loss (minimize HSIC for independence, minimize MMD for similarity)
        alignment_loss = hsic_loss + mmd_loss
        
        # Apply alignment transformation
        # Simple linear transformation for alignment
        align_weight = torch.sigmoid(torch.tensor(0.5))  # Learnable alignment strength
        aligned_features = align_weight * small_proj + (1 - align_weight) * large_proj
        
        return aligned_features
