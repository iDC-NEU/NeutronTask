import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dist_utils import DistEnv
import torch.distributed as dist

try:
    from spmm_cpp import spmm_cusparse_coo, spmm_cusparse_csr
    def spmm(A,B,C): 
        if DistEnv.env.csr_enabled:
            spmm_cusparse_csr(A.crow_indices().int(), A.col_indices().int(), A.values(), A.size(0), A.size(1), \
                B, C, 1.0, 1.0, DistEnv.env.half_enabled)
        else:
            spmm_cusparse_coo(A.indices()[0].int(), A.indices()[1].int(), A.values(), A.size(0), A.size(1), \
                B, C, 1.0, 1.0, DistEnv.env.half_enabled)
except ImportError as e:
    print('no spmm cpp:', e)
    spmm = lambda A,B,C: C.addmm_(A,B)


def broadcast(local_adj_parts, local_feature, tag):
    env = DistEnv.env
    z_loc = torch.zeros_like(local_feature)
    feature_bcast = torch.zeros_like(local_feature)
    
    for src in range(env.world_size):
        if src==env.rank:
            feature_bcast = local_feature.clone()
        # env.barrier_all()
        with env.timer.timing_cuda('broadcast'):
            dist.broadcast(feature_bcast, src=src)

        with env.timer.timing_cuda('spmm'):
            spmm(local_adj_parts[src], feature_bcast, z_loc)
    return z_loc


class DistGCNLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_feature, weight, local_adj_parts, tag):
        ctx.save_for_backward(local_feature, weight)
        ctx.local_adj_parts = local_adj_parts
        ctx.tag = tag
        z_local = broadcast(local_adj_parts, local_feature, 'Forward'+tag)
        with DistEnv.env.timer.timing_cuda('mm'):
            z_local = torch.mm(z_local, weight)
        return z_local

    @staticmethod
    def backward(ctx, grad_output):
        local_feature,  weight = ctx.saved_tensors
        ag = broadcast(ctx.local_adj_parts, grad_output, 'Backward'+ctx.tag)
        with DistEnv.env.timer.timing_cuda('mm'):
            grad_feature = torch.mm(ag, weight.t())
            grad_weight = torch.mm(local_feature.t(), ag)
        with DistEnv.env.timer.timing_cuda('all_reduce'):
            DistEnv.env.all_reduce_sum(grad_weight)
        return grad_feature, grad_weight, None, None


class DistMMLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_feature, weight, tag):
        ctx.save_for_backward(local_feature, weight)
        ctx.tag = tag
        Hw = torch.mm(local_feature, weight)
        all_Hw = DistEnv.env.all_gather_then_cat(Hw)
        return all_Hw

    @staticmethod
    def backward(ctx, grad_output):
        local_feature,  weight = ctx.saved_tensors
        split_sz = local_feature.size(0)
        rank = DistEnv.env.rank
        grad_output = grad_output[split_sz*rank:split_sz*(rank+1),:]
        grad_feature = torch.mm(grad_output, weight.t())
        grad_weight = torch.mm(local_feature.t(), grad_output)
        DistEnv.env.all_reduce_sum(grad_weight)
        return grad_feature, grad_weight, None


class GAT(nn.Module):
    def __init__(self, g, env, hidden_dim=16, nlayers=2):  
        super().__init__()
        self.g, self.env = g, env
        in_dim, out_dim = g.features.size(1), g.num_classes
        self.nlayers = nlayers 

        torch.manual_seed(0)
        self.weights = nn.ParameterList()
        self.attention_weights = nn.ParameterList()
        self.weights.append(nn.Parameter(torch.rand(in_dim, hidden_dim).to(env.device)))
        self.attention_weights.append(nn.Parameter(torch.rand(2 * hidden_dim, 1).to(env.device)))
        
        for _ in range(nlayers - 2): 
            self.weights.append(nn.Parameter(torch.rand(hidden_dim, hidden_dim).to(env.device)))
            self.attention_weights.append(nn.Parameter(torch.rand(2 * hidden_dim, 1).to(env.device)))

        self.weights.append(nn.Parameter(torch.rand(hidden_dim, out_dim).to(env.device)))
        self.attention_weights.append(nn.Parameter(torch.rand(2 * out_dim, 1).to(env.device)))

    def forward(self, local_features):
        local_edge_index = self.g.adj._indices()

        all_H = local_features
        for i in range(self.nlayers): 
            self.env.logger.log(f'L{i + 1}', self.weights[i].sum(), self.attention_weights[i].sum())

            all_H = DistMMLayer.apply(all_H, self.weights[i], f'L{i + 1}')
            edge_features = torch.cat((all_H[local_edge_index[0, :], :], all_H[local_edge_index[1, :]]), dim=1)


            att_input = F.leaky_relu(torch.mm(edge_features, self.attention_weights[i]).squeeze())
            att_input = att_input.to(self.env.device)
            local_edge_index = local_edge_index.to(self.env.device)
            att_input = torch.sparse_coo_tensor(local_edge_index, att_input, self.g.adj.size())
            attention = torch.sparse.softmax(att_input, dim=1)

            all_H = torch.sparse.mm(attention, all_H)

            if i < self.nlayers - 1: 
                all_H = F.elu(all_H)

        return F.log_softmax(all_H, dim=1)

# class GAT(nn.Module):
#     def __init__(self, g, env, hidden_dim=16, nlayers=2):
#         super().__init__()
#         self.g, self.env = g, env
#         in_dim, out_dim = g.local_features.size(1), g.num_classes
#         torch.manual_seed(0)

#         self.weight1 = nn.Parameter(torch.rand(in_dim, hidden_dim)).to(env.device)
#         self.weight2 = nn.Parameter(torch.rand(hidden_dim, out_dim)).to(env.device)

#         self.attention_weight1 = nn.Parameter(torch.rand(2*hidden_dim, 1)).to(env.device)
#         self.attention_weight2 = nn.Parameter(torch.rand(out_dim*2, 1)).to(env.device)

#     def forward(self, local_features):
#         local_edge_index = self.g.local_adj._indices()
#         self.env.logger.log('L1', self.weight1.sum(), self.attention_weight1.sum())

#         # Hw1 = torch.mm(local_features, self.weight1)
#         # all_Hw1 = self.env.all_gather_then_cat(Hw1)
#         all_Hw1 = DistMMLayer.apply(local_features, self.weight1, 'L1')

#         # Hw_bcast = torch.zeros_like(Hw1)
#         # for src in range(self.env.world_size):
#         #     if src == self.env.rank:
#         #         Hw_bcast = Hw1.clone()
#         #     dist.broadcast(Hw_bcast, src=src)

#         edge_features = torch.cat((all_Hw1[local_edge_index[0, :], :], all_Hw1[local_edge_index[1, :], :]), dim=1)

#         att_input = F.leaky_relu(torch.mm(edge_features, self.attention_weight1).squeeze())
#         att_input = torch.sparse_coo_tensor(local_edge_index, att_input, self.g.local_adj.size())
#         attention = torch.sparse.softmax(att_input, dim=1)
#         # print(attention.size(), Hw1.size())

#         hidden_features = torch.sparse.mm(attention, all_Hw1)
#         hidden_features = F.elu(hidden_features)


#         # self.env.logger.log('L2', self.weight2.sum(), self.attention_weight2.sum())
#         # Hw2 = torch.mm(hidden_features, self.weight2)
#         # all_Hw2 = self.env.all_gather_then_cat(Hw2)
#         all_Hw2 = DistMMLayer.apply(hidden_features, self.weight2, 'L2')
#         edge_features = torch.cat((all_Hw2[local_edge_index[0, :], :], all_Hw2[local_edge_index[1, :], :]), dim=1)

#         att_input = F.leaky_relu(torch.mm(edge_features, self.attention_weight2).squeeze())
#         att_input = torch.sparse_coo_tensor(local_edge_index, att_input, self.g.local_adj.size())
#         attention = torch.sparse.softmax(att_input, dim=1)

#         outputs = torch.sparse.mm(attention, all_Hw2)
#         return F.log_softmax(outputs, 1)
