import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import torch_sparse
from torch_scatter import scatter_sum
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from utils_package.utils import (
    build_sim, compute_normalized_laplacian, build_knn_neighbourhood,
    build_knn_normalized_graph
)
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization
from collections import defaultdict
import math
import random
import json

class nrdmc(GeneralRecommender):
    def __init__(self, config, dataset):
        super(nrdmc, self).__init__(config, dataset)
        self.sparse = True
        self.data_name = config['dataset']
        self.embedding_dim = config['embedding_size']
        self.s_drop = config['s_drop']
        self.m_drop = config['m_drop']
        self.align_temp = config['align_temp']
        self.align_loss_weight = config['align_loss_weight']
        self.item_knn_k = config['item_knn_k']
        self.user_knn_k = config['user_knn_k']
        self.i_mm_image_weight = config['i_mm_image_weight']
        self.u_mm_image_weight = config['u_mm_image_weight']
        self.ii_co_weight = config['ii_co_weight']
        self.uu_co_weight = config['uu_co_weight']
        self.n_ii_layers = config['n_ii_layers']
        self.n_uu_layers = config['n_uu_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.n_b = config['n_b']
        self.multi_view_loss_weight = config['multi_view_loss_weight']
        self.multi_view_temp = config['multi_view_temp']
        self.n_nodes = self.n_users + self.n_items

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        self.ui_indices = torch.LongTensor(np.vstack((self.interaction_matrix.row, self.interaction_matrix.col))).to(self.device)
        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        self.dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        dict_user_co_graph_file = os.path.join(self.dataset_path, 'dict_user_co_occ_graph.pt')
        dict_item_co_graph_file = os.path.join(self.dataset_path, 'dict_item_co_occ_graph.pt')
        self.dict_user_co_graph = torch.load(dict_user_co_graph_file)
        self.dict_item_co_graph = torch.load(dict_item_co_graph_file)

        sp_inter_m = self.sparse_mx_to_torch_sparse_tensor(self.interaction_matrix).to(self.device)
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.user_v_interest = torch.sparse.mm(sp_inter_m, self.v_feat) / torch.sparse.sum(sp_inter_m, [1]).unsqueeze(dim=1).to_dense()
            self.user_v_prefer = torch.nn.Parameter(self.user_v_interest.to(self.device), requires_grad=True)
            i_v_sim_adj = self.get_knn_adj_mat(self.image_embedding.weight, self.item_knn_k, self.device)
            u_v_sim_adj = self.get_knn_adj_mat(self.user_v_interest, self.user_knn_k, self.device)
            self.i_mm_adj = i_v_sim_adj.to(self.device)
            self.u_mm_adj = u_v_sim_adj.to(self.device)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.user_t_interest = torch.sparse.mm(sp_inter_m, self.t_feat) / torch.sparse.sum(sp_inter_m, [1]).unsqueeze(dim=1).to_dense()
            self.user_t_prefer = torch.nn.Parameter(self.user_t_interest.to(self.device), requires_grad=True)
            i_t_sim_adj = self.get_knn_adj_mat(self.text_embedding.weight, self.item_knn_k, self.device)
            u_t_sim_adj = self.get_knn_adj_mat(self.user_t_interest, self.user_knn_k, self.device)
            self.i_mm_adj = i_t_sim_adj.to(self.device)
            self.u_mm_adj = u_t_sim_adj.to(self.device)

        if self.v_feat is not None and self.t_feat is not None:
            self.u_mm_adj = (self.u_mm_image_weight * u_v_sim_adj + (1.0 - self.u_mm_image_weight) * u_t_sim_adj).detach()
            self.i_mm_adj = (self.i_mm_image_weight * i_v_sim_adj + (1.0 - self.i_mm_image_weight) * i_t_sim_adj).detach()
            del i_t_sim_adj, i_v_sim_adj, u_t_sim_adj, u_v_sim_adj
            torch.cuda.empty_cache()
        
        self._build_topk_popularity()
        self.user_co_graph = self.topk_sample(self.n_users, self.dict_user_co_graph, self.user_knn_k, self.topK_users, self.topK_users_counts, 'softmax', self.device)
        self.item_co_graph = self.topk_sample(self.n_items, self.dict_item_co_graph, self.item_knn_k, self.topK_items, self.topK_items_counts, 'softmax', self.device)
        self.co_mm_item_graph = (self.ii_co_weight * self.item_co_graph + (1.0 - self.ii_co_weight) * self.i_mm_adj).detach()
        self.co_mm_user_graph = (self.uu_co_weight * self.user_co_graph + (1.0 - self.uu_co_weight) * self.u_mm_adj).detach()

        self.ly_norm = nn.LayerNorm(self.embedding_dim)
        self.prl = nn.PReLU().to(self.device)
        self.self_att_v = nn.MultiheadAttention(1, 1, dropout=self.s_drop, batch_first=True)
        self.self_att_t = nn.MultiheadAttention(1, 1, dropout=self.s_drop, batch_first=True)
        self.cross_att_v = nn.MultiheadAttention(1, 1, dropout=self.m_drop, batch_first=True)
        self.cross_att_t = nn.MultiheadAttention(1, 1, dropout=self.m_drop, batch_first=True)

        self.image_user_space_trans = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.text_user_space_trans = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.image_item_space_trans = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.text_item_space_trans = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.gcn_view_generator = SAV_view(self.ui_indices, in_dim=3*self.embedding_dim, device=self.device)
        self.att_view_generator = IAV_view(self.ui_indices, in_dim=3*self.embedding_dim, device=self.device)
        self.behavior_view_gengerator = PTT_view(self.ui_indices, n_b=self.n_b, in_dim=3*self.embedding_dim, device=self.device)

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_id_embedding.weight, std=0.1)
        nn.init.xavier_normal_(self.image_user_space_trans.weight)
        nn.init.xavier_normal_(self.text_user_space_trans.weight)
        nn.init.xavier_normal_(self.image_item_space_trans.weight)
        nn.init.xavier_normal_(self.text_item_space_trans.weight)
        self.fuse_w = nn.Parameter(torch.randn(self.ui_indices.shape[1]))
        self.fuse_b = nn.Parameter(torch.randn(self.ui_indices.shape[1]))


    def pre_epoch_processing(self, epoch):
        self.epoch = epoch

    def forward(self, train=False):
        user_embeds = self.user_embedding.weight
        item_embeds = self.item_id_embedding.weight 

        image_item_embeds = self.image_item_space_trans(self.image_embedding.weight)
        text_item_embeds = self.text_item_space_trans(self.text_embedding.weight)
        image_user_embeds = self.image_user_space_trans(self.user_v_prefer)
        text_user_embeds = self.text_user_space_trans(self.user_t_prefer)

        item_v_t = torch.cat((image_item_embeds, text_item_embeds), dim=-1)
        item_id_v_t = torch.cat((item_embeds, item_v_t), dim=-1)
        item_id_v_t = self.conv_homo(self.co_mm_item_graph, self.n_ii_layers, item_id_v_t)
        item_id_v_t = F.normalize(item_id_v_t)
        homo_i_id, homo_i_v_feat, homo_i_t_feat = item_id_v_t[:, :self.embedding_dim], item_id_v_t[:, self.embedding_dim:-self.embedding_dim], item_id_v_t[:, -self.embedding_dim:]

        user_v_t = torch.cat((image_user_embeds, text_user_embeds), dim=-1)
        user_id_v_t = torch.cat((user_embeds, user_v_t), dim=-1)
        user_id_v_t = self.conv_homo(self.co_mm_user_graph, self.n_uu_layers, user_id_v_t)
        user_id_v_t = F.normalize(user_id_v_t)
        homo_u_id, homo_u_v_prefer, homo_u_t_prefer = user_id_v_t[:, :self.embedding_dim], user_id_v_t[:, self.embedding_dim:-self.embedding_dim], user_id_v_t[:, -self.embedding_dim:]

        self_enhance_v_feat, _ = self.self_att_v(homo_i_v_feat.unsqueeze(2), homo_i_v_feat.unsqueeze(2), homo_i_v_feat.unsqueeze(2), need_weights=False)
        self_enhance_v_feat = self.ly_norm(homo_i_v_feat + self_enhance_v_feat.squeeze())
        self_enhance_v_feat = self.prl(self_enhance_v_feat)

        self_enhance_t_feat, _ = self.self_att_t(homo_i_t_feat.unsqueeze(2), homo_i_t_feat.unsqueeze(2), homo_i_t_feat.unsqueeze(2), need_weights=False)
        self_enhance_t_feat = self.ly_norm(homo_i_t_feat + self_enhance_t_feat.squeeze())
        self_enhance_t_feat = self.prl(self_enhance_t_feat)

        cross_enhance_v_feat, _ = self.cross_att_v(self_enhance_t_feat.unsqueeze(2), self_enhance_v_feat.unsqueeze(2), self_enhance_v_feat.unsqueeze(2), need_weights=False)
        cross_enhance_v_feat = self.ly_norm(self_enhance_v_feat + cross_enhance_v_feat.squeeze())
        cross_enhance_v_feat = self.prl(cross_enhance_v_feat)

        cross_enhance_t_feat, _ = self.cross_att_t(self_enhance_v_feat.unsqueeze(2), self_enhance_t_feat.unsqueeze(2), self_enhance_t_feat.unsqueeze(2), need_weights=False)
        cross_enhance_t_feat = self.ly_norm(self_enhance_t_feat + cross_enhance_t_feat.squeeze())
        cross_enhance_t_feat = self.prl(cross_enhance_t_feat)

        user_v_prefer = self.prl(homo_u_v_prefer)  
        user_t_prefer = self.prl(homo_u_t_prefer)

        P_id_embeds = torch.cat((homo_u_id, homo_i_id), dim=0)
        P_v_embeds = torch.cat((user_v_prefer, cross_enhance_v_feat), dim=0)
        P_t_embeds = torch.cat((user_t_prefer, cross_enhance_t_feat), dim=0)
        P_full_feat_prefer = torch.cat((P_v_embeds, P_t_embeds), dim=-1)
        P_representation = torch.cat((P_id_embeds, P_full_feat_prefer), dim=-1)
        P_user_representation, P_item_representation = P_representation.split([self.n_users, self.n_items], dim=0)
        
        Ag_gcn = self.gcn_view_generator(P_user_representation, P_item_representation)
        Ag_att = self.att_view_generator(P_user_representation, P_item_representation)
        Ag_behavior = self.behavior_view_gengerator(P_user_representation, P_item_representation)

        gcn_weight = torch.tanh(self.fuse_w * Ag_gcn + self.fuse_b)
        att_weight = torch.tanh(self.fuse_w * Ag_att + self.fuse_b)
        behavior_weight = torch.tanh(self.fuse_w * Ag_behavior + self.fuse_b)

        weight = torch.stack([behavior_weight, gcn_weight, att_weight])
        weight = torch.softmax(weight, dim=1) 

        Ag = weight[0,:] * Ag_behavior + weight[1,:] * Ag_gcn + weight[2,:] * Ag_att

        Ag_gcn = Ag_gcn - Ag
        Ag_att = Ag_att - Ag
        Ag_behavior = Ag_behavior - Ag
        Ag = (Ag + Ag_behavior + Ag_gcn + Ag_att) / 4

        src, dst = self.ui_indices[0], self.ui_indices[1]
        x_u, x_i = P_user_representation[src], P_item_representation[dst]
        edge_logits = torch.mul(x_u, x_i).sum(1)
        pre_matrix = torch.sigmoid(edge_logits).squeeze()
        batch_aug_edge_weight = pre_matrix * Ag

        weight = batch_aug_edge_weight.detach()
        aug_adj = self.new_graph(self.ui_indices, weight)
        aug_adj = aug_adj * self.norm_adj

        multi_view_id_embeds = self.conv_hete(aug_adj, homo_u_id, homo_i_id)
        multi_view_v_embeds = self.conv_hete(aug_adj, user_v_prefer, cross_enhance_v_feat)
        multi_view_t_embeds = self.conv_hete(aug_adj, user_t_prefer, cross_enhance_t_feat)

        final_id_embeds = self.conv_hete(self.norm_adj, homo_u_id, homo_i_id)
        final_v_embeds = self.conv_hete(self.norm_adj, user_v_prefer, cross_enhance_v_feat)
        final_t_embeds = self.conv_hete(self.norm_adj, user_t_prefer, cross_enhance_t_feat)

        final_id_embeds = self.prl(final_id_embeds) + final_id_embeds
        final_v_embeds = self.prl(final_v_embeds) + final_id_embeds
        final_t_embeds = self.prl(final_t_embeds) + final_id_embeds
        tmp_full_feat_prefer = torch.cat((final_v_embeds, final_t_embeds), dim=-1)
        representation = torch.cat((final_id_embeds, tmp_full_feat_prefer), dim=-1)

        if train:
            return representation, (final_id_embeds, final_v_embeds, final_t_embeds), (multi_view_id_embeds, multi_view_v_embeds, multi_view_t_embeds)
        else:
            return representation

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        all_embeds, (final_id_embeds, final_v_embeds, final_t_embeds), multi_view_embeds = self.forward(train=True)
        user_embeddings, item_embeddings = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)
        multi_view_id_embeds, multi_view_v_embeds, multi_view_t_embeds = multi_view_embeds
  
        final_user_v_embeds, final_item_v_embeds = torch.split(final_v_embeds, [self.n_users, self.n_items], dim=0)
        final_user_t_embeds, final_item_t_embeds = torch.split(final_t_embeds, [self.n_users, self.n_items], dim=0)

        u_g_embeddings = user_embeddings[users]
        pos_g_embeddings = item_embeddings[pos_items]
        neg_g_embeddings = item_embeddings[neg_items]

        rec_loss = self.bpr_loss(u_g_embeddings, pos_g_embeddings, neg_g_embeddings)
        u_align_loss = self.InfoNCE(final_user_v_embeds[users], final_user_t_embeds[users], self.align_temp)
        i_align_loss = self.InfoNCE(final_item_v_embeds[pos_items], final_item_t_embeds[pos_items], self.align_temp)
        align_loss = self.align_loss_weight * (u_align_loss + i_align_loss)

        multi_view_loss = self.get_cl_loss(interaction, final_id_embeds, multi_view_id_embeds, self.multi_view_temp)
        multi_view_loss += self.get_cl_loss(interaction, final_v_embeds, multi_view_v_embeds, self.multi_view_temp)
        multi_view_loss += self.get_cl_loss(interaction, final_t_embeds, multi_view_t_embeds, self.multi_view_temp)
        multi_view_loss = self.multi_view_loss_weight * multi_view_loss

        perturb_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        return (rec_loss, align_loss, multi_view_loss, perturb_loss)    

    def get_cl_loss(self, interaction, final_embeds, P_final_embeds, temp):
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        final_user_embeds, final_item_embeds = torch.split(final_embeds, [self.n_users, self.n_items], dim=0)
        P_final_user_embeds, P_final_item_embeds = torch.split(P_final_embeds, [self.n_users, self.n_items], dim=0)
        user_P_cl_loss = self.InfoNCE(final_user_embeds[users], P_final_user_embeds[users], temp)
        item_P_cl_loss = self.InfoNCE(final_item_embeds[pos_items], P_final_item_embeds[pos_items], temp)
        return user_P_cl_loss + item_P_cl_loss

    def new_graph(self, edge_index, weight):
        ui = edge_index.clone().to(torch.long)
        vals = weight.clone().to(torch.float32)
        size = (self.n_nodes, self.n_nodes)
        ui_off = ui.clone()
        ui_off[1] += self.n_users
        ui_graph = torch.sparse_coo_tensor(ui_off, vals, size)
        iu_graph = torch.sparse_coo_tensor(torch.stack([ui_off[1], ui_off[0]]), vals, size)
        aug_adj = (ui_graph + iu_graph).coalesce()
        return aug_adj
    
    def full_sort_predict(self, interaction):
        user = interaction[0]
        representation = self.forward()
        u_reps, i_reps = torch.split(representation, [self.n_users, self.n_items], dim=0)
        score_mat_ui = torch.matmul(u_reps[user], i_reps.t())
        return score_mat_ui

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def torch_sparse_tensor_norm_adj(self, sim_adj, degree_adj, adj_size, device):
        """
        :param sim_adj: Tensor adjacency matrix (The value of 0 or 1 is degree normalised; the value of [0,1] is similarity normalised)
        :param degree_adj: Tensor adjacency matrix (The value of 0 or 1 is degree normalised; the value of [0,1] is similarity normalised)
        :param adj_size: Tensor size of adjacency matrix
        :param device: cpu or gpu
        :return: Laplace degree normalised adjacency matrix
        """
        row_sum = 1e-7 + torch.sparse.sum(degree_adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        col = torch.arange(adj_size[0])
        row = torch.arange(adj_size[1])
        sp_degree = torch.sparse.FloatTensor(torch.stack((col, row)).to(device), r_inv_sqrt.to(device))
        return torch.spmm((torch.spmm(sp_degree, sim_adj)), sp_degree)

    def get_knn_adj_mat(self, mm_embeddings, knn_k, device):
        context_norm = F.normalize(mm_embeddings, dim=1)
        final_sim = torch.mm(context_norm, context_norm.transpose(1, 0)).cpu()
        sim_value, knn_ind = torch.topk(final_sim, knn_k, dim=-1)
        adj_size = final_sim.size()
        indices0 = torch.arange(knn_ind.shape[0])
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        sim_adj = torch.sparse.FloatTensor(indices, sim_value.flatten(), adj_size).to(device)
        degree_adj = torch.sparse.FloatTensor(indices, torch.ones(indices.shape[1]), adj_size)
        return self.torch_sparse_tensor_norm_adj(sim_adj, degree_adj, adj_size, device)
    
    def cal_norm_laplacian(self, adj):
        indices = adj._indices()
        values = adj._values()
        row = indices[0]
        col = indices[1]
        rowsum = torch.sparse.sum(adj, dim=-1).to_dense()
        d_inv_sqrt = torch.pow(rowsum, -0.5)  
        d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)  # 防止数值不稳定
        row_inv_sqrt = d_inv_sqrt[row]
        col_inv_sqrt = d_inv_sqrt[col]
        values = values * row_inv_sqrt * col_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj.shape)

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)    
        return torch.mean(cl_loss)

    def sq_sum(self, emb):
        return 1. / 2 * (emb ** 2).sum()
    
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        return mf_loss
    
    def conv_homo(self, ii_adj, n_layers, single_modal):
        for i in range(n_layers):
            single_modal = torch.sparse.mm(ii_adj, single_modal)
        return single_modal
    
    def conv_hete(self, adj, user_embeds, item_embeds):
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)    
        return all_embeddings

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()
    
    def _build_topk_popularity(self):
        dict_train_u_i = {}
        dict_train_i_u = {}
        inter_file_path = os.path.join(self.dataset_path, f'{self.data_name}.inter')
        full_df = pd.read_csv(inter_file_path, sep='\t')
        train_df = full_df[full_df['x_label'] == 0]
        train_data = train_df[['userID', 'itemID']].values
        for edge in train_data:
            user = np.int64(edge[0])
            item = np.int64(edge[1])
            if item not in dict_train_i_u:
                dict_train_i_u[item] = set()
            dict_train_i_u[item].add(user)
        all_data = full_df[['userID', 'itemID']].values
        for edge in all_data:
            user = np.int64(edge[0])
            item = np.int64(edge[1])
            
            if user not in dict_train_u_i:
                dict_train_u_i[user] = set()
            dict_train_u_i[user].add(item)
        
        sort_item_num = sorted(dict_train_u_i.items(), key=lambda item: len(item[1]), reverse=True)
        self.topK_users = [temp[0] for temp in sort_item_num]
        self.topK_users_counts = [len(temp[1]) for temp in sort_item_num]
        
        sort_user_num = sorted(dict_train_i_u.items(), key=lambda item: len(item[1]), reverse=True)
        self.topK_items = [temp[0] for temp in sort_user_num]  
        self.topK_items_counts = [len(temp[1]) for temp in sort_user_num]


    # 返回的是一个coo稀疏矩阵，每行k个非零元素
    def topk_sample(self, n_ui, dict_graph, k, topK_ui, topK_ui_counts, aggr_mode, device):
        ui_graph_index = []
        user_weight_matrix = torch.zeros(len(dict_graph), k)
        for i in range(len(dict_graph)):
            if len(dict_graph[i][0]) < k: # 若该节点候选数 <k 且不为0 
                # 先拿到已有邻居及其权重，然后用有放回采样从已有邻居中随机补齐到k个
                if len(dict_graph[i][0]) != 0:
                    ui_graph_sample = dict_graph[i][0][:k]
                    ui_graph_weight = dict_graph[i][1][:k]
                    rand_index = np.random.randint(0, len(ui_graph_sample), size=k - len(ui_graph_sample))
                    ui_graph_sample += np.array(ui_graph_sample)[rand_index].tolist()
                    ui_graph_weight += np.array(ui_graph_weight)[rand_index].tolist()
                    ui_graph_index.append(ui_graph_sample)
                else: # 若该节点一个候选都没有，
                    ui_graph_index.append(topK_ui[:k])
                    ui_graph_weight = (np.array(topK_ui_counts[:k]) / sum(topK_ui_counts[:k])).tolist()
            else:
                # 否则直接取前k个
                ui_graph_sample = dict_graph[i][0][:k]
                ui_graph_weight = dict_graph[i][1][:k]
                ui_graph_index.append(ui_graph_sample)
            if aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(ui_graph_weight), dim=0)  # softmax
            elif aggr_mode == 'mean':
                user_weight_matrix[i] = torch.ones(k) / k  # mean
        tmp_all_row = []
        tmp_all_col = []
        for i in range(n_ui):
            row = torch.zeros(1, k) + i
            tmp_all_row += row.flatten()
            tmp_all_col += ui_graph_index[i]
        tmp_all_row = torch.tensor(tmp_all_row).to(torch.int32)
        tmp_all_col = torch.tensor(tmp_all_col).to(torch.int32)
        values = user_weight_matrix.flatten().to(device)
        indices = torch.stack((tmp_all_row, tmp_all_col)).to(device)
        return torch.sparse_coo_tensor(indices, values, (n_ui, n_ui))
    
class PTT_view(nn.Module):
    def __init__(self, edge_index, n_b, in_dim, device):
        super(PTT_view, self).__init__()
        self.edge_index = edge_index
        self.n_b = n_b
        self.E_b = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_b , in_dim)))
        self.device = device

    def forward(self, Eu, Ev):
        src, dst = self.edge_index[0], self.edge_index[1]
        r = Eu[src] * Ev[dst]                           # [E,d]
        logits = r @ self.E_b.T                         # [E,B]
        alpha = torch.softmax(logits, dim=-1)           # over prototypes [E, B] 每条边在B个原型的权重
        w = (alpha @ self.E_b)                          # [E,d] 按各个原型的权重做加权平均，得到每条边的原型向量表示
        return torch.sigmoid((r * w).sum(-1)).clamp(0,1)
    
class SAV_view(nn.Module):
    def __init__(self, edge_index, in_dim, device):
        super(SAV_view,self).__init__()
        self.edge_index = edge_index
        self.device = device

    def forward(self, Eu, Ev):
        src, dst = self.edge_index[0], self.edge_index[1]
        x_u, x_i = Eu[src], Ev[dst]
        edge_logits = torch.mul(x_u, x_i).sum(1)
        Ag = torch.sigmoid(edge_logits).squeeze()
        return Ag

class IAV_view(nn.Module):
    def __init__(self, edge_index, in_dim, device):
        super(IAV_view,self).__init__()
        self.g = nn.Parameter(torch.ones(1, in_dim))
        self.edge_index = edge_index
        self.device = device

    def forward(self, Eu, Ev):
        Xu = Eu * self.g
        Xv = Ev * self.g
        src, dst = self.edge_index[0], self.edge_index[1]
        x_u, x_i = Xu[src], Xv[dst]
        edge_logits = torch.mul(x_u, x_i).sum(1).exp()
        Ag = torch.sigmoid(edge_logits).squeeze()
        Label = torch.tensor(src, dtype=torch.int64).to(self.device)
        sum_result = scatter_sum(Ag, Label, dim=0)
        C = Ag / sum_result[Label]
        C = (C * 5).clamp(0, 1)
        return C
