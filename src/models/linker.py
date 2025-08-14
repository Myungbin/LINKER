import random
from collections import defaultdict
from logging import getLogger

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import (build_knn_neighbourhood, build_sim, compute_normalized_laplacian, get_item_item_graph,
                         get_user_user_graph, precompute_data)


class LINKER(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LINKER, self).__init__(config, dataset)
        self.logger = getLogger()
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.weight_size = config["weight_size"]
        self.knn_k = config["knn_k"]
        self.lambda_coeff = config["lambda_coeff"]
        self.cf_model = config["cf_model"]
        self.n_ui_layers = config["n_ui_layers"]
        self.n_layers = config["n_layers"]
        self.reg_weight = config["reg_weight"]
        self.cl_weights = config["cl_weights"]
        self.user_k = config["user_k"]
        self.item_k = config["item_k"]
        self.modality_ratio = config["modality_ratio"]
        self.threshold = config["threshold"]
        self.generate_ratio = config["generate_ratio"]

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        self.user_id_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

        image_adj_raw = build_sim(self.image_embedding.weight.detach())
        text_adj_raw = build_sim(self.text_embedding.weight.detach())

        image_adj_raw = build_knn_neighbourhood(image_adj_raw)
        text_adj_raw = build_knn_neighbourhood(text_adj_raw)

        image_adj = compute_normalized_laplacian(image_adj_raw)
        self.image_original_adj = image_adj.cuda()

        text_adj = compute_normalized_laplacian(text_adj_raw)
        self.text_original_adj = text_adj.cuda()

        self.fusion_adj_raw = (self.modality_ratio * image_adj_raw) + ((1 - self.modality_ratio) * text_adj_raw)
        self.fusion_adj = compute_normalized_laplacian(self.fusion_adj_raw)

        self.user_user_adj = get_user_user_graph(self.interaction_matrix, top_k=self.user_k).to(self.device)
        self.item_item_adj = get_item_item_graph(self.interaction_matrix, top_k=self.item_k).to(self.device)

        self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)

        self.gate_v = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.gate_f = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())

        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embedding_dim, 1, bias=False),
        )

        self.gate_image_prefer = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.gate_text_prefer = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())

        self.softmax = nn.Softmax(dim=-1)

        self.norm_adj = self.get_adj_mat()
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        self.R_sprse_mat = self.R
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)

        self.fusion_adj_np = None
        self.user_connected_items = None
        self.precomputed_user_new_items = None
        (
            self.user_connected_items,
            self.precomputed_user_new_items,
            self.precomputed_item_probs,
            self.precomputed_num_to_add,
            self.item_tail_scores,
        ) = precompute_data(
            self.fusion_adj_raw, self.interaction_matrix, self.n_users, self.threshold, self.generate_ratio
        )

    def pre_epoch_processing(self):
        R = self.interaction_matrix.tolil()

        for user in range(self.n_users):
            if user not in self.precomputed_user_new_items or not self.precomputed_user_new_items[user]:
                continue

            new_items_list = self.precomputed_user_new_items[user]
            if len(new_items_list) > 0:
                num_to_add = self.precomputed_num_to_add[user]
                probs = self.precomputed_item_probs[user]

                num_to_add = min(num_to_add, len(new_items_list))

                selected_indices = np.random.choice(len(new_items_list), size=num_to_add, replace=False, p=probs)
                selected = [new_items_list[i] for i in selected_indices]

                for new_item, max_sim in selected:
                    R[user, new_item] = max_sim

        self.augment_adj = self.create_norm_adj_matrix(R).to(self.device)

    def create_norm_adj_matrix(self, R):
        R_torch = self.sparse_mx_to_torch_sparse_tensor(R).to(self.device)
        R_torch = R_torch.coalesce()

        total_size = self.n_users + self.n_items

        indices_upper = R_torch.indices()
        values_upper = R_torch.values()

        indices_upper = torch.stack([indices_upper[0], indices_upper[1] + self.n_users])
        indices_lower = torch.stack([indices_upper[1], indices_upper[0]])

        values_lower = values_upper.clone()

        indices = torch.cat([indices_upper, indices_lower], dim=1)
        values = torch.cat([values_upper, values_lower])

        adj_mat = torch.sparse.FloatTensor(indices, values, (total_size, total_size))

        row_sum = torch.sparse.sum(adj_mat, dim=1).to_dense()
        d_inv_sqrt = torch.pow(row_sum, -0.5) + 1e-7
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

        diag_indices = torch.arange(total_size, device=self.device)
        diag_indices = torch.stack([diag_indices, diag_indices])
        d_mat_inv_sqrt = torch.sparse.FloatTensor(diag_indices, d_inv_sqrt, (total_size, total_size))

        adj_mat = torch.sparse.mm(d_mat_inv_sqrt, adj_mat)
        norm_adj_mat = torch.sparse.mm(adj_mat, d_mat_inv_sqrt)

        return norm_adj_mat

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[: self.n_users, self.n_users :] = R
        adj_mat[self.n_users :, : self.n_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten() + 1e-7
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)

        self.R = norm_adj_mat[: self.n_users, self.n_users :]
        return norm_adj_mat.tocoo()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, train=False):
        image_feat = self.image_trs(self.image_embedding.weight)
        text_feat = self.text_trs(self.text_embedding.weight)

        # Co-occurrence View
        user_user_embeds = self.user_id_embedding.weight
        for _ in range(self.n_layers):
            user_user_embeds = torch.sparse.mm(self.user_user_adj, user_user_embeds)

        item_item_embeds = self.item_id_embedding.weight
        for _ in range(self.n_layers):
            item_item_embeds = torch.sparse.mm(self.item_item_adj, item_item_embeds)

        # Modalities View
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_feat))
        for _ in range(self.n_layers):
            image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)

        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_feat))
        for _ in range(self.n_layers):
            text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)

        fusion_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_f(image_feat * text_feat))
        for _ in range(self.n_layers):
            fusion_item_embeds = torch.sparse.mm(self.fusion_adj, fusion_item_embeds)
        fusion_user_embeds = torch.sparse.mm(self.R, fusion_item_embeds)

        # User-Item Interaction View
        ego_embeddings = torch.cat((self.user_id_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_user_embeds, content_item_embeds = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)
        fusion_embeds = torch.cat([fusion_user_embeds, fusion_item_embeds], dim=0)

        att_common = torch.cat([self.query_common(image_embeds), self.query_common(text_embeds)], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = (
            weight_common[:, 0].unsqueeze(dim=1) * image_embeds + weight_common[:, 1].unsqueeze(dim=1) * text_embeds
        )
        sep_image_embeds = image_embeds - common_embeds
        sep_text_embeds = text_embeds - common_embeds

        image_prefer = self.gate_image_prefer(all_embeddings)
        text_prefer = self.gate_text_prefer(all_embeddings)
        sep_image_embeds = torch.multiply(image_prefer, sep_image_embeds)
        sep_text_embeds = torch.multiply(text_prefer, sep_text_embeds)

        side_embeds = (sep_image_embeds + sep_text_embeds + fusion_embeds + common_embeds) / 4

        side_user_embeds, side_item_embeds = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)

        content_user_embeds = content_user_embeds + user_user_embeds
        final_user_embeddings = content_user_embeds + side_user_embeds

        content_item_embeds = content_item_embeds + item_item_embeds
        final_item_embeddings = content_item_embeds + side_item_embeds

        if train:
            return (
                final_user_embeddings,
                final_item_embeddings,
                content_user_embeds,
                content_item_embeds,
                side_user_embeds,
                side_item_embeds,
            )
        return final_user_embeddings, final_item_embeddings

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1.0 / 2 * (users**2).sum() + 1.0 / 2 * (pos_items**2).sum() + 1.0 / 2 * (neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))  # in-batch negative sampling
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def view_orthogonal_loss(self, view1, view2):
        return F.mse_loss(view1, view2)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        (
            ua_embeddings,
            ia_embeddings,
            content_user_embeds,
            content_item_embeds,
            side_user_embeds,
            side_item_embeds,
        ) = self.forward(self.augment_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )

        infoNCE_loss = self.InfoNCE(
            content_item_embeds[pos_items], side_item_embeds[pos_items], temperature=0.2
        ) + self.InfoNCE(content_user_embeds[users], side_user_embeds[users], temperature=0.2)

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + (self.cl_weights * infoNCE_loss)

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj, train=False)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
