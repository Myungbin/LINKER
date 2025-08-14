"""
Utility functions
##########################
"""

import datetime
import importlib
import random
from collections import defaultdict

import numpy as np
import torch


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y-%H-%M-%S")

    return cur


def get_model(model_name):
    r"""Automatically select model class based on model name
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """
    model_file_name = model_name.lower()
    module_path = ".".join(["models", model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer():
    return getattr(importlib.import_module("common.trainer"), "Trainer")


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r"""validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def dict2str(result_dict):
    r"""convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ""
    for metric, value in result_dict.items():
        result_str += str(metric) + ": " + "%.04f" % value + "    "
    return result_str


def build_knn_neighbourhood(adj, threshold=0.8):
    result = torch.zeros_like(adj)
    sorted_val, sorted_idx = torch.sort(adj, descending=True, dim=-1)
    for i in range(adj.size(0)):
        values = sorted_val[i]
        indices = sorted_idx[i]

        mask_high = values >= threshold
        kept_values = values[mask_high]
        kept_indices = indices[mask_high]

        result[i].scatter_(-1, kept_indices, kept_values)

    return result


def build_knn_neighbourhood(adj, topk=10):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization="none"):
    from torch_scatter import scatter_add

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == "sym":
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == "rw":
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float("inf"), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight


def get_dense_laplacian(adj, normalization="none"):
    if normalization == "sym":
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == "rw":
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.0
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == "none":
        L_norm = adj
    return L_norm


def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        row = [i[0] for i in tuple_list]
        col = [i[1] for i in tuple_list]
        i = torch.LongTensor([row, col]).to(device)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)


def get_item_item_graph(interaction_matrix, top_k=10):
    matrix = interaction_matrix.transpose().dot(interaction_matrix).tocoo()
    rows, cols, values = matrix.row, matrix.col, matrix.data
    n_items = interaction_matrix.shape[1]

    mask = rows != cols
    rows, cols, values = rows[mask], cols[mask], values[mask]

    item_degrees = np.bincount(rows, minlength=n_items)
    item_degrees = np.maximum(item_degrees, 1.0)

    norm_factor = np.sqrt(item_degrees[rows] * item_degrees[cols])
    normalized_values = values / norm_factor

    item_scores = defaultdict(list)

    for i, j, val in zip(rows, cols, normalized_values):
        item_scores[i].append((val, j))

    new_rows, new_cols, new_values = [], [], []

    for i in item_scores:
        top = sorted(item_scores[i], reverse=True)[:top_k]
        for val, j in top:
            new_rows.append(i)
            new_cols.append(j)
            new_values.append(val)

    indices = torch.LongTensor(np.vstack((new_rows, new_cols)))
    values = torch.FloatTensor(new_values)
    shape = (n_items, n_items)

    return torch.sparse.FloatTensor(indices, values, shape)


def get_user_user_graph(interaction_matrix, top_k=10):
    matrix = interaction_matrix.dot(interaction_matrix.transpose()).tocoo()
    rows, cols, values = matrix.row, matrix.col, matrix.data
    n_users = interaction_matrix.shape[0]

    mask = rows != cols
    rows, cols, values = rows[mask], cols[mask], values[mask]

    user_degrees = np.bincount(rows, minlength=n_users)
    user_degrees = np.maximum(user_degrees, 1.0)

    norm_factor = np.sqrt(user_degrees[rows] * user_degrees[cols])
    normalized_values = values / norm_factor

    user_scores = defaultdict(list)

    for i, j, val in zip(rows, cols, normalized_values):
        user_scores[i].append((val, j))

    new_rows, new_cols, new_values = [], [], []

    for i in user_scores:
        top = sorted(user_scores[i], reverse=True)[:top_k]
        for val, j in top:
            new_rows.append(i)
            new_cols.append(j)
            new_values.append(val)

    indices = torch.LongTensor(np.vstack((new_rows, new_cols)))
    values = torch.FloatTensor(new_values)
    shape = (n_users, n_users)

    return torch.sparse.FloatTensor(indices, values, shape)


def precompute_data(fusion_adj, interaction_matrix, n_users, topk, generate_ratio):
    fusion = fusion_adj
    fusion_adj_np = fusion.cpu().numpy()

    del fusion

    R = interaction_matrix.tolil()
    user_connected_items = R.rows

    # Calculate item popularity
    item_popularity = np.array(interaction_matrix.sum(axis=0)).flatten()
    item_tail_scores = 1.0 / (item_popularity + 1)

    precomputed_user_new_items = {}
    precomputed_item_probs = {}
    precomputed_num_to_add = {}

    for user in range(n_users):
        connected_items = user_connected_items[user]
        if not connected_items:
            continue

        sim_matrix = fusion_adj_np[connected_items]
        np.fill_diagonal(sim_matrix[:, connected_items], -np.inf)

        min_threshold = 1e-6
        sim_matrix[sim_matrix <= min_threshold] = -np.inf

        topk_idx = np.argpartition(sim_matrix, -topk, axis=1)[:, -topk:]
        topk_sim = np.take_along_axis(sim_matrix, topk_idx, axis=1)

        valid_mask = topk_sim > -np.inf

        new_items = dict()
        for item_indices, sim_values, valid_row in zip(topk_idx, topk_sim, valid_mask):
            valid_item_indices = item_indices[valid_row]
            valid_sim_values = sim_values[valid_row]

            for idx, sim in zip(valid_item_indices, valid_sim_values):
                if idx not in connected_items:
                    if idx not in new_items or new_items[idx] < sim:
                        new_items[idx] = sim

        new_items_list = list(new_items.items())
        precomputed_user_new_items[user] = new_items_list

        if new_items_list:
            precomputed_num_to_add[user] = max(1, int(len(connected_items) * generate_ratio))
            item_indices = [item_idx for item_idx, _ in new_items_list]
            probs = item_tail_scores[item_indices] / item_tail_scores[item_indices].sum()
            precomputed_item_probs[user] = probs

    return precomputed_user_new_items, precomputed_item_probs, precomputed_num_to_add


def precompute_data(
    fusion_adj_raw,
    interaction_matrix,
    n_users,
    threshold,
    generate_ratio,
):
    fusion_adj_np = fusion_adj_raw.cpu().numpy()
    R = interaction_matrix.tolil()
    user_connected_items = R.rows

    # Calculate item popularity
    item_popularity = np.array(interaction_matrix.sum(axis=0)).flatten()
    item_tail_scores = 1.0 / (item_popularity + 1)

    precomputed_user_new_items = {}
    precomputed_item_probs = {}
    precomputed_num_to_add = {}

    for user in range(n_users):
        connected_items = user_connected_items[user]
        if not connected_items:
            continue

        sim_matrix = fusion_adj_np[connected_items]
        np.fill_diagonal(sim_matrix[:, connected_items], -np.inf)

        sim_matrix[sim_matrix <= threshold] = -np.inf
        candidate_mask = sim_matrix > -np.inf

        new_items = dict()
        for row_idx, (item_indices, sim_values) in enumerate(
            zip(np.tile(np.arange(fusion_adj_np.shape[0]), (len(connected_items), 1)), sim_matrix)
        ):
            valid_indices = item_indices[candidate_mask[row_idx]]
            valid_sims = sim_values[candidate_mask[row_idx]]

            for idx, sim in zip(valid_indices, valid_sims):
                if idx not in connected_items:
                    if idx not in new_items or new_items[idx] < sim:
                        new_items[idx] = sim

        new_items_list = list(new_items.items())
        precomputed_user_new_items[user] = new_items_list

        if new_items_list:
            precomputed_num_to_add[user] = max(1, int(len(connected_items) * generate_ratio))
            item_indices = [item_idx for item_idx, _ in new_items_list]
            probs = item_tail_scores[item_indices] / item_tail_scores[item_indices].sum()
            precomputed_item_probs[user] = probs

    return (
        user_connected_items,
        precomputed_user_new_items,
        precomputed_item_probs,
        precomputed_num_to_add,
        item_tail_scores,
    )
