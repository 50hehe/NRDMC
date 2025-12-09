import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
import pandas as pd
import os
from os.path import join, splitext
import yaml
import argparse

def create_co_occur_matrix(type_ui, all_edge, num_ui):
    edge_dict = defaultdict(set)
    for edge in all_edge:
        user, item = edge
        edge_dict[user].add(item) if type_ui == "user" else None
        edge_dict[item].add(user) if type_ui == "item" else None
    co_graph_matrix = torch.zeros(num_ui, num_ui)
    key_list = sorted(list(edge_dict.keys()))
    bar = tqdm(total=len(key_list))
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head+1, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            ui_head = edge_dict[head_key]
            ui_rear = edge_dict[rear_key]
            inter_len = len(ui_head.intersection(ui_rear))
            if inter_len > 0:
                co_graph_matrix[head_key][rear_key] = inter_len
                co_graph_matrix[rear_key][head_key] = inter_len
    bar.close()
    return co_graph_matrix

def create_dict_graph(co_graph_matrix, num_ui):
    dict_graph = {}
    for i in tqdm(range(num_ui)):
        num_co_ui = len(torch.nonzero(co_graph_matrix[i]))
        if num_co_ui <= 200:
            topk_ui = torch.topk(co_graph_matrix[i], num_co_ui)
            edge_list_i = topk_ui.indices.tolist()
            edge_list_j = topk_ui.values.tolist()
            edge_list = [edge_list_i, edge_list_j]
            dict_graph[i] = edge_list
        else:
            topk_ui = torch.topk(co_graph_matrix[i], 200)
            edge_list_i = topk_ui.indices.tolist()
            edge_list_j = topk_ui.values.tolist()
            edge_list = [edge_list_i, edge_list_j]
            dict_graph[i] = edge_list
    return dict_graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of dataset')
    args = parser.parse_args()
    dataset_name = args.dataset
    print(f'Generating u-u matrix for {dataset_name} ...\n')

    config = {}
    os.chdir('../src')
    cur_dir = os.getcwd()
    con_dir = os.path.join(cur_dir, 'configs')
    overall_config_file = os.path.join(con_dir, "overall.yaml")
    dataset_config_file = os.path.join(con_dir, "dataset", "{}.yaml".format(dataset_name))
    conf_files = [overall_config_file, dataset_config_file]
    
    for file in conf_files:
        if os.path.isfile(file):
            with open(file, 'r', encoding='utf-8') as f:
                tmp_d = yaml.safe_load(f)
                config.update(tmp_d)
    dataset_path = os.path.abspath(config['data_path'] + dataset_name)
    print('data path:\t', dataset_path)
    uid_field = config['USER_ID_FIELD']
    iid_field = config['ITEM_ID_FIELD']
    all_df = pd.read_csv(os.path.join(dataset_path, config['inter_file_name']), sep='\t')
    num_user = len(pd.unique(all_df[uid_field]))
    num_item = len(pd.unique(all_df[iid_field]))
    all_df = all_df[all_df['x_label'] == 0].copy()
    all_data = all_df[[uid_field, iid_field]].to_numpy()
    user_co_occ_matrix = create_co_occur_matrix("user", all_data, num_user)
    item_co_occ_matrix = create_co_occur_matrix("item", all_data, num_item)

    dict_user_co_occ_graph = create_dict_graph(user_co_occ_matrix, num_user)
    dict_item_co_occ_graph = create_dict_graph(item_co_occ_matrix, num_item)
    
    # 原配置里多半是 .npy 名称，这里把后缀替换成 .pt
    user_pt = join(dataset_path, splitext(config['dict_user_co_occ_graph_file'])[0] + ".pt")
    item_pt = join(dataset_path, splitext(config['dict_item_co_occ_graph_file'])[0] + ".pt")

    # 正确的保存顺序：先对象，后文件名
    torch.save(dict_user_co_occ_graph, user_pt)
    torch.save(dict_item_co_occ_graph, item_pt)

    print("saved:", user_pt)
    print("saved:", item_pt)
    
