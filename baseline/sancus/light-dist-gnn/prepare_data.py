'''
Author: fzb fzb0316@163.com
Date: 2024-03-10 15:10:52
LastEditors: fzb fzb0316@163.com
LastEditTime: 2024-12-13 20:02:11
FilePath: /light-dist-gnn/prepare_data.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Copyright 2021, Zhao CHEN
# All rights reserved.
import coo_graph
import argparse


def main():
    cached = True
    # cached = False
    # r = coo_graph.COO_Graph('ogbn-products')
    # r = coo_graph.COO_Graph('cora')
    # r = coo_graph.COO_Graph('cora', full_graph_cache_enabled=cached)
    # r = coo_graph.COO_Graph('flickr', full_graph_cache_enabled=cached)
    # r = coo_graph.COO_Graph('reddit', full_graph_cache_enabled=cached)
    # r = coo_graph.COO_Graph('ogbn-arxiv', full_graph_cache_enabled=cached)
    # r.partition(2)

    edge = ['80M', "160M", '320M']
    feature = [256, 512, 1024]
    label = [16, 32, 64]
    tr = [0.1, 0.5, 0.8]


    for e in edge:
        r = coo_graph.COO_Graph(f'e{e}_f512_l32_t0.5')
        r.partition(4)
        print(f'finish e{e}_f512_l32_t0.5')
    for f in feature:
        r = coo_graph.COO_Graph(f'e160M_f{f}_l32_t0.5')
        r.partition(4)
        print(f'finish e160M_f{f}_l32_t0.5')
    for l in label:
        r = coo_graph.COO_Graph(f'e160M_f512_l{l}_t0.5')
        r.partition(4)
        print(f'finish e160M_f512_l{l}_t0.5')
    for t in tr:
        r = coo_graph.COO_Graph(f'e160M_f512_l32_t{t}')
        r.partition(4)
        print(f'finish e160M_f512_l32_t{t}')

    return
    # for name in ['amazon-products', 'ogbn-products']:
    # for name in ['ogbn-arxiv', 'ogbn-products']:
    #     r = coo_graph.COO_Graph(name, full_graph_cache_enabled=cached)
    #     r.partition(4)
    #     r.partition(8)
    #     print(r)
    # return
    # for name in ['reddit', 'yelp', 'flickr', 'cora', 'ogbn-arxiv']:
    #     r = coo_graph.COO_Graph(name, full_graph_cache_enabled=cached)
    #     r.partition(8)
    #     r.partition(4)
    #     print(r)
    # return


if __name__ == '__main__':
    main()
