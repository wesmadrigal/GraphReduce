#!/usr/bin/env python

import os

from graphreduce.node import GraphReduceNode, DynamicNode
from graphreduce.graph_reduce import GraphReduce
from graphreduce.enum import ComputeLayerEnum, PeriodUnit, StorageFormatEnum, ProviderEnum


data_path = '/'.join(os.path.abspath(__file__).split('/')[0:-1]) + '/data'


def test_node_instance():
    node = DynamicNode(
            fpath=os.path.join(data_path, 'cust.csv'),
            fmt='csv',
            prefix='cust',
            compute_layer=ComputeLayerEnum.pandas,
            date_key=None
            )
    assert isinstance(node, DynamicNode)


def test_get_data():
    node = DynamicNode(
            fpath=os.path.join(data_path, 'cust.csv'),
            fmt='csv',
            prefix='cust',
            compute_layer=ComputeLayerEnum.pandas,
            date_key=None
            )
    node.do_data()
    print(node.df)
    assert len(node.df) == 2


def test_filter_data():
    node = DynamicNode(
            fpath=os.path.join(data_path, 'cust.csv'),
            fmt='csv',
            prefix='cust',
            compute_layer=ComputeLayerEnum.pandas,
            date_key=None
            )
    node.do_data()
    node.do_filters()
    print(node.df)
    assert len(node.df) == 2


def test_multi_node():

    cust_node = DynamicNode(
            fpath=os.path.join(data_path, 'cust.csv'),
            fmt='csv',
            prefix='cust',
            date_key=None
            )

    order_node = DynamicNode(
            fpath=os.path.join(data_path, 'orders.csv'),
            fmt='csv',
            prefix='ord',
            date_key='ts'
            )

    gr = GraphReduce(
            parent_node=cust_node,
            fmt='csv',
            compute_layer=ComputeLayerEnum.pandas,
            dynamic_propagation=True,
            compute_period_val=730
            )
    gr.add_node(cust_node)
    gr.add_node(order_node)

    assert len(gr) == 2

    gr.add_entity_edge(
            parent_node=cust_node,
            relation_node=order_node,
            parent_key='id',
            relation_key='customer_id',
            relation_type='parent_child',
            reduce=True
            )

    gr.do_transformations()
    print(gr.parent_node.df)
    assert len(gr.parent_node.df) == 2



