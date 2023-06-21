# GraphReduce


## Functionality
GraphReduce is an abstraction for building machine learning feature
engineering pipelines in a scalable, extensible, and production-ready way.
The library is intended to help bridge the gap between research feature
definitions and production deployment without the overhead of a full 
feature store.  Underneath the hood, GraphReduce uses graph data
structures to represent tables/files as nodes and foreign keys
as edges.

GraphReduce allows for a unified feature engineering interface
to plug & play with multiple backends: `dask`, `pandas`, and `spark` are currently supported

## Motivation
As the number of features in an ML experiment grows so does the likelihood
for duplicate, one off implementations of the same code.  This is further
exacerbated if there isn't seamless integration between R&D and deployment.
Feature stores are a good solution, but they are quite complicated to setup
and manage.  GraphReduce is a lighter weight design pattern to production ready
feature engineering pipelines.  

### Installation
```
pip install 'graphreduce@git+https://github.com/wesmadrigal/graphreduce.git'
```

## Usage

### Pandas
```python
from graphreduce.node import GraphReduceNode
from graphreduce.graph_reduce import GraphReduce

class NodeA(GraphReduceNode):
    def do_annotate(self):
        pass    

    def do_filters(self):
        pass

    def do_clip_cols(self):
        pass
    
    def do_slice_data(self):
        pass
    
    def do_post_join_annotate(self):
        import uuid
        self.df[self.colabbr('uuid')] = self.df[self.colabbr(self.pk)].apply(lambda x: str(uuid.uuid4()))
    
    def do_reduce(self, key):
        pass
    
    def do_labels(self, key):
        pass

class NodeB(GraphReduce):
    def do_annotate(self):
        pass
    
    def do_filters(self):
        pass
    
    def do_clip_cols(self):
        pass
    
    def do_slice_data(self):
        pass
    
    def do_post_join_annotate(self):
        import uuid
        self.df[self.colabbr('uuid')] = self.df[self.colabbr(self.pk)].apply(lambda x: str(uuid.uuid4()))
    
    def do_reduce(self, key):
        return self.prep_for_features().groupby(self.colabbr(reduce_key)).agg(
            **{
                self.colabbr(f'{self.pk}_counts') : pd.NamedAgg(column=self.colabbr(self.pk), aggfunc='count'),
                self.colabbr(f'{self.pk}_min') : pd.NamedAgg(column=self.colabbr(self.pk), aggfunc='min'),
                self.colabbr(f'{self.pk}_min'): pd.NamedAgg(column=self.colabbr(self.pk), aggfunc='max'),
                self.colabbr(f'{self.date_key}_min') : pd.NamedAgg(column=self.colabbr(self.date_key), aggfunc='min'),
                self.colabbr(f'{self.date_key}_max') : pd.NamedAgg(column=self.colabbr(self.date_key), aggfunc='max')
            }
        ).reset_index()
    
    def do_labels(self, key):
        pass

nodea = NodeA(fpath='nodea.parquet', fmt='parquet', date_key='ts', prefix='nodea', pk='id')
nodeb = NodeB(fpath='nodeb.parquet', fmt='parquet', date_key='created_at', prefix='nodeb', pk='id')

gr = GraphReduce(
    cut_date=datetime.datetime(2023,5,1),
    parent_node=nodea,
    compute_layer=ComputeLayerEnum.pandas
)

gr.add_entity_edge(
    parent_node=nodea,
    relation_node=nodeb,
    parent_key='id',
    relation_key='nodea_foreign_key_id',
    relation_type='parent_child',
    reduce=True
)

# plot the graph to see what compute graph will run
# note, you may have to open this file in a  browser
gr.plot_graph(fname='demo_graph.html')

# perform all transformations
gr.do_transformations()
```
