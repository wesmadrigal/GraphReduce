import logging
from collections import deque

import datetime
import pandas as pd
import abc

from abstract import PlowAbstract


class PandasPlowBase(PlowAbstract):
    def __init__(self,
                 cutDate: datetime.datetime = datetime.datetime.now(),
                 fpath : str = '',
                 backend : str = 'local',
                 fmt : str = 'parquet',
                 prefix : str = '',
                 date_key : str = 'created',
                 pk : str = 'id',
                 pk_child_name : str = '',
                 cols : list = [],
                 train : bool = True,
                 label_period_val : int = 7,
                 label_period_unit : str = 'days',
                 training_period_val : int = 365,
                 training_period_unit : str = 'days',
                 **kwargs
                ):

        self.prefix = prefix
        # the date to cut data around (X/Y)
        self.cutDate = cutDate
        self.fpath = fpath
        self.fmt = fmt
        # should all be defined in subclasses
        self.pk = pk
        self.date_key = date_key
        self.pk_child_name = pk_child_name
        self.cols = cols
        self.train = train
        self.label_period_val = label_period_val
        self.label_period_unit = label_period_unit
        self.training_period_val = training_period_val
        self.training_period_unit = training_period_unit

        self.df = None

        # NOTE:
        # this is mostly used for dynamic
        # data like a one off name of a parent
        # primary key in a child table/relation
        for k, v in kwargs.items():
            setattr(self, k, v)


        self._peers = []
        self._children = []
        

    def getData(self):
        if hasattr(self, 'df') and getattr(self, 'df') is not None:
            return self.df
        if self.fpath:
            df = getattr(pd, f'read_{self.fmt}')(self.fpath)
        cmap = {
                c : f'{self.prefix}_{c}' 
                for c in df.columns
                }
        df = df.rename(columns=cmap)
        self.df = df
        return self.df


    def join(self, child, df=None):
        '''
        Join child data back to self
        '''

        if isinstance(df, pd.DataFrame):
            child_df = df
        else:
            child_df = child.df
        if hasattr(child, 'df') and getattr(child, 'df') is not None:
            if isinstance(df, pd.DataFrame):
                child_df = df
            if hasattr(child, 'parentJoinKey') and hasattr(child, 'myCustomJoinKey'):
                merged = self.df.merge(
                        child_df,
                        left_on=self.df[f"{self.prefix}_{child.parentJoinKey}"],
                        right_on=child_df[f"{child.prefix}_{child.myCustomJoinKey}"],
                        suffixes=('', '_dupe'),
                        how='left'
                        )
                return merged

            if f"{child.prefix}_{self.pk_child_name}" in child.df.columns:
                merged = self.df.merge(
                        child_df,
                        left_on=f"{self.prefix}_{self.pk}",
                        right_on=f"{child.prefix}_{self.pk_child_name}",
                        suffixes=('', '_dupe'),
                        how='left'
                    )
                merged = merged[[c for c in merged.columns if not c.endswith('_dupe')]]
                return merged
            # the expected naming convention of my pk is not
            # being followed...check if the child has
            # a definition for it's parent's pk due
            # to business logic programmed in
            # example:
            # parent `customers`
            # child `orders`
            # parent primary key `id`
            # child foreign key would normally follow convention `orders.customer_id`
            # however, it's possible it's something like `orders.cust_id`
            # the `self.pk_parent_name` is a reference to the key within
            # the table that references the pertinent parent
            elif f"{child.prefix}_{self.pk_child_name}" not in child.df.columns:
                if hasattr(child, 'pk_parent_name'):
                    merged = self.df.merge(
                        child_df,
                        left_on=f"{self.prefix}_{self.pk}",
                        right_on=f"{child.prefix}_{child.pk_parent_name}",
                        suffixes=('', '_dupe'),
                        how='left'
                    )
                    merged = merged[[c for c in merged.columns if not c.endswith('_dupe')]]
                    return merged
                else:
                    raise PlowKeyException(
                        f"{self.__class__.__name__} inst cannot find a join to {child.__class__.__name__}"
                    )

            else:
                raise PlowKeyException(
                    f"{self.__class__.__name__} inst cannot find a join to {child.__class__.__name__}"
                )
        else:
            raise PlowKeyException(
                f"{self.__class__.__name__} inst cannot find a join to {child.__class__.__name__}"
            )
            
    @abc.abstractmethod
    def getDeps(self) -> list:
        return []

    @abc.abstractmethod
    def peers(self) -> list:
        return []
    
    def colabbr(self, col: str) -> str:
        return f"{self.prefix}_{col}"


    def depthFirst(self):
        '''
        Returns an array of depth first
        children so we can map functions
        to them in reverse order (bottom up)
        '''
        nodes = []
        stack = [self]
        while stack:
            cur_node = stack[0]
            stack = stack[1:]
            nodes.append(cur_node)
            if isinstance(cur_node._children, list):
                for child in cur_node._children:
                    stack.insert(0, child)
            if isinstance(cur_node._peers, list):
                for peer in cur_node._peers:
                    stack.insert(0, peer)
            else:
                continue
        return list(reversed(nodes))


    def recurseMap(self, fname, *args, **kwargs):
        '''
        traverse our requirements / relations and apply
        a function...
        '''
        inst = self
        print(f"running {fname} on {inst.__class__.__name__}")
        getattr(inst, fname)(*args, **kwargs)
        if inst._children:
            if isinstance(inst._children, list):
                for _inst in inst._children:
                    _inst.recurseMap(fname, *args, **kwargs)
            else:
                inst = inst._children
                inst.recurseMap(fname, *args, **kwargs)

        if inst._peers:
            if isinstance(inst._peers, list):
                for _inst in inst._peers:
                    _inst.recurseMap(fname, *args, **kwargs)
        return None


    def recurseMap2(self, fname, *args, **kwargs):
        if self._children:
            if isinstance(self._children, list):
                for c in self._children:
                    return c.recurseMap(self, fname)
        else:
            return getattr(self, fname)(*args, **kwargs)


    def doFilters(self):
        '''
        Implement custom business logic
        for filtering pertinent data in
        this function
        '''
        pass

    def doAnnotate(self, reduceKey=None):
        '''
        Specific to each dataset
        '''
        pass

    def postJoinAnnotate(self):
        pass

    def doSliceData(self):
        pass

    def clipCols(self, *args, **kwargs):
        pass

    def reduce(self, reduceKey):
        pass
    
    def reduceAndJoin(self):
        '''
        Reduce all children and join back
        to parent.

        Optionally provide callback to apply
        on reduced/joined dataset
        '''
        if len(self._children):
            for child in self._children:
                print(f"Running reduce operations on child {child.__class__.__name__} and joining up to parent {self.__class__.__name__}")
                if hasattr(child, 'pk_parent_name'):
                    reduce_key = child.pk_parent_name
                else:
                    reduce_key = self.pk_child_name


                print(f"reducing child {child.__class__.__name__} with key {reduce_key}")
                df = child.reduce(reduce_key)
                print(f"joining child {child.__class__.__name__} back to parent {self.__class__.__name__}")
                self.df = self.join(child, df=df)
        else:
            print(f"{self.__class__.__name__} had no children to reduce")

        if len(self._peers):
            print(f"{self.__class__.__name__} had {len(self._peers)} peers")
            for peer in self._peers:
                print(f"joining peer {peer.__class__.__name__} to head peer {self.__class__.__name__}")
                self.df = self.join(peer)

        # TODO: find a better place to put this
        if hasattr(self, 'postJoinAnnotate'):
            self.postJoinAnnotate()
            
            
    def computeLabels(self):
        pass
    
    def prepForLabels(self):
        """
        Prepare the dataset for labels
        """
        if self.cutDate:
            return self.df[
                (self.df[self.colabbr(self.date_key)] > (self.cutDate))
                &
                (self.df[self.colabbr(self.date_key)] < (self.cutDate + datetime.timedelta(days=self.label_period_val)))
            ]
        else:
            return self.df[
                self.df[self.colabbr(self.date_key)] > (datetime.datetime.now() - datetime.timedelta(days=self.label_period_val))
            ]
    
    def computeLabelsAndJoin(self):
        for c in self._children:

            labeldf = c.computeLabels(c.colabbr(self.pk_child_name))
            self.df = self.join(c, df=labeldf)
    
    def doTransformations(self):
        '''
        Applies all transformations to the
        graph of data
        '''
        logging.info("map operations")
        self.recurseMap('getDeps')
        self.recurseMap('getData')
        self.recurseMap('doFilters')
        self.recurseMap('doAnnotate')
        self.recurseMap('clipCols')

        
        if self.train:
            
            
            deque(map(lambda x: x.computeLabelsAndJoin(), self.depthFirst()))

        logging.info("reduce operations")
        print("reduce operations")
        deque(map(lambda x: x.reduceAndJoin(), self.depthFirst()))
        logging.info("we're done!")
        print("we're done!")
