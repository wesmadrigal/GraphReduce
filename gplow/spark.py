import logging
from collections import deque

from abstract import PlowAbstract


class SparkPlow(PlowAbstract):
    '''
    Base Plow interface for plowing data
    '''
    def __init__(self, sqlCtx,
                 holdout=30,
                 env='local',
                 cutDate=None,
                 prefix='',
                 db='',
                 schema='',
                 table='', **kwargs):

        self.sqlCtx = sqlCtx
        self.holdout = holdout
        self.prefix = prefix
        self.env = env
        self.cutDate = cutDate

        self.db = db
        self.schema = schema
        self.table = table
        self.df = None

        # should all be defined in subclasses
        self.pk = None
        self.date_key = None
        self.pk_child_name = None

        # NOTE:
        # this is mostly used for dynamic
        # data like a one off name of a parent
        # primary key in a child table/relation
        for k, v in kwargs.items():
            setattr(self, k, v)


        self._peers = self.peers()
        self._children = self.children()

        # could have been passed a logger in kwargs
        if not hasattr(self, 'logger'):
            self.logger = self._setup_logging()


    def _setup_logging(self):
        print("running setup logger")
        if not hasattr(self, 'logger'):
            FORMAT = '%(asctime)s %(levelname)s: %(message)s'
            logging.basicConfig(filename="plow.log", format=FORMAT, level=logging.INFO)
            logger = logging.getLogger(__name__)
            print("setting logger")
            return logger


    def getData(self, fmt='parquet'):
        if self.df:
            return self.df
        if self.table:
            df = getattr(self.sqlCtx.read, fmt)(self.table)
        for c in df.columns:
            cname = f'{self.prefix}_{c}'
            df = df.withColumnRenamed(c, cname)
        self.df = df
        return self.df


    def join(self, child):
        '''
        Join child data back to self
        '''
        if hasattr(child, 'df') and getattr(child, 'df') is not None:

            if hasattr(child, 'parentJoinKey') and hasattr(child, 'myCustomJoinKey'):
                return self.df.join(
                        child.df,
                        on=self.df[f"{self.prefix}_{child.parentJoinKey}"] == child.df[f"{child.prefix}_{child.myCustomJoinKey}"],
                        how="left"
                        )

            if f"{child.prefix}_{self.pk_child_name}" in child.df.columns:

                return self.df.join(
                        child.df,
                        on=self.df[f"{self.prefix}_{self.pk}"] == child.df[f"{child.prefix}_{self.pk_child_name}"],
                        how='left'
                    )
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
            # the table that references the pertinent parent name
            elif f"{child.prefix}_{self.pk_child_name}" not in child.df.columns:
                if hasattr(child, 'pk_parent_name'):
                    return self.df.join(
                        child.df,
                        on=self.df[f'{self.prefix}_{self.pk}'] == child.df[f'{child.prefix}_{child.pk_parent_name}'],
                        how='left'
                    )
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
        if self.children():
            if isinstance(self.children, list):
                for c in self.children:
                    return c.recurseMap(self, fname)
        else:
            return getattr(self, fname)(*args, **kwargs)


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


                self.logger.info(f"reducing child {child.__class__.__name__} with key {reduce_key}")
                child.reduce(reduce_key)
                self.logger.info(f"joining child {child.__class__.__name__} back to parent {self.__class__.__name__}")
                self.df = self.join(child)
        else:
            self.logger.info(f"{self.__class__.__name__} had no children to reduce")

        if len(self._peers):
            self.logger.info(f"{self.__class__.__name__} had {len(self._peers)} peers")
            for peer in self._peers:
                self.logger.info(f"joining peer {peer.__class__.__name__} to head peer {self.__class__.__name__}")
                self.df = self.join(peer)

        # TODO: find a better place to put this
        if hasattr(self, 'postJoinAnnotate'):
            self.postJoinAnnotate()



    def doFilters(self):
        '''
        Implement custom business logic
        for filtering pertinent data in
        this function
        '''
        pass

    def children(self):
        pass

    def peers(self):
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

    def clipColumns(self, *args, **kwargs):
        pass

    def reduce(self, reduceKey):
        pass

    def doTransformations(self):
        '''
        Applies all transformations to the
        graph of data
        '''
        logging.info("map operations")
        self.recurseMap('getData')
        self.recurseMap('doFilters')
        self.recurseMap('doAnnotate')
        self.recurseMap('clipColumns')

        logging.info("reduce operations")
        print("reduce operations")
        deque(map(lambda x: x.reduceAndJoin(), self.depthFirst()))
        logging.info("we're done!")
        print("we're done!")

