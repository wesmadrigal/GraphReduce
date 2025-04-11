#!/usr/bin/env python
"""Storage abstractions for offloading
and checkpointing.
"""

# standard library
import datetime
import typing
import pathlib

# internal
from graphreduce.enum import StorageFormatEnum, ProviderEnum, ComputeLayerEnum

# third party 
import dask.dataframe as dd
import pandas as pd
import pyspark
import daft


class StorageClient(object):
    def __init__ (
            self,
            provider: ProviderEnum = ProviderEnum.local,
            storage_format: StorageFormatEnum = StorageFormatEnum.csv,
            compute_layer: ComputeLayerEnum = ComputeLayerEnum.pandas,
            offload_root: str = '/tmp/graphreduce',
            compute_object: typing.Optional[pyspark.sql.SQLContext] = None,
            ):
        """
Constructor
        """
        self.provider = provider
        self.storage_format = storage_format
        self.compute_layer = compute_layer
        self._offload_root = offload_root
        self._compute_object = compute_object

        if self.provider == ProviderEnum.local:
            p = pathlib.Path(self._offload_root)
            if not p.exists():
                p.mkdir()


    def prefix (
            self
            ) -> str:
        """
Get the path prefix
        """
        if self.provider == ProviderEnum.local:
            return self._offload_root
        elif self.provider == ProviderEnum.s3:
            return f"s3://{self._offload_root}"
        elif self.provider == ProviderEnum.gcs:
            return f"gcs://{self._offload_root}"
        elif self.provider == ProviderEnum.blob:
            return f"blob://{self._offload_root}"
        elif self.provider == ProviderEnum.databricks:
            return self._offload_root
        return self._offloat_root


    def get_path (
            self,
            name: str,
            ) -> str:
        """
Get the file path for offload.
        """
        path = self.prefix() + f"/{name}"
        # TODO: Make this more elegant
        if self.provider == ProviderEnum.databricks:
            if path.find('/'):
                path = path.replace('/', '.')
            if path.find('.table'):
                path = path.replace('.table', '')

        return path


    def offload (
            self,
            df: typing.Union[dd.DataFrame, pd.DataFrame, pyspark.sql.dataframe.DataFrame, 'pyspark.sql.connect.dataframe.DataFrame'],
            name: str,
            ) -> bool:
        """
Offload implementation.
        """
        if self.compute_layer == ComputeLayerEnum.pandas:
            getattr(df, f"to_{self.storage_format.value}")(self.get_path(name), index=False)
        elif self.compute_layer == ComputeLayerEnum.dask:
            getattr(df, f"to_{self.storage_format.value}")(self.get_path(name), index=False)
        elif self.compute_layer == ComputeLayerEnum.spark:
            # cls_name = f"{df.__class__.__module__}.{df.__class__.__name__}"
            # if cls_name == 'pyspark.sql.connect.dataframe.DataFrame':
            if self.provider == ProviderEnum.databricks:
                out_path = self.get_path(name)

                if self._compute_object.catalog.tableExists(out_path):
                    # If table already exists: write to temp name, delete original, rename temp
                    timestamp = datetime.datetime.now().isoformat().replace('-', '').replace('.', '').replace(':', '')
                    temp_path = f'{out_path}_temp_{timestamp}'
                    df.write.format(self.storage_format.value).mode("append").option("mergeSchema", "true").saveAsTable(temp_path)
                    # Delete out_path (old version)
                    self._compute_object.sql(f'drop table {out_path}')
                    # Rename the temp_path to out_path
                    self._compute_object.sql(f'alter table {temp_path} rename to {out_path}')
                else:
                    df.write.format(self.storage_format.value).mode("append").option("mergeSchema", "true").saveAsTable(out_path)
            else:
                getattr(df.write, self.storage_format.value)(self.get_path(name), mode="overwrite")

        return True


    def load (
            self,
            path: str
            ):
        """
Load implementation.
        """
        if self.compute_layer == ComputeLayerEnum.pandas:
            return getattr(pd, f"read_{self.storage_format.value}")(path)
        elif self.compute_layer == ComputeLayerEnum.dask:
            return getattr(dd, f"read_{self.storage_format.value}")(path)
        elif self.compute_layer == ComputeLayerEnum.spark:
            if self._compute_object:
                cls_name = f"{self._compute_object.__class__.__module__}.{self._compute_object.__class__.__name__}"
                if cls_name == 'pyspark.sql.connect.session.SparkSession':
                    return self._compute_object.read.table(path)
                else:
                    return getattr(self._compute_object.read, f"{self.storage_format.value}")(path)
            else:
                raise Exception(f"no compute object found to load path: {path}")
