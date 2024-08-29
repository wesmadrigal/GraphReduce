#!/usr/bin/env python
"""Storage abstractions for offloading
and checkpointing.
"""

# standard library
import typing
import pathlib

# internal
from graphreduce.enum import StorageFormatEnum, ProviderEnum, ComputeLayerEnum

# third party 
import dask.dataframe as dd
import pandas as pd
import pyspark


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
        return self._offloat_root


    def get_path (
            self,
            name: str,
            ) -> str:
        """
Get the file path for offload.
        """
        return self.prefix() + f"/{name}"


    def offload (
            self,
            df: typing.Union[dd.DataFrame, pd.DataFrame, pyspark.sql.dataframe.DataFrame],
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
                return getattr(self._compute_object.read, f"{self.storage_format.value}")(path)
            else:
                raise Exception(f"no compute object found to load path: {path}")
