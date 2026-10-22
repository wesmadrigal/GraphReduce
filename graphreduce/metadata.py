#!/usr/bin/env
"""Metadata tracking for features.
"""

class Metadata:
    def __init__ (
            self,
            graph: GraphReduce,
            storage_client: StorageClient,
            ):
        """
Constructor.
        """
        self.graph = graph
        self.storage_client = storage_client


    def log_graph (
            self
            ):
        """
Logs graph activity.
        """
        # node logging.
        # function logging.
        # execution time.
        # resources
        pass
