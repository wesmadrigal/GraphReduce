#!/usr/bin/env python

# standard lib
import typing
import datetime

# third party
from pydantic import BaseModel, validator

# internal
from graphreduce.enum import SQLOpType


class sqlop(BaseModel):
    optype: SQLOpType
    opval: str


    def __init__ (
        self,
        *args,
        **kwargs,
    ):
        """
Constructor.
        """
        super(sqlop, self).__init__(*args, **kwargs)
        if self.opval.lower().startswith(f"{self.optype.value.lower()} "):
            raise Exception(f"{self.optype.value.lower()} cannot be present in `opval`")
