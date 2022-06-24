#!/usr/bin/env python
__version__ = '0.1.0'

import os
import sys
import abc

class PlowAbstract(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getData(self):
        return
    
    @abc.abstractmethod
    def join(self):
        return
    
    @abc.abstractmethod
    def getDeps(self):
        return []
    
    @abc.abstractmethod
    def peers(self):
        return []

    @abc.abstractmethod
    def doFilters(self):
        '''
        Implement custom business logic
        for filtering pertinent data in
        this function
        '''
        return

    @abc.abstractmethod
    def doAnnotate(self):
        '''
        Implement custom annotation functionality
        for annotating this particular data
        '''
        return

    @abc.abstractmethod
    def postJoinAnnotate(self):
        '''
        Implement custom annotation functionality
        for annotating data after joining with 
        child data
        '''
        pass

    @abc.abstractmethod
    def doSliceData(self):
        return

    @abc.abstractmethod
    def clipCols(self):
        return

    @abc.abstractmethod
    def reduce(self, reduce_key):
        pass
    
    @abc.abstractmethod
    def reduceAndJoin(self):
        pass
    
    @abc.abstractmethod
    def doTransformations(self):
        pass
