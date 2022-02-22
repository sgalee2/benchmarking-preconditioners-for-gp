# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:40:27 2022

@author: adayr
"""

class Preconditioner:
    
    def mvp(self, v):
        """
        Performs a matrix vector product with the preconditioner matrix

        Parameters
        ----------
        v : RHS vector.

        """
        raise NotImplementedError
        
    def inv_mvp(self, v):
        """
        Performs a matrix vector product with the inverse of the preconditioner

        Parameters
        ----------
        v : RHS vector.
        
        """
        raise NotImplementedError
        
class Nystrom_(Preconditioner):
    pass

class Pivoted_Cholesky_(Preconditioner):
    pass

class FITC_(Preconditioner):
    pass

class SKI_(Preconditioner):
    pass

class Spectral(Preconditioner):
    pass
