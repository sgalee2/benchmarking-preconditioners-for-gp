# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:32:49 2022

@author: adayr
"""

from __init__ import *
from gpytorch.lazy import LazyTensor as GPLazyTensor
from typing import Union

import torch


Tensor = torch.Tensor

class linear_cg(conjugate_gradients):
    """
    Linear Conjugate Gradients solves the system
    
        Av = b
        
    with initial guess x_0. We build this version of CG using PyTorch Tensors.
    """
    
    def no_precon(
            self, 
            x: Union[Tensor, GPLazyTensor]
        ):
        """
        

        Parameters
        ----------
        x : Union[Tensor, GPLazyTensor]
            System to be preconditioned.

        Returns
        -------
        x.clone()
            Returns a clone of x.

        """
        return x.clone()
    
    def __call__(
            self,
            A: Union[Tensor, GPLazyTensor],
            b: Tensor,
            x_0: Tensor = None,
            precon = None,
        ):
        """
        Parameters
        ----------
        A : Union[Tensor, GPLazyTensor]
            [N, N] matrix we want to backsolve from.
        b : Tensor
            [N x M] matrix of vectors we want to backsolve.
        x_0 : Tensor, optional
            Initial guess for solution. The default is None.
        precon : TYPE, optional
            DESCRIPTION. The default is None.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        sol : Tuple[Tensor, int]
              List of solution and number of iterations

        """
        b_shape = b.shape
        n = b_shape[0]
        
        
        
        
        if precon is None:
            precon = self.no_precon
        return precon(A)

x = torch.rand(size=[5])
print(x.reshape([5,1]))