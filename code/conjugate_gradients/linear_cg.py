# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:32:49 2022

@author: adayr
"""

from __init__ import *
import gpytorch
from gpytorch.utils.linear_cg_k import linear_cg_k
from gpytorch.lazy import LazyTensor as GPLazyTensor
from gpytorch.utils.pivoted_cholesky import pivoted_cholesky
import matplotlib.pyplot as plt
from typing import Union
from time import time

from Preconditioners.Preconditioners import *

import torch
import gpytorch

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
            pmvm = None,
            tol = 1e-8,
            max_its = 1000
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
        #extract some parameters
        b = b.clone()
        b_shape = b.shape
        n = b_shape[0]
        
        #ensure b is n x d array
        if len(b_shape) == 1:
            b.reshape(n, 1)
            d = 1
        else:
            d = b_shape[1]
            TypeError('We do not yet support multidimension problems')
            
        mvm = A.matmul
        
        if pmvm is None:
            pmvm = self.no_precon
        elif type(pmvm) == torch.Tensor:
            pmvm = pmvm.matmul
        elif not callable(pmvm):
            TypeError('Preconditioner is not a tensor or callable.')
        
        """
        We need to change the work vectors 
        """
        work_vectors = torch.zeros(n,5)
        
        if x_0 is not None: #do we have an initial guess?
            
            work_vectors[:,1:2] = x_0 #1:2 is current solution
            work_vectors[:,2:3] = b - mvm(x_0) #2:3 is current residual
            residual = torch.norm(work_vectors[:,2:3]).item() #residual norm
            
            work_vectors[:,4:5] = pmvm(work_vectors[:,2:3]) #4:5 is preconditioned residual
            work_vectors[:,3:4] = work_vectors[:,4:5] #3:4 current search direction
            precon_res = work_vectors[:,2:3].T @ work_vectors[:,4:5] #res.T @ precon_res
            
        else:
            
            work_vectors[:,2:3] = b
            residual = torch.norm(work_vectors[:,2:3]).item()
            work_vectors[:,4:5] = pmvm(work_vectors[:,2:3])
            work_vectors[:,3:4] = work_vectors[:,4:5]
            precon_res = work_vectors[:,2:3].T @ work_vectors[:,4:5]
            
        if max_its is None:
            max_its = 10 * n + 1
        i = 0
        while i < max_its and abs(residual) > tol:
            i += 1
            work_vectors[:,0:1] = mvm(work_vectors[:,3:4])
            alpha = precon_res / (work_vectors[:,3:4].T @ work_vectors[:,0:1]).item()
            work_vectors[:,1:2] += alpha*work_vectors[:,3:4] #x = x + alpha*d
            if i%50 == 0:
                work_vectors[:, 2:3] = b - mvm(work_vectors[:, 1:2]) #force an extra MVP to combat round off errors
            else:
                work_vectors[:, 2:3] += - alpha*work_vectors[:, 0:1] #r = r - alpha*q
            residual = torch.norm(work_vectors[:,2:3])
            work_vectors[:,4:5] = pmvm(work_vectors[:,2:3])
            new_res = (work_vectors[:,2:3].T @ work_vectors[:,4:5]).item()
            beta = new_res  / precon_res
            work_vectors[:,3:4] = work_vectors[:,4:5] + beta*work_vectors[:,3:4]
            precon_res = new_res
            
        if abs(residual) <= tol:
            return work_vectors[:,1:2], 1, i
        else:
            return work_vectors[:,1:2], 0, i
