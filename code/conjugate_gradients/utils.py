# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:12:31 2022

@author: sgalee2
"""

class Common_terms:
    
    """
    A class for computation of common terms that appear in most conjugate gradients and related routines.
    """
    
    def precon(self, data):
        """
        Terms that commonly appear in GP approximations for preconditioning:
            
            K_xx + sigma ** 2 * I \approx Q_xx = UCV + sigma ** 2 * I
            
        then to compute Q_xx^{-1} v:
            
            A = sigma ** {-1} * L^{-1}V, where C = LL^T
            B = I + AA^T
            L_B = cholesky(B)
            
            then Q_xx^{-1} v = 1/sigma**2 [ v - A.T @ L_B^{-T} @ L_B^{-1} @ A @ v].
            
        This routine will compute and return all of A, B, L_B and L.
        """
        pass