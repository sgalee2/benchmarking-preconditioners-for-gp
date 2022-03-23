# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:40:27 2022

@author: adayr
"""
import torch

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
    """
    Implements the preconditioner
    
    P = (K_xu K_uu^ (-1) K_ux + sig^2 * I)^{-1}
    
    where K_ab is a kernel function evaluated at all ordered pairs {a, b}.
    """
    
    def __init__(self, K_xu, K_uu, sigma, err = 1e-3):
        
        self.K_xu = K_xu
        self.K_uu = K_uu
        self.K_ux = self.K_xu.T
        self.sigma = sigma
        
    def mvp(self, v):
        """
        Parameters
        ----------
        v : RHS vector.

        Returns
        -------
        prod: matrix vector product Pv.
        
        Uses the Woodbury inversion lemma:
                (sig^2 * I + K_xu K_uu^{-1} K_ux)^{-1} v = sig^{-2} * v - sig^{-4} * K_xu ( I + sig^{-2} * K_ux @ K_uu @ K_xu )^{-1} K_ux v
        
        Reverts to tri_solve if sigma is near 0.

        """
        n, m = self.K_xu.shape
            
        sig_inv = float(1)/(self.sigma ** 2)
        
        if self.safe_woodbury:
            
            scaled_v = sig_inv * self.v
            
            K_ux_v = self.sigma ** 2 * self.K_ux @ self.v
            K_ux_K_uu_K_xu = self.K_ux @ self.K_uu @ self.K_xu
            
            mid_prod = torch.linalg.inv(torch.eye(m) + sig_inv * K_ux_K_uu_K_xu)
            mat_vec_prod = scaled_v - self.K_xu @ mid_prod @ K_ux_v
        else:
            
            if m < 1000:
                
                safe_inv = torch.linalg.inv(self.K_uu + 1e-8 * torch.eye(m))
                K_uu_inv_K_ux = safe_inv @ self.K_ux
                K_approx = self.K_xu @ K_uu_inv_K_ux
                
                mat_vec_prod = torch.linalg.inv(self.sigma ** 2 * torch.eye(n) + K_approx) @ v
            
            else:
                pass
            
            
        return mat_vec_prod
        

class Pivoted_Cholesky_(Preconditioner):
    """
    Implements the preconditioner
    
    P = (LL^T + sig^2 * I)^{-1},
    
    where L is the lower triangular factor of a pivoted Cholesky decomposition for our system.
    """
    
    def __init__(self, L, sigma, err = 1e-3):
        self.L = L
        self.sigma = sigma
        
        #Woodbury breaks down as sigma \to 0
        if self.sigma < err:
            self.safe_woodbury = 0
        else:
            self.safe_woodbury = 1

        
    def mvp(self, v):
        """
        Parameters
        ----------
        v : RHS vector.

        Returns
        -------
        prod : matrix vector product Pv.
        
        Uses the Woodbury inversion lemma:
            (sig^2 * I + LL^T)^{-1} v = sig^{-2} * v - sig^{-4} * L @ (I + sig^{-2} * L^T @ L)^{-1} L^T v
            
        Reverts to tri_solve if sigma is near 0.

        """
        if self.safe_woodbury:      
            
            n, k = self.L.shape
            
            sig_inv = float(1)/(self.sigma ** 2)
            
            scaled_v = sig_inv * v
            
            L_t_dot_v = sig_inv ** (2) * self.L.T @ v
            L_t_dot_L = self.L.T @ self.L
            
            mid_prod = torch.linalg.inv(torch.eye(k) + sig_inv * L_t_dot_L)
            
            mat_vec_prod = scaled_v - self.L @ mid_prod @ L_t_dot_v
            
        else:
            tri_solve = torch.triangular_solve

            Linvv = tri_solve(v, self.L, upper=False)[0]
            
            mat_vec_prod = tri_solve(Linvv, self.L.T, upper=True)[0]
            
        return mat_vec_prod

    def inv_mvp(self, v):
        
        scaled_v = self.sigma ** 2 * v
        
        Ltv = self.L.T @ v
        LLtv = self.L @ Ltv
        
        return LLtv + scaled_v

class FITC_(Preconditioner):
    pass

class SKI_(Preconditioner):
    pass

class Spectral(Preconditioner):
    pass

