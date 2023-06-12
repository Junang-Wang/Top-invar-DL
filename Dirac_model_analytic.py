import torch

class Analytic_D:
    """This function return analytic Superfluid stiffness of Dirac model which is a [t,mu0,mu_offset,B] tensor"""
    def __init__(self,n,mu0,mu_offset,B,Delta=1,device='cuda'):
        self.n = n.reshape(n.shape[0],1,1,1).to(device)
        self.mu = (mu0.reshape(-1,1) + mu_offset.reshape(1,-1)).reshape(1,mu0.shape[0],mu_offset.shape[0],1).to(device)
        self.B = B.reshape(1,1,1,B.shape[0]).to(device)
        self.Delta = Delta
        self.device = device
    
    def intra(self):
        D_intra = self.n*(self.mu**2+self.Delta**2).sqrt()/(2*torch.pi)* ( self.step_f( self.Delta-abs(self.B) ) 
                                                                          + torch.nan_to_num(self.step_f( abs(self.B)- self.Delta)*self.step_f( (self.mu**2+self.Delta**2).sqrt() - abs(self.B) )
                                                                           *(1 - abs(self.mu)*abs(self.B) /(self.mu**2+self.Delta**2).sqrt() / (self.B**2-self.Delta**2).sqrt()  ),nan=0 ) )
        return D_intra
    
    def inter(self):
        mediate = (self.mu**2 + self.Delta**2).sqrt()
        mask = self.mu == 0

        D_inter = ~mask*torch.nan_to_num( self.n*self.Delta**2/ 2 /torch.pi / abs(self.mu)* (  self.step_f(self.Delta-abs(self.B))*torch.log((mediate+abs(self.mu))/self.Delta ) 
                                                                     + self.step_f(abs(self.B)-self.Delta)*self.step_f(mediate-abs(self.B))*torch.log((mediate+abs(self.mu))/((self.B**2-self.Delta**2).sqrt()+abs(self.B)) )),nan=0  ) \
                                                                     + mask*self.n*self.Delta/ 2 /torch.pi * self.step_f(self.Delta-abs(self.B))
        
        return D_inter
    
    def total(self):
        return self.intra() + self.inter()
    
    def trivial(self):
        """This function return SS for trivial model which is a [1,mu0,mu_offset,B] tensor """
        n = torch.ones(1,1,1,1).to(device=self.device)
        D_intra = n*(self.mu**2+self.Delta**2).sqrt()/(2*torch.pi)* ( self.step_f( self.Delta-abs(self.B) ) 
                                                                          + torch.nan_to_num(self.step_f( abs(self.B)- self.Delta)*self.step_f( (self.mu**2+self.Delta**2).sqrt() - abs(self.B) )
                                                                           *(1 - abs(self.mu)*abs(self.B) /(self.mu**2+self.Delta**2).sqrt() / (self.B**2-self.Delta**2).sqrt()  ),nan=0 ) )
        return D_intra


    def step_f(self,x):
        """Step function"""
        values = torch.tensor([0.5]).to(self.device)
        return torch.heaviside(x,values)
