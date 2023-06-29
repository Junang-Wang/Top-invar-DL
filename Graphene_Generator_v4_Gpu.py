import torch
from math import *
import numpy as np
from torchquad import Simpson,Boole, set_up_backend


###########################################################################
# Graphene superfluid stiffness generator
###########################################################################
class Graphene_Vec:
    
    def __init__(self,t,mu_0,mu_offset,B,nx,ny,Delta=1,device='cuda'):
        """
        This class aims to pre-vectorized graphene in lattice model
        t:         nearest neighbor hopping energy 2.8ev
        mu_0:      centra chemical potential (a tensor)
        mu_offset: chemical potential varying range center at mu_0
        B:         Zeeman energy varying range
        Delta:     superconducting order parameters, default: 1
        nx:        number of lattice sites in x direction
        ny:        number of lattice sites in y direction 
\\飞书.lnk
        Output vectorized tensor in the order of
        t,mu,B,kx,ky
        """
        self.t = t.reshape(t.shape[0],1,1,1,1,1).to(device)
        self.mu = (mu_0.reshape(-1,1) + mu_offset.reshape(1,-1)).reshape(1,mu_0.shape[0],mu_offset.shape[0],1,1,1).to(device)
        #self.mu[self.mu==0] = 0.01
        self.B = B.reshape(1,1,1,B.shape[0],1,1).to(device)
        self.kx = torch.linspace(0,4*torch.pi/3,nx).reshape(1,1,1,1,nx,1).to(device)
        self.ky = torch.linspace(-2*torch.pi/sqrt(3),2*torch.pi/sqrt(3),ny).reshape(1,1,1,1,1,ny).to(device)
        self.Delta = Delta
        self.ny = ny
    def vec(self):
        return self.t, self.mu, self.B, self.kx, self.ky, self.Delta
    def samplying(self):
      return self.ny
    def energy(self):
        """calculate graphene energy, return energy in the order of epsilon_plus, epsilon_minus, E_plus, E_minus"""
        self.epsilon_p = +self.t*torch.sqrt(abs(3+2*torch.cos(sqrt(3)*self.ky) + 4*torch.cos(sqrt(3)/2*self.ky)*torch.cos(3/2*self.kx)) ) + self.mu 
        self.epsilon_m = -self.t*torch.sqrt(abs(3+2*torch.cos(sqrt(3)*self.ky) + 4*torch.cos(sqrt(3)/2*self.ky)*torch.cos(3/2*self.kx)) ) + self.mu 
        #self.E_p = torch.sqrt(self.epsilon_p**2 + self.Delta**2)
        #self.E_m = torch.sqrt(self.epsilon_m**2 + self.Delta**2)
        return self.epsilon_p, self.epsilon_m
    
    def derivative_energy(self):
        """
        calculate derivative of epsilon with respect to kx, ky
        """
        self.epsilon_0 = self.t*torch.sqrt(abs(3+2*torch.cos(sqrt(3)*self.ky) + 4*torch.cos(sqrt(3)/2*self.ky)*torch.cos(3/2*self.kx)) ) 
        #derivative_kx =  self.t**2 * (3*torch.cos(sqrt(3)/2*self.ky)*torch.sin(3/2*self.kx) )
        #derivative_ky =  self.t**2 * sqrt(3) * torch.sin(sqrt(3)/2*self.ky) *(torch.cos(3/2*self.kx) + 2*torch.cos(sqrt(3)/2*self.ky) )

        epsilon_p_kx = - torch.nan_to_num((self.t**2 * (3*torch.cos(sqrt(3)/2*self.ky)*torch.sin(3/2*self.kx) ))/self.epsilon_0,nan=0,posinf=0,neginf=0)
        epsilon_p_ky = - torch.nan_to_num(( self.t**2 * sqrt(3) * torch.sin(sqrt(3)/2*self.ky) *(torch.cos(3/2*self.kx) + 2*torch.cos(sqrt(3)/2*self.ky) ))/self.epsilon_0,nan=0,posinf=0,neginf=0)
        #epsilon_m_kx = - epsilon_p_kx
        #epsilon_m_ky = - epsilon_p_ky
        return epsilon_p_kx,epsilon_p_ky, self.epsilon_0
    
    def derivative_state(self):
        """
        compute derivative of states with respect to kx, ky
        Caveat: I omit i in the following equations
        """
        self.state_p_kx_m = torch.nan_to_num(-1/4 + self.t**2*(1.5 + 3*torch.cos(3/2*self.kx)*torch.cos(sqrt(3)/2*self.ky))/self.epsilon_0**2/2,nan=1/4,posinf=1/4,neginf=1/4)
        self.state_p_ky_m = torch.nan_to_num(self.t**2 * (sqrt(3)*torch.sin(3/2*self.kx)*torch.sin(sqrt(3)/2*self.ky))/self.epsilon_0**2/2,nan=0,posinf=0,neginf=0)
        
        return self.state_p_kx_m,self.state_p_ky_m 





class Graphene_SS:
    
    def __init__(self,Params_vec,device='cuda'):
        """
        This class aims to simulate graphene's superfluid stiffness in lattice model
        t:         nearest neighbor hopping energy 2.8ev
        mu_0:      centra chemical potential (a tensor)
        mu_offset: chemical potential varying range center at mu_0
        B:         Zeeman energy varying range
        Delta:     superconducting order parameters, default: 1
        nx:        number of lattice sites in x direction
        ny:        number of lattice sites in y direction 

        Output: total superfluid stiffness (t,mu_0,mu,B)
        """

        # self.t, self.mu, self.B, self.kx, self.ky,self.Delta = (item.to(device) if type(item)!=int else item for item in Params_vec.vec() )
        # self.epsilon_p, self.epsilon_m, self.E_p, self.E_m = (item.to(device) if type(item)!=int else item for item in Params_vec.energy() )
        # self.epsilon_p_kx, self.epsilon_p_ky, self.epsilon_m_kx, self.epsilon_m_ky,self.epsilon_0 = (item.to(device) if type(item)!=int else item for item in Params_vec.derivative_energy())
        # self.state_p_kx_m, self.state_p_ky_m = (item.to(device) if type(item)!=int else item for item in Params_vec.derivative_state()) 

        self.t, self.mu, self.B, self.kx, self.ky,self.Delta =  Params_vec.vec() 

        # self.epsilon_p, self.epsilon_m = Params_vec.energy() 

        # self.epsilon_p_kx,self.epsilon_p_ky,self.epsilon_0 =  Params_vec.derivative_energy()

        # self.state_p_kx_m,self.state_p_ky_m =  Params_vec.derivative_state()
        
        self.ny = Params_vec.samplying()

        self.device = device
    
    def intra_xx(self):
        
      D_intra_xx = ((self.integral(self.intra_xx_integrand))
             - (self.integral_delta(self.intra_xx_integrand_delta) )     ).cpu()
      torch.cuda.empty_cache()

      return D_intra_xx

    def intra_yy(self):

      D_intra_yy = ((self.integral(self.intra_yy_integrand))
            - (self.integral_delta(self.intra_yy_integrand_delta) )     ).cpu()
      torch.cuda.empty_cache()
      return D_intra_yy
    
    def inter_xx(self):
      masker_mu_neq_0= (abs(self.mu) ==0).squeeze(dim=(-1,-2))
      #masker_mu_neq_0= (abs(self.mu) <=0.005*self.t).squeeze(dim=(-1,-2)) # since our sampling is not dense enough which leads to loss the contribution of delta function, we analytically compute inter part when mu<0.005
      D_inter_xx = ((self.integral(self.inter_xx_integrand))
             + (masker_mu_neq_0*self.integral(self.inter_xx_integrand_0) ) 
             - (2*masker_mu_neq_0*self.integral_delta(self.inter_xx_integrand_0_delta) )      ).cpu()
      torch.cuda.empty_cache()

      return D_inter_xx
    def inter_yy(self):
      masker_mu_neq_0= (abs(self.mu) ==0).squeeze(dim=(-1,-2))
      D_inter_yy = ((self.integral(self.inter_yy_integrand))
              + (masker_mu_neq_0*self.integral(self.inter_yy_integrand_0) ) 
              - (2*masker_mu_neq_0*self.integral_delta(self.inter_yy_integrand_0_delta) )      ).cpu()
      
      torch.cuda.empty_cache()

      return D_inter_yy
    def integral(self,integrand):
        if self.device == 'cuda':
            set_up_backend('torch',data_type='float32')
            
        bo = Boole() # using 4th interpolation Boole's rule
        integral_value = bo.integrate(
            integrand,
            dim = 2,
            N = 2001*2001,
            integration_domain = [[0,torch.pi],[0,torch.pi]],
            #integration_domain = [[torch.pi-0.1,torch.pi+0.1],[torch.pi/3-0.1,torch.pi/3+0.1]],
            backend = 'torch',
        )
        return integral_value
    
    def integral_delta(self,integrand):
        if self.device == 'cuda':
            set_up_backend('torch',data_type='float32')
            
        bo = Boole() # using 4th interpolation Boole's rule
        integral_value = bo.integrate(
            integrand,
            dim = 1,
            N = 10001,
            #integration_domain = [[-torch.pi,torch.pi]],
            integration_domain = [[torch.pi/(3)-0.1,torch.pi/(3)+0.1]],
            backend = 'torch',
        )
        # print(integral_value)
        return integral_value
    
    def intra_xx_integrand(self,x):
        """Integrant intra_xx by using torchquad integrate"""
        u = torch.cos(x[:,1]).reshape(-1,1,1,1,1)
        v = torch.cos(x[:,0]).reshape(-1,1,1,1,1)
        n = torch.sin(x[:,0]).reshape(-1,1,1,1,1)
        t = self.t.unsqueeze(0).squeeze(-1,-2)
        mu = self.mu.unsqueeze(0).squeeze(-1,-2)
        B = self.B.unsqueeze(0).squeeze(-1,-2)
        E_p = (torch.sqrt((t*torch.sqrt(4*(u*(u+v))+1)+1*mu)**2+self.Delta**2))
        E_m = (torch.sqrt((t*torch.sqrt(4*(u*(u+v))+1)+-1*mu)**2+self.Delta**2))

        I = torch.nan_to_num(t**2*(3*u*n)**2/(abs(4*(u*(u+v))+1))*( Occupy_f(E_p,B)/E_p**3 +  Occupy_f(E_m,B)/E_m**3 )/np.pi**2/8,nan=0,posinf=0)
        return I
    def intra_xx_integrand_delta(self,x):
        """Integrate intra_xx, delta function part """
        t = self.t.unsqueeze(0).squeeze(-1,-2)
        B = self.B.unsqueeze(0).squeeze(-1,-2)
        mu = self.mu.unsqueeze(0).squeeze(-1,-2)
        # setting z_i, u(u+v); [-1/4,2]
        z_pp = (1/4*( ((-mu+ torch.sqrt(B**2-self.Delta**2))/t)**2 - 1 )).round(decimals=7) # torch.sqrt(-1) = nan
        z_pp[z_pp>2] = float('nan')
        z_pp[z_pp<=-1/4] = float('nan')
        z_pm = (1/4*( ((-mu- torch.sqrt(B**2-self.Delta**2))/t)**2 - 1 )).round(decimals=7) # torch.sqrt(-1) = nan
        z_pm[z_pm>2] = float('nan')
        z_pm[z_pm<=-1/4] = float('nan')
        z_mp = (1/4*( ((mu+ torch.sqrt(B**2-self.Delta**2))/t)**2 - 1 )).round(decimals=7) # torch.sqrt(-1) = nan
        z_mp[z_mp>2] = float('nan')
        z_mp[z_mp<=-1/4] = float('nan')
        z_mm = (1/4*( ((mu- torch.sqrt(B**2-self.Delta**2))/t)**2 - 1 )).round(decimals=7) # torch.sqrt(-1) = nan
        z_mm[z_mm>2] = float('nan')
        z_mm[z_mm<=-1/4] = float('nan')
        #----------------------------------------------
        u = torch.cos(x[:,0]).reshape(-1,1,1,1,1).round(decimals=7)
        A1 = torch.nan_to_num((u**2-(z_pp-u**2)**2).sqrt()/(4*z_pp+1).sqrt(),nan=0,posinf=0)
        A2 = torch.nan_to_num((u**2-(z_pm-u**2)**2).sqrt()/(4*z_pm+1).sqrt(),nan=0,posinf=0)
        A3 = torch.nan_to_num((u**2-(z_mp-u**2)**2).sqrt()/(4*z_mp+1).sqrt(),nan=0,posinf=0)
        A4 = torch.nan_to_num((u**2-(z_mm-u**2)**2).sqrt()/(4*z_mm+1).sqrt(),nan=0,posinf=0)
        I = self.Delta**2*torch.nan_to_num(t*( A1+A2+A3+A4)/(B**2-self.Delta**2).sqrt()/abs(B)/np.pi**2*9/4,nan=0,posinf=0)
        return I
    
    def intra_yy_integrand(self,x):
        """Integrant intra_yy by using torchquad integrate"""
        u = torch.cos(x[:,1]).reshape(-1,1,1,1,1)
        v = torch.cos(x[:,0]).reshape(-1,1,1,1,1)
        m = torch.sin(x[:,1]).reshape(-1,1,1,1,1)
        t = self.t.unsqueeze(0).squeeze(-1,-2)
        mu = self.mu.unsqueeze(0).squeeze(-1,-2)
        B = self.B.unsqueeze(0).squeeze(-1,-2)
        E_p = (torch.sqrt((t*torch.sqrt(4*(u*(u+v))+1)+1*mu)**2+self.Delta**2))
        E_m = (torch.sqrt((t*torch.sqrt(4*(u*(u+v))+1)+-1*mu)**2+self.Delta**2))

        I = torch.nan_to_num(t**2*(sqrt(3)*m*(2*u+v))**2/(abs(4*(u*(u+v))+1))*( Occupy_f(E_p,B)/E_p**3 +  Occupy_f(E_m,B)/E_m**3 )/np.pi**2/8,nan=0,posinf=0)
        return I
    def intra_yy_integrand_delta(self,x):
        """Integrate intra_yy, delta function part """
        t = self.t.unsqueeze(0).squeeze(-1,-2)
        B = self.B.unsqueeze(0).squeeze(-1,-2)
        mu = self.mu.unsqueeze(0).squeeze(-1,-2)
        # setting z_i, u(u+v); [-1/4,2]
        z_pp = (1/4*( ((-mu+ torch.sqrt(B**2-self.Delta**2))/t)**2 - 1 )).round(decimals=5) # torch.sqrt(-1) = nan
        z_pp[z_pp>2] = float('nan')
        z_pp[z_pp<=-1/4] = float('nan')
        z_pm = (1/4*( ((-mu- torch.sqrt(B**2-self.Delta**2))/t)**2 - 1 )).round(decimals=5) # torch.sqrt(-1) = nan
        z_pm[z_pm>2] = float('nan')
        z_pm[z_pm<=-1/4] = float('nan')
        z_mp = (1/4*( ((mu+ torch.sqrt(B**2-self.Delta**2))/t)**2 - 1 )).round(decimals=5) # torch.sqrt(-1) = nan
        z_mp[z_mp>2] = float('nan')
        z_mp[z_mp<=-1/4] = float('nan')
        z_mm = (1/4*( ((mu- torch.sqrt(B**2-self.Delta**2))/t)**2 - 1 )).round(decimals=5) # torch.sqrt(-1) = nan
        z_mm[z_mm>2] = float('nan')
        z_mm[z_mm<=-1/4] = float('nan')
        #----------------------------------------------
        u = torch.cos(x[:,0]).reshape(-1,1,1,1,1).round(decimals=5)
        ut = torch.nan_to_num(torch.tan(x[:,0]),nan=0,posinf=0).reshape(-1,1,1,1,1).round(decimals=5)
        A1 = torch.nan_to_num((z_pp+u**2)**2/(u**2-(z_pp-u**2)**2).sqrt()/(4*z_pp+1).sqrt(),nan=0,posinf=0)
        A2 = torch.nan_to_num((z_pm+u**2)**2/(u**2-(z_pm-u**2)**2).sqrt()/(4*z_pm+1).sqrt(),nan=0,posinf=0)
        A3 = torch.nan_to_num((z_mp+u**2)**2/(u**2-(z_mp-u**2)**2).sqrt()/(4*z_mp+1).sqrt(),nan=0,posinf=0)
        A4 = torch.nan_to_num((z_mm+u**2)**2/(u**2-(z_mm-u**2)**2).sqrt()/(4*z_mm+1).sqrt(),nan=0,posinf=0)
        I = self.Delta**2*torch.nan_to_num(t*ut**2*( A1+A2+A3+A4)/(B**2-self.Delta**2).sqrt()/abs(B)/np.pi**2*3/4,nan=0,posinf=0)
        return I
    def inter_xx_integrand(self,x):
        """Integrant inter_xx by using torchquad integrate"""
        u = torch.cos(x[:,1]).reshape(-1,1,1,1,1)
        v = torch.cos(x[:,0]).reshape(-1,1,1,1,1)
        t = self.t.unsqueeze(0).squeeze(-1,-2)
        mu = self.mu.unsqueeze(0).squeeze(-1,-2)
        B = self.B.unsqueeze(0).squeeze(-1,-2)
        E_p = (torch.sqrt((t*torch.sqrt(4*(u*(u+v))+1)+1*mu)**2+self.Delta**2))
        E_m = (torch.sqrt((t*torch.sqrt(4*(u*(u+v))+1)+-1*mu)**2+self.Delta**2))

        I = torch.nan_to_num(t*(-2*u**2+u*v+1)**2/(abs(4*(u*(u+v))+1))**1.5*( -Occupy_f(E_p,B)/E_p +  Occupy_f(E_m,B)/E_m )/2/np.pi**2/mu/4,nan=0,posinf=0)
        return I
    
    def inter_xx_integrand_0(self,x):
        """Integrate inter_xx by using torchquad integrate mu~0"""
        u = torch.cos(x[:,1]).reshape(-1,1,1,1,1)
        v = torch.cos(x[:,0]).reshape(-1,1,1,1,1)
        t = self.t.unsqueeze(0).squeeze(-1,-2)
        B = self.B.unsqueeze(0).squeeze(-1,-2)
        E_0 = (torch.sqrt((t*torch.sqrt(4*(u*(u+v))+1))**2+self.Delta**2))

        I = t**2*torch.nan_to_num((-2*u**2+u*v+1)**2/(4*(u*(u+v))+1),nan=0,posinf=0)*( Occupy_f(E_0,B)/E_0**3)/np.pi**2/4
        return I

    def inter_xx_integrand_0_delta(self,x):
        """Integrate inter_xx mu~0, delta function part """
        t = self.t.unsqueeze(0).squeeze(-1,-2)
        B = self.B.unsqueeze(0).squeeze(-1,-2)
        # setting z_i, u(u+v); [-1/4,2]
        z = (1/4*( (( torch.sqrt(B**2-self.Delta**2))/t)**2 - 1 )).round(decimals=5) # torch.sqrt(-1) = nan
        z[z>2] = float('nan')
        z[z<=-1/4] = float('nan')
        u = torch.cos(x[:,0]).reshape(-1,1,1,1,1).round(decimals=5)
        #print(torch.nan_to_num(z,nan=0))
        #print(I.shape)
        I = self.Delta**2*torch.nan_to_num(torch.nan_to_num((-3*u**2+z+1)**2,nan=0)/(4*z+1)/abs(B)/(u**2-(z-u**2)**2).sqrt()/np.pi**2/2,nan=0,posinf=0)
        return I
    
    def inter_yy_integrand(self,x):
        """Integrant inter_xx by using torchquad integrate"""
        u = torch.cos(x[:,1]).reshape(-1,1,1,1,1)
        v = torch.cos(x[:,0]).reshape(-1,1,1,1,1)
        m = torch.sin(x[:,1]).reshape(-1,1,1,1,1)
        n = torch.sin(x[:,0]).reshape(-1,1,1,1,1)
        t = self.t.unsqueeze(0).squeeze(-1,-2)
        mu = self.mu.unsqueeze(0).squeeze(-1,-2)
        B = self.B.unsqueeze(0).squeeze(-1,-2)
        E_p = (torch.sqrt((t*torch.sqrt(4*(u*(u+v))+1)+1*mu)**2+self.Delta**2))
        E_m = (torch.sqrt((t*torch.sqrt(4*(u*(u+v))+1)+-1*mu)**2+self.Delta**2))

        I = torch.nan_to_num(t*(sqrt(3)*m*n)**2/(abs(4*(u*(u+v))+1))**1.5*( -Occupy_f(E_p,B)/E_p +  Occupy_f(E_m,B)/E_m )/2/np.pi**2/mu/4,nan=0,posinf=0)
        return I 
    
    def inter_yy_integrand_0(self,x):
        """Integrate inter_xx by using torchquad integrate mu~0"""
        u = torch.cos(x[:,1]).reshape(-1,1,1,1,1)
        v = torch.cos(x[:,0]).reshape(-1,1,1,1,1)
        m = torch.sin(x[:,1]).reshape(-1,1,1,1,1)
        n = torch.sin(x[:,0]).reshape(-1,1,1,1,1)
        t = self.t.unsqueeze(0).squeeze(-1,-2)
        B = self.B.unsqueeze(0).squeeze(-1,-2)
        E_0 = (torch.sqrt((t*torch.sqrt(4*(u*(u+v))+1))**2+self.Delta**2))

        I = t**2*torch.nan_to_num((sqrt(3)*m*n)**2/(4*(u*(u+v))+1),nan=0,posinf=0)*( Occupy_f(E_0,B)/E_0**3)/np.pi**2/4
        return I

    def inter_yy_integrand_0_delta(self,x):
        """Integrate inter_xx mu~0, delta function part """
        t = self.t.unsqueeze(0).squeeze(-1,-2)
        B = self.B.unsqueeze(0).squeeze(-1,-2)
        # setting z_i, u(u+v); [-1/4,2]
        z = (1/4*( (( torch.sqrt(B**2-self.Delta**2))/t)**2 - 1 )).round(decimals=5) # torch.sqrt(-1) = nan
        z[z>2] = float('nan')
        z[z<=-1/4] = float('nan')
        u = torch.cos(x[:,0]).reshape(-1,1,1,1,1).round(decimals=5)
        m = torch.tan(x[:,0]).reshape(-1,1,1,1,1).round(decimals=5)
        #print(torch.nan_to_num(z,nan=0))
        #print(I.shape)
        I = 3*self.Delta**2*torch.nan_to_num(torch.nan_to_num(m**2,nan=0)/(4*z+1)/abs(B)*(u**2-(z-u**2)**2).sqrt()/np.pi**2/2,nan=0,posinf=0)
        return I

    def total(self):
        #D_inter_xx = self.inter()
        # D_intra_xx,D_intra_yy = self.intra()
        D_deter = (self.inter_xx()+self.intra_xx())*(self.inter_yy()+self.intra_yy())
        return D_deter
############################################################################# Lorentzian function
############################################################################
def Lorentzian(x,eta=0.01):
    """Lorentzian function is a numerical way to simulate Dirac delta function"""
    return 1/torch.pi*eta/(x**2+eta**2)

###########################################################################
# Occupy function
##########################################################################
def Occupy_f(E,B=0,device='cuda'):
    #E: eigen values shape = [W,Y,X,Kx,Ky,1,mu,2,1]
    #B: Zeeman field shape = [W,1,1,1,1,B,1,1,1]
    #-----------------------
    #Output: shape = [M,Y,X,Kx,Ky,B,mu,2,1]
    values = torch.tensor([0.5]).to(device)
    #f = torch.heaviside( torch.round(B+E,decimals=5), values= values) - torch.heaviside(torch.round(B-E,decimals=5), values=values)
    f = torch.heaviside( E-abs(B), values= values)
    return f

