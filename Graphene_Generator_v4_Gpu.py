import torch
from math import *
import numpy as np
from scipy import integrate
from numba import jit,cuda,vectorize,float64


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

        self.epsilon_p, self.epsilon_m = Params_vec.energy() 

        self.epsilon_p_kx,self.epsilon_p_ky,self.epsilon_0 =  Params_vec.derivative_energy()

        self.state_p_kx_m,self.state_p_ky_m =  Params_vec.derivative_state()
        
        self.ny = Params_vec.samplying()
    def intra(self):
        E_p = torch.sqrt(self.epsilon_p**2 + self.Delta**2)
        E_m = torch.sqrt(self.epsilon_m**2 + self.Delta**2)
        D_intra_xx = (self.Delta**2 *( ( ( self.epsilon_p_kx**2/E_p**3 )*(Occupy_f(E_p,self.B) ) + ( self.epsilon_p_kx**2/E_m**3 )*(Occupy_f(E_m,self.B) )) ).mean(dim=(-2,-1))  /2 -self.Integrated_delta_xx(E_p - abs(self.B),1) - self.Integrated_delta_xx(E_m - abs(self.B),-1) ).cpu() # the last division of 2 is for the first brillouin zone
        
        torch.cuda.empty_cache()


        D_intra_yy = (self.Delta**2 *( ( ( self.epsilon_p_ky**2/E_p**3 )*(Occupy_f(E_p,self.B) ) + ( self.epsilon_p_ky**2/E_m**3 )*(Occupy_f(E_m,self.B) )) ).mean(dim=(-2,-1))  /2 -self.Integrated_delta_yy(E_p - abs(self.B),1) - self.Integrated_delta_yy(E_m - abs(self.B),-1) ).cpu()

        torch.cuda.empty_cache()

        #D_intra_xy = (self.Delta**2 *( ( self.epsilon_p_kx*self.epsilon_p_ky/self.E_p**3 )*(Occupy_f(self.E_p,self.B)-self.E_p*Lorentzian(self.E_p - abs(self.B)) ) + ( self.epsilon_m_kx*self.epsilon_m_ky/self.E_m**3 )*(Occupy_f(self.E_m,self.B)-self.E_m*Lorentzian(self.E_m - abs(self.B)) ) ).mean(dim=(-2,-1)) /2).cpu()


        return D_intra_xx,D_intra_yy
    
    def inter(self):
        masker_mu_neq_0= abs(self.mu) < 0.005 # since our sampling is not dense enough which leads to loss the contribution of delta function, we analytically compute inter part when mu<0.005
        
        E_p = torch.sqrt(self.epsilon_p**2 + self.Delta**2)
        E_m = torch.sqrt(self.epsilon_m**2 + self.Delta**2) 
        E_0 = torch.sqrt(self.epsilon_0**2 + self.Delta**2)
        D_inter_xx = ((~masker_mu_neq_0*self.scipy_integral()) - self.inter_step()
               + (masker_mu_neq_0*4*self.Delta**2 *self.state_p_kx_m**2* 2*self.epsilon_0**2/E_0**3*Occupy_f(E_0,self.B)  ).mean(dim=(-2,-1)) /2
               - self.Integrated_inter_xx(masker_mu_neq_0)      ).cpu()

        torch.cuda.empty_cache()

        D_inter_yy = ((~masker_mu_neq_0*4*self.Delta**2 *self.state_p_ky_m**2* (  torch.nan_to_num( self.epsilon_0 / self.mu,nan=0 )*(- Occupy_f(E_p,self.B)/E_p + Occupy_f(E_m,self.B)/E_m )  )).mean(dim=(-2,-1)) /2
               + (masker_mu_neq_0*4*self.Delta**2 *self.state_p_ky_m**2* 2*self.epsilon_0**2/E_0**3*Occupy_f(E_0,self.B)  ).mean(dim=(-2,-1)) /2
               - self.Integrated_inter_yy(masker_mu_neq_0)      ).cpu()
        
        torch.cuda.empty_cache()


        #D_inter_xy = ((4*self.Delta**2 *self.state_p_kx_m*self.state_p_ky_m* ( torch.nan_to_num( ( self.epsilon_0 / self.mu )*(- Occupy_f(self.E_p,self.B)/self.E_p + Occupy_f(self.E_m,self.B)/self.E_m ),nan=0 ) + masker_mu_neq_0* 2*self.epsilon_p**2/self.E_p**3 *(Occupy_f(self.E_p,self.B) - self.E_p*Lorentzian(self.E_p - abs(self.B))) )).mean(dim=(-2,-1)) /2).cpu()

        return D_inter_xx,D_inter_yy

    def scipy_integral(self):
        """Integral inter_xx by using scipy integrate"""
        @vectorize([float64(float64,float64)])
        def step(E,B):
            if E > B:
                return 1
            elif E<B:
                return 0
            else:
                return 0.5

        @vectorize([float64(float64,float64,float64,float64,float64)])
        def f(y,x,t,mu,B):
            u = np.cos(y)
            v = np.cos(x)
            E_p = (np.sqrt((t*np.sqrt(4*(u*(u+v))+1)+1*mu)**2+self.Delta**2))
            E_m = (np.sqrt((t*np.sqrt(4*(u*(u+v))+1)+-1*mu)**2+self.Delta**2))
            if 4*(u*(u+v))+1 == 0:
                I = 0
            else: 
                I = t*(-2*u**2+u*v+1)**2/(4*(u*(u+v))+1)**1.5*( -step(E_p,B)/E_p +  step(E_m,B)/E_m )/2/np.pi**2/mu/4
            return I

        @vectorize([float64(float64,float64,float64)])
        def Integrate(t,mu,B):
            I=integrate.dblquad(f,0,2*np.pi,-np.pi,np.pi,args=(t,mu,B))[0]
            return I

        return Integrate(self.t.cpu(),self.mu.cpu(),self.B.cpu()) 
        # u = lambda y: np.cos(y)
        # v = lambda x: np.cos(x)
        # E = lambda alpha,t,mu,x,y: (np.sqrt((t*np.sqrt(4*(u(y)*(u(y)+v(x)))+1)+alpha*mu)**2+self.Delta**2))
        # def step(E,B=0):
        #   return np.heaviside(E,abs(B))
        # def f(y,x,t,mu,B):
        #     if 4*(u(y)*(u(y)+v(x)))+1 == 0:
        #         I = 0
        #     else: 
        #         I = t*(-2*u(y)**2+u(y)*v(x)+1)**2/(4*(u(y)*(u(y)+v(x)))+1)**1.5*( -step(E(1,t,mu,x,y),B)/E(1,t,mu,x,y) +  step(E(-1,t,mu,x,y),B)/E(-1,t,mu,x,y) )/2/np.pi**2/mu/4
        #     return I
        
        # I = torch.zeros(self.t.shape[0],self.mu.shape[1],self.mu.shape[2],self.B.shape[3])
        # for i in self.t.shape[0]:
        #     for j in self.mu.shape[1]:
        #         for m in self.mu.shape[2]:
        #             for n in self.B.shape[3]:
        #                 I[i,j,m,n]=integrate.dblquad(f,0,2*np.pi,-np.pi,np.pi,args=(self.t[i].item(),self.mu[0,j,m].item(),self.B[0,0,0,n].item()))[0]
        # return I
    
    def inter_step(self):
        u = torch.cos(torch.linspace(-torch.pi,0,1500)).reshape(1,1,1,1,-1,1).to('cuda')
        # upper boundary
        z_u = 1/4*( ((-abs(self.mu) - torch.sqrt(self.B**2-self.Delta**2))/self.t)**2 - 1 ) # torch.sqrt(-1) = nan
        # lower boundary
        z_l = 1/4*( ((-abs(self.mu) + torch.sqrt(self.B**2-self.Delta**2))/self.t)**2 - 1 ) # torch.sqrt(-1) = nan
        z_u[z_u>2] = float('nan')
        z_l[z_l<=-1/4] = float('nan')
        # samples
        N = 1500
        z = z_l+(z_u-z_l)/N*(torch.arange(0,N).reshape(1,1,1,1,1,N).to('cuda'))
        # setting u^2-(z_i-u^2)^2 > 0
        deno = u**2 - (z-u**2)**2
        deno[deno<0] = float('nan') 
        E_p = ((self.t*(4*z+1).sqrt()+abs(self.mu))**2 + self.Delta**2).sqrt()
        E_m = ((self.t*(4*z+1).sqrt()-abs(self.mu))**2 + self.Delta**2).sqrt()
        integral = torch.nan_to_num(self.Delta**2/8/torch.pi*self.t/abs(self.mu)*(-3*u**2+z+1)**2/deno.sqrt()/(4*z+1)**1.5*(1/E_p + 1/E_m)*2*u,nan=0,posinf=0,neginf=0).sum(dim=(-1,-2))/(~torch.isnan(deno)).sum()
        return integral


    def Integrated_inter_xx(self,masker_mu_neq_0):
        """This function Integrate D inter for mu~0"""
        u = torch.cos(torch.pi*torch.linspace(-1,0,10000)).reshape(1,1,1,1,1,-1).to('cuda')
        z, deno = self.define_z(u,0,1)
        Integrate_delta = (masker_mu_neq_0*self.Delta**2/torch.pi/2*torch.nan_to_num((-3*u**2+z+1)**2/deno.sqrt()/(4*z+1)/abs(self.B),nan=0,posinf=0) ).mean(dim=(-1,-2))
        return Integrate_delta

    def Integrated_inter_yy(self,masker_mu_neq_0):
        """This function Integrate D inter for mu~0"""
        u = torch.cos(torch.pi*torch.linspace(-1,0,10000)).reshape(1,1,1,1,1,-1).to('cuda')
        z, deno = self.define_z(u,0,1)
        Integrate_delta = (masker_mu_neq_0*3*self.Delta**2/torch.pi/2*torch.nan_to_num((1-u**2)*deno.sqrt()/u**2/(4*z+1)/abs(self.B),nan=0,posinf=0) ).mean(dim=(-1,-2))
        return Integrate_delta
    
    def Integrated_delta_xx(self,G,alpha):
        """Integrate the part including delta function, by converting kx,ky to u,z space. The function output a [t,mu0,mu,B] tensor"""
        if G.min() >= 0: return 0
        # setting u, u=cos sqrt(3)/2ky
        u = torch.cos(torch.pi*torch.linspace(-1,0,10000)).reshape(1,1,1,1,1,-1).to('cuda')
        
        # setting z_i = u(u+v); in [-1/4,2]
        # z_plus = 1/4*( ((-alpha*mu + sqrt(B**2-Delta**2))/t)**2 - 1 )
        z,deno = self.define_z(u,alpha,1)

        Integrate_delta = 9/(4*torch.pi)*(self.Delta**2*torch.nan_to_num(( self.t*deno.sqrt() )/( abs(self.B)*(4*z+1).sqrt()*abs(self.t*(4*z+1).sqrt() + alpha*self.mu) ),nan=0)).mean(dim=(-1,-2))

        # z_minus = 1/4*( ((-alpha*mu - sqrt(B**2-Delta**2))/t)**2 - 1 )
        z,deno = self.define_z(u,alpha,-1)
        
        Integrate_delta += 9/(4*torch.pi)*(self.Delta**2*torch.nan_to_num(( self.t*deno.sqrt() )/( abs(self.B)*(4*z+1).sqrt()*abs(self.t*(4*z+1).sqrt() + alpha*self.mu) ),nan=0)).mean(dim=(-1,-2))


        return Integrate_delta
    
    def Integrated_delta_yy(self,G,alpha):
        """Integrate the part including delta function, by converting kx,ky to u,z space. The function output a [t,mu0,mu,B] tensor"""
        if G.min() >= 0: return 0
        # setting u, u=cos sqrt(3)/2ky
        u = torch.cos(torch.linspace(-torch.pi,0,10000)).reshape(1,1,1,1,1,-1).to('cuda')
        
        # setting z_i = u(u+v); in [-1/4,2]
        # z_plus = 1/4*( ((-alpha*mu + sqrt(B**2-Delta**2))/t)**2 - 1 )
        z,deno = self.define_z(u,alpha,1)
        
        # extra term for u==0 (z==0), (t+alpha*mu)^2 == B^2 - Delta^2
        extra_term = 0
        if (0. in z) and type(extra_term)==int: 
          extra_term = ( ( (self.t+alpha*self.mu)**2==self.B**2 )* torch.nan_to_num(3*self.Delta**2/(2*torch.pi**2)*self.t/(abs(self.B)*abs(self.t+alpha*self.mu)) ,posinf=0 )).sum(dim=(-1,-2))
          print(extra_term.shape)
        Integrate_delta = 3/(4*torch.pi)*(self.Delta**2*torch.nan_to_num(( self.t*(z/u+u)**2*(1-u**2) )/( deno.sqrt()*abs(self.B)*(4*z+1).sqrt()*abs(self.t*(4*z+1).sqrt() + alpha*self.mu) ), nan=0,posinf=0)).mean(dim=(-1,-2)) + extra_term

        # z_minus = 1/4*( ((-alpha*mu - sqrt(B**2-Delta**2))/t)**2 - 1 )
        z,deno = self.define_z(u,alpha,-1)
        
        if (0. in z) and type(extra_term)==int:
          extra_term =  ( ( (self.t+alpha*self.mu)**2== self.B**2 )* torch.nan_to_num(3*self.Delta**2/(2*torch.pi**2)*self.t/(abs(self.B)*abs(self.t+alpha*self.mu)) ,posinf=0 )).sum(dim=(-1,-2))
          print(extra_term.shape)
        Integrate_delta += 3/(4*torch.pi)*(self.Delta**2*torch.nan_to_num(( self.t*(z/u+u)**2*(1-u**2) )/( deno.sqrt()*abs(self.B)*(4*z+1).sqrt()*abs(self.t*(4*z+1).sqrt() + alpha*self.mu) ), nan=0,posinf=0)).mean(dim=(-1,-2)) + extra_term

        return Integrate_delta
    
    def define_z(self,u,alpha,sign):
        # setting z_i, u(u+v); [-1/4,2]
        z = 1/4*( ((-alpha*self.mu + sign*torch.sqrt(self.B**2-self.Delta**2))/self.t)**2 - 1 ) # torch.sqrt(-1) = nan
        # setting |B| > Delta, -alpha*mu(+-)sqrt(B^2-Delta^2) > 0
        #z[(-alpha*self.mu + sign*torch.sqrt(self.B**2-self.Delta**2))<0] = float('nan')
        z[z>2] = float('nan')
        z[z<=-1/4] = float('nan')
        # setting u^2-(z_i-u^2)^2 > 0
        deno = u**2 - (z-u**2)**2
        deno[deno<0] = float('nan') 
        return z,deno
        

    def total(self):
        D_inter_xx,D_inter_yy = self.inter()
        D_intra_xx,D_intra_yy = self.intra()
        D_deter = (D_intra_xx+D_inter_xx)*(D_intra_yy+D_inter_yy)
        return D_inter_xx
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

