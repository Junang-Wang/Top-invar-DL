import torch
from math import *


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

        Output vectorized tensor in the order of
        t,mu,B,kx,ky
        """
        self.t = t.reshape(t.shape[0],1,1,1,1,1).to(device)
        self.mu = (mu_0.reshape(-1,1) + mu_offset.reshape(1,-1)).reshape(1,mu_0.shape[0],mu_offset.shape[0],1,1,1).to(device)
        self.B = B.reshape(1,1,1,B.shape[0],1,1).to(device)
        self.kx = torch.linspace(0,4*torch.pi/3,nx).reshape(1,1,1,1,nx,1).to(device)
        self.ky = torch.linspace(-2*torch.pi/sqrt(3),2*torch.pi/sqrt(3),ny).reshape(1,1,1,1,1,ny).to(device)
        #self.kx = torch.linspace(2*torch.pi/3-0.1,2*torch.pi/3+0.1,nx).reshape(1,1,1,1,nx,1).to(device)
        #self.ky = torch.linspace(2*torch.pi/(3*sqrt(3))-0.1,2*torch.pi/(3*sqrt(3))+0.1,ny).reshape(1,1,1,1,1,ny).to(device)
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
        # epsilon_p_ky = - torch.nan_to_num(( self.t**2 * sqrt(3) * torch.sin(sqrt(3)/2*self.ky) *(torch.cos(3/2*self.kx) + 2*torch.cos(sqrt(3)/2*self.ky) ))/self.epsilon_0,nan=0,posinf=0,neginf=0)
        #epsilon_m_kx = - epsilon_p_kx
        #epsilon_m_ky = - epsilon_p_ky
        return epsilon_p_kx, self.epsilon_0
    
    def derivative_state(self):
        """
        compute derivative of states with respect to kx, ky
        Caveat: I omit i in the following equations
        """
        self.state_p_kx_m = torch.nan_to_num(-1/4 + self.t**2*(1.5 + 3*torch.cos(3/2*self.kx)*torch.cos(sqrt(3)/2*self.ky))/self.epsilon_0**2/2,nan=1/4,posinf=1/4,neginf=1/4)
        #self.state_p_ky_m = torch.nan_to_num(self.t**2 * (sqrt(3)*torch.sin(3/2*self.kx)*torch.sin(sqrt(3)/2*self.ky))/self.epsilon_0**2/2,nan=0,posinf=0,neginf=0)
        
        return self.state_p_kx_m 





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

        self.epsilon_p_kx,self.epsilon_0 =  Params_vec.derivative_energy()

        self.state_p_kx_m =  Params_vec.derivative_state()
        
        self.ny = Params_vec.samplying()
    def intra(self):
        E_p = torch.sqrt(self.epsilon_p**2 + self.Delta**2)
        E_m = torch.sqrt(self.epsilon_m**2 + self.Delta**2)
        D_intra_xx = (self.Delta**2 *( ( ( self.epsilon_p_kx**2/E_p**3 )*(Occupy_f(E_p,self.B) ) + ( self.epsilon_p_kx**2/E_m**3 )*(Occupy_f(E_m,self.B) )) ).mean(dim=(-2,-1))  /2 -self.Integrated_delta(E_p - abs(self.B),1) - self.Integrated_delta(E_m - abs(self.B),-1) ).cpu() # the last division of 2 is for the first brillouin zone
        torch.cuda.empty_cache()


        #D_intra_yy = (self.Delta**2 *( ( self.epsilon_p_ky**2/self.E_p**3 )*(Occupy_f(self.E_p,self.B)-self.E_p*Lorentzian(self.E_p - abs(self.B)) ) + ( self.epsilon_m_ky**2/self.E_m**3 )*(Occupy_f(self.E_m,self.B)-self.E_m*Lorentzian(self.E_m - abs(self.B)) ) ).mean(dim=(-2,-1)) /2).cpu()


        #D_intra_xy = (self.Delta**2 *( ( self.epsilon_p_kx*self.epsilon_p_ky/self.E_p**3 )*(Occupy_f(self.E_p,self.B)-self.E_p*Lorentzian(self.E_p - abs(self.B)) ) + ( self.epsilon_m_kx*self.epsilon_m_ky/self.E_m**3 )*(Occupy_f(self.E_m,self.B)-self.E_m*Lorentzian(self.E_m - abs(self.B)) ) ).mean(dim=(-2,-1)) /2).cpu()


        return D_intra_xx
    
    def inter(self):
        masker_mu_neq_0 = torch.zeros_like(self.mu)
        masker_mu_neq_0[self.mu == 0] = 1
        
        E_p = torch.sqrt(self.epsilon_p**2 + self.Delta**2)
        E_m = torch.sqrt(self.epsilon_m**2 + self.Delta**2) 
        D_inter_xx = ((4*self.Delta**2 *self.state_p_kx_m**2* ( torch.nan_to_num( ( self.epsilon_0 / self.mu )*(- Occupy_f(E_p,self.B)/E_p + Occupy_f(E_m,self.B)/E_m ),nan=0 )+ 0*masker_mu_neq_0* 2*self.epsilon_p**2/E_p**3 *(Occupy_f(E_p,self.B) - E_p*Lorentzian(E_p - abs(self.B)))  )).mean(dim=(-2,-1)) /2).cpu()
        torch.cuda.empty_cache()


        #D_inter_yy = ((4*self.Delta**2 *self.state_p_ky_m**2* ( torch.nan_to_num( ( self.epsilon_0 / self.mu )*(- Occupy_f(self.E_p,self.B)/self.E_p + Occupy_f(self.E_m,self.B)/self.E_m ),nan=0 ) + masker_mu_neq_0* 2*self.epsilon_0**2/self.E_p**3 *(Occupy_f(self.E_p,self.B) - self.E_p*Lorentzian(self.E_p - abs(self.B))) )).mean(dim=(-2,-1)) /2).cpu()


        #D_inter_xy = ((4*self.Delta**2 *self.state_p_kx_m*self.state_p_ky_m* ( torch.nan_to_num( ( self.epsilon_0 / self.mu )*(- Occupy_f(self.E_p,self.B)/self.E_p + Occupy_f(self.E_m,self.B)/self.E_m ),nan=0 ) + masker_mu_neq_0* 2*self.epsilon_p**2/self.E_p**3 *(Occupy_f(self.E_p,self.B) - self.E_p*Lorentzian(self.E_p - abs(self.B))) )).mean(dim=(-2,-1)) /2).cpu()

        return D_inter_xx
    
    def Integrated_delta(self,G,alpha):
        """Adopted sampling around ky where E-B ~ 0, Integrate Delta^2* dx_eps^2/E^2 delta(E-B)"""
        if G.min() >= 0: return 0
        ny = self.ny
        # find boundary for abs(G)<0.01*t, x_extre[:,:,:,:,0] is (x_max-x_min)/ny, [1] is x_min for one Dirac cone
        x_extre, y_extre = Boundary(G,self.t,torch.tensor(ny))        
        
        # print((x_extre[:,:,:,:,0:1].unsqueeze(-1)).shape)
        # print((torch.arange(0,ny).reshape(self.kx.shape)).shape)
        k_x = x_extre[:,:,:,:,0:1].unsqueeze(-1)*(torch.arange(0,ny).reshape(self.kx.shape)).to('cuda')+ x_extre[:,:,:,:,1:].unsqueeze(-1).to('cuda') 

        k_y = y_extre[:,:,:,:,0:1].unsqueeze(-2)*(torch.arange(0,ny).reshape(self.ky.shape)).to('cuda') + y_extre[:,:,:,:,1:].unsqueeze(-2).to('cuda')

        # compute energy in zoom in region        
        epsilon_0 = self.t*torch.sqrt(abs(3+2*torch.cos(sqrt(3)*k_y) + 4*torch.cos(sqrt(3)/2*k_y)*torch.cos(3/2*k_x)) )

        E_zoom =  (( epsilon_0  + alpha* self.mu  )**2 + self.Delta**2).sqrt()

        epsilon_p_kx = - torch.nan_to_num((self.t**2 * (3*torch.cos(sqrt(3)/2*k_y)*torch.sin(3/2*k_x) ))/epsilon_0,nan=0,posinf=0,neginf=0)

        eta = x_extre[:,:,:,:,0:1].unsqueeze(-1)/ny *10
        
        Integrate_delta = self.Delta**2 * ( ( epsilon_p_kx**2/E_zoom**2 )*(Lorentzian(E_zoom - abs(self.B),eta=eta) )).mean(dim=(-2,-1))*(x_extre[:,:,:,:,0])*2 # the reason of the multiplication of 2 is for the 2 Dirac cones.
        return Integrate_delta
    
    def total(self):
        D_total_xx = self.intra() + self.inter()
        print(D_total_xx.shape)
        return D_total_xx 
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
    f = torch.heaviside( B+E, values= values) - torch.heaviside(B-E, values=values)
    return f

############################################################################
# Boundary function
############################################################################
#@torch.jit.script
def Boundary(G,threshold,ny):
    """This function compute the boundary(kx,ky plane) of G<0.01*threshold"""

    # Define the range of x and y values
    x = torch.linspace(0, 4*torch.pi/3, ny)
    y = torch.linspace(-2*torch.pi/sqrt(3), 2*torch.pi/sqrt(3),ny)

    # focus on the upper Dirac point region
    X, Y = torch.meshgrid(x, y)
    mask1 = sqrt(3)*torch.abs(X) >= torch.abs(Y)
    mask2 = torch.abs(-sqrt(3)*X+4*torch.pi*sqrt(3)/3) >= torch.abs(Y)
    mask3 = Y >=0
    mask = mask1*mask2*mask3
    mask = 100*((~mask).reshape(1,1,1,1,mask.shape[0],mask.shape[1]).long().to('cuda'))
    #G = mask.reshape(1,1,1,1,mask.shape[0],mask.shape[1]).to('cuda') * G      
    

    # find the index abs(G)< threshold
    index = torch.argwhere((abs(G)+mask)<0.01*threshold).to('cpu') # index is a N by 6  matrix, each row is a sample of abs(G) including 6 indeices for t,mu0,mu,B,kx,ky
    
    x_extre = torch.zeros(G.shape[0],G.shape[1],G.shape[2],G.shape[3],2).to('cuda')
    y_extre = torch.zeros_like(x_extre)
    for i in range(index.shape[0]):
        cond = (index[:,0]==index[i,0])* (index[:,1]==index[i,1])*(index[:,2]==index[i,2])* (index[:,3]==index[i,3])
        #x_extre[index[i,0],index[i,1],index[i,2],index[i,3],:] = torch.tensor([1.,2.]).to('cuda')

        
        x_extre[index[i,0],index[i,1],index[i,2],index[i,3],:] =torch.tensor([(x[index[cond][:,4].max()]-x[index[cond][:,4].min()])/ny,x[index[cond][:,4].min()]]).to('cuda') #find the extremum of x

        y_extre[index[i,0],index[i,1],index[i,2],index[i,3]] = torch.tensor([(y[index[cond][:,5].max()]-y[index[cond][:,5].min()])/ny,y[index[cond][:,5].min()]]).to('cuda') #find the extremum of x
    return x_extre,y_extre