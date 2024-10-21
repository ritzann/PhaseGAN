import numpy as np
import torch

class Propagator():
    def __init__(self,opt):
        self.E = opt.energy
        self.pxs = opt.pxs
        self.z = opt.z
        self.wavelength = 12.4 / self.E * 1e-10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fresnel_prop(self, Di):
        nz,n1,n2,layer = [torch.tensor(i,device=self.device) for i in Di.shape]
        pi,lamda,z,pxs = [torch.tensor(i,device=self.device) for i in [np.pi,self.wavelength,self.z,self.pxs]]
        fx = (torch.cat((torch.arange(0,n2/2),torch.arange(-n2/2,0)))).to(self.device)/pxs/n2
        fy = (torch.cat((torch.arange(0,n1/2),torch.arange(-n1/2,0)))).to(self.device)/pxs/n1
        fx,fy = torch.meshgrid(fx,fy)
        f2 = fx*2 + fy*2
        angle = -pi*lamda*z*f2
        H = torch.exp(1j*angle)
        Din = Di[:,:,:,0] + 1j*Di[:,:,:,1]
        d = torch.fft.ifftshift(Din)
        temp = torch.fft.fft2(d) * H
        Do = torch.fft.fftshift(torch.fft.ifft2(temp))
        Idet = torch.abs(Do * torch.conj(Do))
        return Idet
    
    def roll_n(self,X, axis, n):
        f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
        b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)

    def batch_fftshift2d(self,x):
        # Provided by PyTorchSteerablePyramid
        real, imag = torch.unbind(x, -1)
        for dim in range(1, len(real.size())):
            n_shift = real.size(dim)//2
            if real.size(dim) % 2 != 0:
                n_shift += 1  # for odd-sized images
            real = self.roll_n(real, axis=dim, n=n_shift)
            imag = self.roll_n(imag, axis=dim, n=n_shift)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

    def batch_ifftshift2d(self, x):
        real, imag = torch.unbind(x, -1)
        for dim in range(len(real.size()) - 1, 0, -1):
            real = self.roll_n(real, axis=dim, n=real.size(dim)//2)
            imag = self.roll_n(imag, axis=dim, n=imag.size(dim)//2)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

