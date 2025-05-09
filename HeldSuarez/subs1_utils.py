#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
from torch_harmonics.quadrature import legendre_gauss_weights
import torch_harmonics.distributed as dist
import xarray as xr


# In[3]:


#
def precompute_latitudes(nlat, a=-1.0, b=1.0):
    #
    #    Convenience routine to precompute latitudes
    #
    xlg, wlg = legendre_gauss_weights(nlat, a=a, b=b)
    lats = np.flip(np.arccos(xlg)).copy()
    wlg = np.flip(wlg).copy()

    return xlg, wlg, lats


# In[4]:


#### Constants, parameters and vertical structure issues set here
####
#### The key issue here is defining delsig(k) - this is the spacing of
#### the vertical sigma levels. In this example, we have used the sigma
#### levels from the 11 level Linear Baroclinic Model (LBM), and then
#### calculate delsig(k), but one could simple specify delsig(k). Note k=kmax
#### is the bottom (lowest layer) of the model. delsig(k) is a return
#### variable
####
def bscst(kmax):
    kmaxp1 = kmax + 1
    delsig = np.zeros(kmax) 
    #
    #### This example below corresponds to the LBM
    if ( kmax == 11):
        sigLBM = [0.02075, 0.09234, 0.2025, 0.3478, 0.5133, 0.6789, 0.8146, 0.8999, 0.9499, 0.9800, 0.9950]
        siglLBM = []
        siglLBM.append(sigLBM[0]/2.0)
        [siglLBM.append((sigLBM[k] + sigLBM[k-1])/2.0 ) for k in np.arange(1, kmax, 1)]
        siglLBM.append(1.0)
        for kk in range(kmax):
            delsig[kmax - kk - 1] = siglLBM[kmax - kk - 1+1]-siglLBM[kmax - kk - 1]
    #
    #
    # This example below corresponds the CAM with 26 levels
    #
    #
    if ( kmax == 26 ):
        shybridl = np.zeros(kmaxp1)
        shybridl = [2.194, 4.895, 9.882, 18.052, 29.837, 44.623, 61.606, 78.512, 92.366, 108.664,                    127.837, 150.394, 176.930, 208.149, 244.877, 288.085, 338.917, 398.917,                    469.072, 551.839, 649.210, 744.383, 831.021, 903.300, 955.997,                    985.112, 1000.0]
        for kk in range(kmax):
            k = kmax - kk - 1
            delsig[k] = (shybridl[k+1]-shybridl[k])/1000.0
        #
    #
    sum_delsig = delsig.sum()
    delsig = delsig/sum_delsig # making sure delsig sums to 1.0
    #
    # End delsig calculation for the LBM vertical structure.
    #
    # Below is where any other choices for delsig(k) could be made
    #        delsig[k] = ???
    #  Could introduce different delsig structures here as desired
    #
    #
    #  Set mandatory sigma structure based on delsig - si, sil, sikap,
    #  slkap, cth1, cth2 are returned 
    #  variables and are used for vertical differencing
    #
    rkappa = 287.05/1005.0 #### R/Cp
    si = []
    si.append(0.0)
    [si.append(si[k] + delsig[k]) for k in range(kmax-1)]
    si.append(1.0)
    si = np.array(si)
    
    sikap = []
    [sikap.append(si[k]**rkappa) for k in range(kmax)]
    sikap.append(1.0)
    sikap = np.array(sikap)
    rk1=1.0+rkappa
    #
    slkap = np.stack([(si[k+1]**rk1-si[k]**rk1) / (rk1*(si[k+1]-si[k])) for k in range(kmax)])
    sl = np.stack([slkap[k]**(1.0/rkappa) for k in range(kmax)])
    #
    cth1 = []
    cth1.append(0.0)
    [cth1.append(sikap[k+1]/(2.0*slkap[k+1]) ) for k in range(kmax-1)]
    cth1 = np.array(cth1)
    
    cth2 = []
    [cth2.append( sikap[k+1]/(2.0*slkap[k])) for k in range(kmax-1)]
    cth2.append(0.0)
    cth2 = np.array(cth2)

    t0h = []
    t0h.append(0.0)
    [t0h.append(cth1[k+1]*300.0+cth2[k]*300.0) for k in range(kmax-1)]
    t0h.append(300.0)
    t0h = np.array(t0h)

    r1b = []
    [r1b.append(300.0-slkap[k]/sikap[k+1]*t0h[k+1]) for k in range(kmax) if k < (kmax-1)]
    r1b.append(0.0)
    r1b = np.array(r1b)
    
    r2b = []
    r2b.append(0.0)
    [r2b.append(t0h[k]*slkap[k]/sikap[k]-300.0) for k in range(kmax) if k > 0]
    r2b = np.array(r2b)
    #
    return delsig, si, sl, sikap, slkap, cth1, cth2, r1b, r2b


# In[5]:


##
## This matrix inversion is used every time step in the implicit scheme
## This computes it once and then passes it to the implicit to speed things up
##
def inv_em(dmtrx,steps_per_day,kmax,mw,zw):
    em = torch.zeros((mw,zw,kmax,kmax),dtype=torch.float64)
    dt2 =2.0*86400.0/steps_per_day
    ccc = 0.5*dt2
    cccs = ccc*ccc
    ae = 6.371E+06
    nn = torch.arange(0,mw).reshape(mw, 1).double()
    nn = nn.expand(mw,zw)
    aaa = nn * (nn + 1)/(ae*ae)
    for k in range(kmax):
        for l in range(kmax):
            em[:,:,k,l] = aaa[:,:]*cccs*dmtrx[k,l]
        for k in range(kmax):
            em[:,:,k,k] = 1.0 + em[:,:,k,k]
    qq = torch.linalg.inv(em).double()
    #                               
    return qq


# In[6]:


###
### AMTRX, CMTRX and DMTRX used to calculate geopotential
### height from the temperature and used in the semi-implicit
### time differencing.
###
def mcoeff(kmax,si,sl,slkap,r1b,r2b,delsig):
    cmtrx = np.zeros((kmax,kmax))
    #
    # local variables
    b = np.zeros((kmax,kmax))
    h = np.zeros((kmax,kmax))
    
    aa = []
    bb = []
    [aa.append(0.5*1005.0*(slkap[k]-slkap[k+1])/slkap[k+1]) for k in range(kmax-1)]
    [bb.append(0.5*1005.*(slkap[k]-slkap[k+1])/slkap[k]) for k in range(kmax-1)]
    
    lamda = []
    mu = []
    nu = []
    [lamda.append((287.05/1005.0)*300.0-(si[k]*r2b[k]+si[k+1]*r1b[k])/delsig[k]) for k in range(kmax)]
    [mu.append(lamda[k]+r1b[k]/delsig[k]) for k in range(kmax)]
    [nu.append(mu[k]+r2b[k]/delsig[k]) for k in range(kmax)]

    for k in range(kmax-1):
        h[k,k]=-1.0
        h[k,k+1]=1.0
        b[k,k]=bb[k]
        b[k,k+1]=aa[k]

    for k in range(kmax):
        h[kmax-1,k]=delsig[k]
        b[kmax-1,k]=287.05*delsig[k]

    for i in range(kmax-1):
        cmtrx[i,i]=mu[i]*delsig[i]
        for j in np.arange(i, kmax-1, 1, dtype=int):
            cmtrx[i,j+1]=lamda[i]*delsig[j+1]
        for j in np.arange(0, i+1, 1, dtype=int):
            cmtrx[i+1,j]=nu[i+1]*delsig[j]
    
    cmtrx[kmax-1,kmax-1]=mu[kmax-1]*delsig[kmax-1]
    hinv = np.linalg.inv(h)
    amtrx = hinv @ b
    cm = amtrx @ cmtrx
    am = [[287.05*300.0*delsig[l] for l in range(kmax)] for k in range(kmax)]

    dmtrx = cm + am

    return amtrx,cmtrx,dmtrx


# In[7]:


#
#
# Horizontal Diffusion del*4 
#
def diffsn(zmn1,zmn3,dmn1,dmn3,tmn1,tmn3,mw,zw):
    ae = torch.tensor(6.371E+06)
    a4 = torch.pow(ae,4)
    dkh = a4/(mw*mw*(mw+1)*(mw+1)*21*60*60)
    ekh = a4/(mw*mw*(mw+1)*(mw+1)*28*60*60)
    dkha4 = dkh/a4
    ekha4 = ekh/a4
    #
    nn = torch.arange(0,mw).reshape(mw, 1).double()
    nn = nn.expand(mw,zw)
    nn2 = nn * ( nn + 1 )
    nn4 = torch.square(nn2)
    #
    dmn3 -= dkha4*nn4*dmn1
    zmn3 -= ekha4*nn4*zmn1
    tmn3 -= ekha4*nn4*tmn1
    #
    #
    return zmn3,dmn3,tmn3
#


# In[8]:


def damp(zmn1,zmn3,dmn1,dmn3,tmn1,tmn3,qmn1,qmn3,tclim,lnpsclim,kmax,mw,zw):
    newton = torch.zeros((kmax,mw,zw),dtype=torch.float64) + 1/(20*24*60*60)
    ray = torch.zeros((kmax,mw,zw),dtype=torch.float64) + 1/(20*24*60*60)
    newton[:,:,0] = 1/(2*24*60*60) # Enhanced damping of the zonal mean
    ray[kmax-1] = 1/(2*24*60*60)
    newton[kmax-1] = 1/(2*24*60*60)
    ray[kmax-2] = 1/(5*24*60*60)
    newton[kmax-2] = 1/(5*24*60*60)
    zmn3 -= ray*zmn1
    dmn3 -= ray*dmn1
    tmn3 -= newton*(tmn1-tclim)
    qmn3 -= newton[0]*(qmn1-lnpsclim)
    #
    return zmn3,dmn3,tmn3,qmn3

#
def damp_heldsuarez(zmn1,zmn3,dmn1,dmn3,tmn1,tmn3,qmn1,qmn3,lnpsclim,kmax,mw,zw,sl):
    ray = torch.zeros((kmax,mw,zw),dtype=torch.float64)
    newton = 1.0/(20.0*24.0*60.0*60)
    slb = 0.7
    vert = (sl-slb)/(1.0-slb)
    vert = np.where(vert < 0.0,1.0/150.0,vert)*(1.0/(24.0*60.0*60.0))
    for k in range(kmax):
        ray[k] = vert[k]
    zmn3 -= ray*zmn1
    dmn3 -= ray*dmn1
    qmn3 -= newton*(qmn1-lnpsclim)
    return zmn3,dmn3,tmn3,qmn3


# In[9]:


def damp_test(zmn1,zmn3,dmn1,dmn3,tmn1,tmn3,qmn1,qmn3,tclim,lnpsclim,zclim,dclim,kmax,mw,zw):
    newton = torch.zeros((kmax,mw,zw),dtype=torch.float64) + 1/(40*24*60*60)
    ray = torch.zeros((kmax,mw,zw),dtype=torch.float64) + 1/(150*24*60*60)
    ray[:,:,0] = 1/(7*24*60*60) # Enhanced damping of the zonal mean
    newton[:,:,0] = 1/(7*24*60*60) # Enhanced damping of the zonal mean
    ray[kmax-1] = 1/(2*24*60*60)
    newton[kmax-1] = 1/(2*24*60*60)
    ray[kmax-2] = 1/(3*24*60*60)
    newton[kmax-2] = 1/(3*24*60*60)
    #
    zmn3 -= ray*(zmn1-zclim)
    dmn3 -= ray*(dmn1-dclim)
    tmn3 -= newton*(tmn1-tclim)
    qmn3 = qmn3 - newton[0]*(qmn1-lnpsclim)
    #
    return zmn3,dmn3,tmn3,qmn3


# In[10]:


def damp_prescribed_mean(zmn1,zmn3,dmn1,dmn3,tmn1,tmn3,qmn1,qmn3,kmax,mw,zw):
    newton = torch.zeros((kmax,mw,zw),dtype=torch.float64) + 1/(40*24*60*60)
    ray = torch.zeros((kmax,mw,zw),dtype=torch.float64) + 1/(150*24*60*60)
    ray[:,:,0] = 1/(3*24*60*60) # Enhanced damping of the zonal mean
    newton[:,:,0] = 1/(3*24*60*60) # Enhanced damping of the zonal mean
    ray[kmax-1] = 1/(2*24*60*60)
    newton[kmax-1] = 1/(2*24*60*60)
    ray[kmax-2] = 1/(3*24*60*60)
    newton[kmax-2] = 1/(3*24*60*60)
    #
    zmn3 -= ray*zmn1
    dmn3 -= ray*dmn1
    tmn3 -= newton*tmn1
    qmn3 = qmn3 - newton[0]*qmn1
    #
    return zmn3,dmn3,tmn3,qmn3


# In[11]:


#
# Calculate non-linear products on the Gaussian grid
# and the vertical derivatives
#
def nlprod(u,v,vort,div,temp,dxq,dyq,heat,coriolis,delsig,si,sikap,slkap,          r1b,r2b,cth1,cth2,cost_lg,kmax,imax,jmax):
    cbs = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64)
    dbs = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64)
    th = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64)
    mu2 = torch.sqrt(1.0-torch.tensor(cost_lg[:]*cost_lg[:]))
    cs = mu2[...,None].broadcast_to((jmax,imax))
    #
    # Remove mean temperature (300.0) from temp
    #
    temp = temp - 300.0
    #
    #
    # Compute c=V.del(q), cbs, dbs, cbar, dbar (AFGL Documentation)
    #
    c = torch.stack([u[k]*dxq + v[k]*dyq for k in range(kmax)])
    #
    for k in range(kmax):
        cbs[k+1]=cbs[k] + c[k]*delsig[k]
        dbs[k+1]=dbs[k] + div[k]*delsig[k]
    cbar=cbs[kmax]
    dbar=dbs[kmax]
    #
    # Compute sd = si*(cbar+dbar)-cbs-dbs
    #
    sd = torch.stack([si[k]*(cbar + dbar) - cbs[k] - dbs[k] for k in np.arange(1, kmax, 1, dtype=int)])
    sd = torch.vstack((torch.zeros((jmax,imax),dtype=torch.float64).unsqueeze(0),sd))
    sd = torch.vstack((sd,torch.zeros((jmax,imax),dtype=torch.float64).unsqueeze(0)))
    #
    # Compute th
    #
    th[0] = 0.0
    th[kmax]=temp[kmax-1]
    for k in range(kmax-1):
        th[k+1]=cth1[k+1]*temp[k+1] + cth2[k]*temp[k]
    #
    # Compute a,b,e,ut,vt - see afgl documentation
    #
    a = torch.stack([((vort[k]+coriolis)*u[k] + 287.05*temp[k]*dyq)*cs for k in range(kmax)])
    b = torch.stack([((vort[k]+coriolis)*v[k] - 287.05*temp[k]*dxq)*cs for k in range(kmax)])
    e = torch.stack([(u[k]*u[k] + v[k]*v[k])/2.0 for k in range(kmax)])
    ut = torch.stack([u[k]*temp[k]*cs for k in range(kmax)])
    vt = torch.stack([v[k]*temp[k]*cs for k in range(kmax)])
    #
    # Vertical Advection
    #
    sd2d = torch.stack([sd[k]/(2.*delsig[k]) for k in range(kmax)])
    sd2d1 = torch.stack([sd[k+1]/(2.*delsig[k]) for k in range(kmax)])

    r1p = torch.stack([temp[k]-(th[k+1]*slkap[k])/sikap[k+1] for k in range(kmax-1)])
    r1p = torch.vstack((r1p,torch.zeros((jmax,imax),dtype=torch.float64).unsqueeze(0)))
    
    sduk1 = torch.stack([sd2d1[k]*(u[k+1]-u[k])*cs for k in range(kmax-1)])
    sduk1 = torch.vstack((sduk1,torch.zeros((jmax,imax),dtype=torch.float64).unsqueeze(0)))
    
    sdvk1 = torch.stack([sd2d1[k]*(v[k+1]-v[k])*cs for k in range(kmax-1)])
    sdvk1 = torch.vstack((sdvk1,torch.zeros((jmax,imax),dtype=torch.float64).unsqueeze(0)))
    #
    #sdwk1 = torch.stack([sd2d1[k]*(w[k+1]-w[k]) for k in range(kmax-1)])
    sdwk1 = torch.stack([torch.zeros((jmax,imax),dtype=torch.float64) for k in range(kmax-1)])
    sdwk1 = torch.vstack((sdwk1,torch.zeros((jmax,imax),dtype=torch.float64).unsqueeze(0)))

    r2p = torch.stack([((th[k]*slkap[k])/sikap[k])-temp[k] for k in np.arange(1, kmax, 1, dtype=int)])
    r2p = torch.vstack((torch.zeros((jmax,imax),dtype=torch.float64).unsqueeze(0), r2p))

    sduk = torch.stack([sd2d[k]*(u[k]-u[k-1])*cs for k in np.arange(1, kmax, 1, dtype=int)])
    sduk = torch.vstack((torch.zeros((jmax,imax),dtype=torch.float64).unsqueeze(0), sduk))

    sdvk = torch.stack([sd2d[k]*(v[k]-v[k-1])*cs for k in np.arange(1, kmax, 1, dtype=int)])
    sdvk = torch.vstack((torch.zeros((jmax,imax),dtype=torch.float64).unsqueeze(0), sdvk))

    #sdwk = torch.stack([sd2d[k]*(w[k]-w[k-1]) for k in np.arange(1, kmax, 1, dtype=int)]) # no moisture equation
    sdwk = torch.stack([torch.zeros((jmax,imax),dtype=torch.float64) for k in np.arange(1, kmax, 1, dtype=int)])
    sdwk = torch.vstack((torch.zeros((jmax,imax),dtype=torch.float64).unsqueeze(0), sdwk))
    #
    # Update a, b and ri for the temperature equation
    #
    a =  torch.stack([a[k] + sdvk[k] + sdvk1[k] for k in range(kmax)])
    b =  torch.stack([b[k] - sduk[k] - sduk1[k] for k in range(kmax)])
    ri = torch.stack([temp[k]*div[k]+(sd[k+1]*r1p[k]
                                      +sd[k]*r2p[k]+r1b[k]*(si[k+1]*cbar-cbs[k+1])
                                      +r2b[k]*(si[k]*cbar-cbs[k]))/delsig[k]
                      +(287.05/1005.0)*((temp[k]+300.0)*(c[k]-cbar)-temp[k]*dbar) for k in range(kmax)])
    ri = torch.stack([ri[k]+heat[k] for k in range(kmax)])
    wj = torch.stack([heat[k] for k in range(kmax)]) # this is so that heating is easily accsessible
    #
    #
    a /= cs
    b /= cs
    ut /= cs
    vt /= cs
        ### Normalization by cs is required
                         ### to get the right inverse transform
    #
    return a,b,e,ut,vt,ri,wj,cbar,dbar


# In[12]:


#
# Calculate non-linear products on the Gaussian grid
# and the vertical derivatives
#
def nlprod_prescribed_mean(u,v,vort,div,temp,dxq,dyq,heat,coriolis,delsig,si,sikap,slkap,          r1b,r2b,cth1,cth2,cost_lg,kmax,imax,jmax):
    #### Using stacked variables [0] corresponds to the prescribed mean
    #### and [1] corresponds to the perturbation
    c = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    c = torch.stack((c,c))
    cbs = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64)
    cbs = torch.stack((cbs,cbs))
    dbs = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64)
    dbs = torch.stack((dbs,dbs))
    cbar = torch.zeros((jmax,imax),dtype=torch.float64)
    cbar = torch.stack((cbar,cbar))
    dbar = torch.zeros((jmax,imax),dtype=torch.float64)
    dbar = torch.stack((dbar,dbar))
    sd = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64) # sigma dot - vertical vel.
    sd = torch.stack((sd,sd))
    th = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64)
    th = torch.stack((th,th))
    cs = torch.zeros((jmax,imax),dtype=torch.float64)
    for i in range(imax):
        mu2 = np.sqrt(1.0-cost_lg[:]*cost_lg[:])
        cs[:,i] = torch.from_numpy(mu2[:])
    sd2d = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sd2d = torch.stack((sd2d,sd2d))
    sd2d1 = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sd2d1 = torch.stack((sd2d1,sd2d1))
    r1p = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    r1p = torch.stack((r1p,r1p))
    sduk1 = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sdvk1 = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sdwk1 = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    r2p = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    r2p = torch.stack((r2p,r2p))
    sduk = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sdvk = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sdwk = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    #
    # Return variables
    #
    a = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    b = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    e = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    ut = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    vt = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    ri = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    wj = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    #
    #
    # Remove mean temperature (300.0) from temp climo
    #
    temp[0] = temp[0] - 300.0
    #
    #
    # Compute c=V.del(q), cbs, dbs, cbar, dbar (AFGL Documentation)
    #
    for k in range(kmax):
        c[0,k] = u[0,k]*dxq[0] + v[0,k]*dyq[0] # mean * mean
        c[1,k] = u[0,k]*dxq[1] + u[1,k]*dxq[0] +                 v[0,k]*dyq[1] + v[1,k]*dyq[0] +                 u[1,k]*dxq[1] + v[1,k]*dyq[1] # top two lines
                                               # are prime * bar terms
                                               # and last line is
                                               # prime * prime term.
                                               # Comment last line
                                               # for linear model
    for k in range(kmax):
        cbs[0,k+1]=cbs[0,k] + c[0,k]*delsig[k]
        dbs[0,k+1]=dbs[0,k] + div[0,k]*delsig[k]
        cbs[1,k+1]=cbs[1,k] + c[1,k]*delsig[k]
        dbs[1,k+1]=dbs[1,k] + div[1,k]*delsig[k]
    cbar[0]=cbs[0,kmax]
    dbar[0]=dbs[0,kmax]
    cbar[1]=cbs[1,kmax]
    dbar[1]=dbs[1,kmax]
    #
    # Compute sd = si*(cbar+dbar)-cbs-dbs
    #
    sd[0,0] = 0.0
    sd[1,0] = 0.0
    for k in np.arange(1, kmax, 1, dtype=int):
        sd[0,k]=si[k]*(cbar[0] + dbar[0]) - cbs[0,k] - dbs[0,k]
        sd[1,k]=si[k]*(cbar[1] + dbar[1]) - cbs[1,k] - dbs[1,k]
    sd[0,kmax] = 0.0
    sd[1,kmax] = 0.0
    #
    # Compute th
    #
    th[0,0] = 0.0
    th[1,0] = 0.0
    th[0,kmax]=temp[0,kmax-1]
    th[1,kmax]=temp[1,kmax-1]
    for k in range(kmax-1):
        th[0,k+1]=cth1[k+1]*temp[0,k+1] + cth2[k]*temp[0,k]
        th[1,k+1]=cth1[k+1]*temp[1,k+1] + cth2[k]*temp[1,k]
    #
    # Compute a,b,e,ut,vt - see afgl documentation, vort[0,k] is the relative background
    #  vorticity.
    #
    for k in range(kmax):
        a[k]=((vort[0,k]+coriolis)*u[1,k] + 287.05*temp[0,k]*dyq[1])*cs+             (vort[1,k]*u[0,k] + 287.05*temp[1,k]*dyq[0])*cs+             (vort[1,k]*u[1,k] + 287.05*temp[1,k]*dyq[1])*cs # non-linear term
        b[k]=((vort[0,k]+coriolis)*v[1,k] - 287.05*temp[0,k]*dxq[1])*cs+             (vort[1,k]*v[0,k] - 287.05*temp[1,k]*dxq[0])*cs+             (vort[1,k]*v[1,k] - 287.05*temp[1,k]*dxq[1])*cs # non-linear term
        e[k]=(u[0,k]*u[1,k] + v[0,k]*v[1,k])/2.0+             (u[1,k]*u[0,k] + v[1,k]*v[0,k])/2.0+             (u[1,k]*u[1,k] + v[1,k]*v[1,k])/2.0 # non-linear term
        ut[k]=u[0,k]*temp[1,k]*cs+              u[1,k]*temp[0,k]*cs+              u[1,k]*temp[1,k]*cs # non-linear term
        vt[k]=v[0,k]*temp[1,k]*cs+              v[1,k]*temp[0,k]*cs+              v[1,k]*temp[1,k]*cs # non-linear term
    #
    # Vertical Advection
    #
    for k in range(kmax):
        sd2d[0,k]=sd[0,k]/(2.*delsig[k])
        sd2d[1,k]=sd[1,k]/(2.*delsig[k])
        sd2d1[0,k]=sd[0,k+1]/(2.*delsig[k])
        sd2d1[1,k]=sd[1,k+1]/(2.*delsig[k])
    for k in range(kmax-1):
        r1p[0,k]=temp[0,k]-(th[0,k+1]*slkap[k])/sikap[k+1]
        r1p[1,k]=temp[1,k]-(th[1,k+1]*slkap[k])/sikap[k+1]
        sduk1[k]=sd2d1[0,k]*(u[1,k+1]-u[1,k])*cs+                 sd2d1[1,k]*(u[0,k+1]-u[0,k])*cs+                 sd2d1[1,k]*(u[1,k+1]-u[1,k])*cs # non-linear term
        sdvk1[k]=sd2d1[0,k]*(v[1,k+1]-v[1,k])*cs+                 sd2d1[1,k]*(v[0,k+1]-v[0,k])*cs+                 sd2d1[1,k]*(v[1,k+1]-v[1,k])*cs # non-linear term
        ###sdwk1[k]=sd2d1[k]*(w[k+1]-w[k]) # no moisture equation
    r1p[0,kmax-1] = 0.0
    r1p[1,kmax-1] = 0.0
    sduk1[kmax-1]=0.0
    sdvk1[kmax-1]=0.0
    #sdwk1[kmax-1]=0.0
    #
    r2p[0,0]=0.0
    r2p[1,0]=0.0
    sduk[0]=0.0
    sdvk[0]=0.0
    #sdwk[0]=0.0
    for k in np.arange(1, kmax, 1, dtype=int):
        r2p[0,k]=((th[0,k]*slkap[k])/sikap[k])-temp[0,k]
        r2p[1,k]=((th[1,k]*slkap[k])/sikap[k])-temp[1,k]
        sduk[k]=sd2d[0,k]*(u[1,k]-u[1,k-1])*cs+                sd2d[1,k]*(u[0,k]-u[0,k-1])*cs+                sd2d[1,k]*(u[1,k]-u[1,k-1])*cs
        sdvk[k]=sd2d[0,k]*(v[1,k]-v[1,k-1])*cs+                sd2d[1,k]*(v[0,k]-v[0,k-1])*cs+                sd2d[1,k]*(v[1,k]-v[1,k-1])*cs
        ###sdwk[k]=sd2d[k]*(w[k]-w[k-1]) # no moisture equation
    #
    # Update a, b and ri for the temperature equation
    #
    for k in range(kmax):
        xx = a[k] + sdvk[k] + sdvk1[k]
        a[k] = xx
        xx = b[k] - sduk[k] - sduk1[k]
        b[k] = xx
        rmp=temp[0,k]*div[1,k]+(sd[0,k+1]*r1p[1,k]                +sd[0,k]*r2p[1,k])/delsig[k]                +(287.05/1005.0)*((temp[0,k]+300.0)*                (c[1,k]-cbar[1])-temp[0,k]*dbar[1])
        #
        rpm=temp[1,k]*div[0,k]+(sd[1,k+1]*r1p[0,k]                +sd[1,k]*r2p[0,k]+r1b[k]*(si[k+1]*cbar[1]-cbs[1,k+1])                +r2b[k]*(si[k]*cbar[1]-cbs[1,k]))/delsig[k]                +(287.05/1005.0)*((temp[1,k])*                (c[0,k]-cbar[0])-temp[1,k]*dbar[0])
        #
        rpp=temp[1,k]*div[1,k]+(sd[1,k+1]*r1p[1,k]                +sd[1,k]*r2p[1,k])/delsig[k]                +(287.05/1005.0)*((temp[1,k])*                (c[1,k]-cbar[1])-temp[1,k]*dbar[1]) # non-linear terms
        #
        ri[k]=rmp+rpm+rpp+heat[k]
        wj[k]=heat[k] # this is so that heating is easily accsessible
            # in the post-processed data
    #
    #
    for k in range(kmax):
        xx = a[k]/cs
        a[k] = xx
        xx = b[k]/cs
        b[k] = xx
        xx = ut[k]/cs
        ut[k] = xx
        xx = vt[k]/cs 
        vt[k] = xx
        ### Normalization by cs is required
                         ### to get the right inverse transform
    #
    return a,b,e,ut,vt,ri,wj,cbar,dbar


# In[13]:


#
# Linear version of nlprod_prescribed_mean
#
def nlprod_prescribed_mean_linear(u,v,vort,div,temp,dxq,dyq,heat,coriolis,delsig,si,sikap,slkap,          r1b,r2b,cth1,cth2,cost_lg,kmax,imax,jmax):
    #### Using stacked variables [0] corresponds to the prescribed mean
    #### and [1] corresponds to the perturbation
    c = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    c = torch.stack((c,c))
    cbs = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64)
    cbs = torch.stack((cbs,cbs))
    dbs = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64)
    dbs = torch.stack((dbs,dbs))
    cbar = torch.zeros((jmax,imax),dtype=torch.float64)
    cbar = torch.stack((cbar,cbar))
    dbar = torch.zeros((jmax,imax),dtype=torch.float64)
    dbar = torch.stack((dbar,dbar))
    sd = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64) # sigma dot - vertical vel.
    sd = torch.stack((sd,sd))
    th = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64)
    th = torch.stack((th,th))
    cs = torch.zeros((jmax,imax),dtype=torch.float64)
    for i in range(imax):
        mu2 = np.sqrt(1.0-cost_lg[:]*cost_lg[:])
        cs[:,i] = torch.from_numpy(mu2[:])
    sd2d = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sd2d = torch.stack((sd2d,sd2d))
    sd2d1 = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sd2d1 = torch.stack((sd2d1,sd2d1))
    r1p = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    r1p = torch.stack((r1p,r1p))
    sduk1 = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sdvk1 = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sdwk1 = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    r2p = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    r2p = torch.stack((r2p,r2p))
    sduk = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sdvk = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sdwk = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    #
    # Return variables
    #
    a = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    b = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    e = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    ut = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    vt = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    ri = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    wj = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    #
    #
    # Remove mean temperature (300.0) from temp climo
    #
    temp[0] = temp[0] - 300.0
    #
    #
    # Compute c=V.del(q), cbs, dbs, cbar, dbar (AFGL Documentation)
    #
    for k in range(kmax):
        c[0,k] = u[0,k]*dxq[0] + v[0,k]*dyq[0] # mean * mean
        c[1,k] = u[0,k]*dxq[1] + u[1,k]*dxq[0] +                 v[0,k]*dyq[1] + v[1,k]*dyq[0]
                ## u[1,k]*dxq[1] + v[1,k]*dyq[1] # top two lines
                                               # are prime * bar terms
                                               # and last line is
                                               # prime * prime term.
                                               # Comment last line
                                               # for linear model
    for k in range(kmax):
        cbs[0,k+1]=cbs[0,k] + c[0,k]*delsig[k]
        dbs[0,k+1]=dbs[0,k] + div[0,k]*delsig[k]
        cbs[1,k+1]=cbs[1,k] + c[1,k]*delsig[k]
        dbs[1,k+1]=dbs[1,k] + div[1,k]*delsig[k]
    cbar[0]=cbs[0,kmax]
    dbar[0]=dbs[0,kmax]
    cbar[1]=cbs[1,kmax]
    dbar[1]=dbs[1,kmax]
    #
    # Compute sd = si*(cbar+dbar)-cbs-dbs
    #
    sd[0,0] = 0.0
    sd[1,0] = 0.0
    for k in np.arange(1, kmax, 1, dtype=int):
        sd[0,k]=si[k]*(cbar[0] + dbar[0]) - cbs[0,k] - dbs[0,k]
        sd[1,k]=si[k]*(cbar[1] + dbar[1]) - cbs[1,k] - dbs[1,k]
    sd[0,kmax] = 0.0
    sd[1,kmax] = 0.0
    #
    # Compute th
    #
    th[0,0] = 0.0
    th[1,0] = 0.0
    th[0,kmax]=temp[0,kmax-1]
    th[1,kmax]=temp[1,kmax-1]
    for k in range(kmax-1):
        th[0,k+1]=cth1[k+1]*temp[0,k+1] + cth2[k]*temp[0,k]
        th[1,k+1]=cth1[k+1]*temp[1,k+1] + cth2[k]*temp[1,k]
    #
    # Compute a,b,e,ut,vt - see afgl documentation, vort[0,k] is the relative background
    #  vorticity.
    #
    for k in range(kmax):
        a[k]=((vort[0,k]+coriolis)*u[1,k] + 287.05*temp[0,k]*dyq[1])*cs+             (vort[1,k]*u[0,k] + 287.05*temp[1,k]*dyq[0])*cs
             ##(vort[1,k]*u[1,k] + 287.05*temp[1,k]*dyq[1])*cs # non-linear term
        b[k]=((vort[0,k]+coriolis)*v[1,k] - 287.05*temp[0,k]*dxq[1])*cs+             (vort[1,k]*v[0,k] - 287.05*temp[1,k]*dxq[0])*cs
             ##(vort[1,k]*v[1,k] - 287.05*temp[1,k]*dxq[1])*cs # non-linear term
        e[k]=(u[0,k]*u[1,k] + v[0,k]*v[1,k])/2.0+             (u[1,k]*u[0,k] + v[1,k]*v[0,k])/2.0
             ##(u[1,k]*u[1,k] + v[1,k]*v[1,k])/2.0 # non-linear term
        ut[k]=u[0,k]*temp[1,k]*cs+              u[1,k]*temp[0,k]*cs
             ##u[1,k]*temp[1,k]*cs # non-linear term
        vt[k]=v[0,k]*temp[1,k]*cs+              v[1,k]*temp[0,k]*cs
              ##v[1,k]*temp[1,k]*cs # non-linear term
    #
    # Vertical Advection
    #
    for k in range(kmax):
        sd2d[0,k]=sd[0,k]/(2.*delsig[k])
        sd2d[1,k]=sd[1,k]/(2.*delsig[k])
        sd2d1[0,k]=sd[0,k+1]/(2.*delsig[k])
        sd2d1[1,k]=sd[1,k+1]/(2.*delsig[k])
    for k in range(kmax-1):
        r1p[0,k]=temp[0,k]-(th[0,k+1]*slkap[k])/sikap[k+1]
        r1p[1,k]=temp[1,k]-(th[1,k+1]*slkap[k])/sikap[k+1]
        sduk1[k]=sd2d1[0,k]*(u[1,k+1]-u[1,k])*cs+                 sd2d1[1,k]*(u[0,k+1]-u[0,k])*cs
                 ##sd2d1[1,k]*(u[1,k+1]-u[1,k])*cs # non-linear term
        sdvk1[k]=sd2d1[0,k]*(v[1,k+1]-v[1,k])*cs+                 sd2d1[1,k]*(v[0,k+1]-v[0,k])*cs
                 ##sd2d1[1,k]*(v[1,k+1]-v[1,k])*cs # non-linear term
        ###sdwk1[k]=sd2d1[k]*(w[k+1]-w[k]) # no moisture equation
    r1p[0,kmax-1] = 0.0
    r1p[1,kmax-1] = 0.0
    sduk1[kmax-1]=0.0
    sdvk1[kmax-1]=0.0
    #sdwk1[kmax-1]=0.0
    #
    r2p[0,0]=0.0
    r2p[1,0]=0.0
    sduk[0]=0.0
    sdvk[0]=0.0
    #sdwk[0]=0.0
    for k in np.arange(1, kmax, 1, dtype=int):
        r2p[0,k]=((th[0,k]*slkap[k])/sikap[k])-temp[0,k]
        r2p[1,k]=((th[1,k]*slkap[k])/sikap[k])-temp[1,k]
        sduk[k]=sd2d[0,k]*(u[1,k]-u[1,k-1])*cs+                sd2d[1,k]*(u[0,k]-u[0,k-1])*cs
                ##sd2d[1,k]*(u[1,k]-u[1,k-1])*cs # non-linear term
        sdvk[k]=sd2d[0,k]*(v[1,k]-v[1,k-1])*cs+                sd2d[1,k]*(v[0,k]-v[0,k-1])*cs
                ##sd2d[1,k]*(v[1,k]-v[1,k-1])*cs # non-linear term
        ###sdwk[k]=sd2d[k]*(w[k]-w[k-1]) # no moisture equation
    #
    # Update a, b and ri for the temperature equation
    #
    for k in range(kmax):
        xx = a[k] + sdvk[k] + sdvk1[k]
        a[k] = xx
        xx = b[k] - sduk[k] - sduk1[k]
        b[k] = xx
        rmp=temp[0,k]*div[1,k]+(sd[0,k+1]*r1p[1,k]                +sd[0,k]*r2p[1,k])/delsig[k]                +(287.05/1005.0)*((temp[0,k]+300.0)*                (c[1,k]-cbar[1])-temp[0,k]*dbar[1])
        #
        rpm=temp[1,k]*div[0,k]+(sd[1,k+1]*r1p[0,k]                +sd[1,k]*r2p[0,k]+r1b[k]*(si[k+1]*cbar[1]-cbs[1,k+1])                +r2b[k]*(si[k]*cbar[1]-cbs[1,k]))/delsig[k]                +(287.05/1005.0)*((temp[1,k])*                (c[0,k]-cbar[0])-temp[1,k]*dbar[0])
        #
        #rpp=temp[1,k]*div[1,k]+(sd[1,k+1]*r1p[1,k]\
        #        +sd[1,k]*r2p[1,k])/delsig[k]\
        #        +(287.05/1005.0)*((temp[1,k])*\
        #        (c[1,k]-cbar[1])-temp[1,k]*dbar[1]) # non-linear terms
        #
        ri[k] = rmp + rpm + heat[k]
        #ri[k]=rmp+rpm+rpp+heat[k]
        wj[k]=heat[k] # this is so that heating is easily accsessible
            # in the post-processed data
    #
    #
    for k in range(kmax):
        xx = a[k]/cs
        a[k] = xx
        xx = b[k]/cs
        b[k] = xx
        xx = ut[k]/cs
        ut[k] = xx
        xx = vt[k]/cs 
        vt[k] = xx
        ### Normalization by cs is required
                         ### to get the right inverse transform
    #
    return a,b,e,ut,vt,ri,wj,cbar,dbar


# In[14]:


#
# Implicit time differencing - Not implemented correctly
#
def implicit(dt,amtrx,cmtrx,dmtrx,emtrx,zmn1,zmn2,zmn3,dmn1,dmn2,dmn3,tmn1,tmn2,             tmn3,wmn1,wmn2,wmn3,qmn1,qmn2,qmn3,phismn,delsig,kmax,mw,zw):
    em = torch.zeros((mw,zw,kmax,kmax),dtype=torch.float64)
    ae = 6.371E+06
    andree = 2.0e-02
    alpha = 0.5
    dt2 = 2.0*dt
    ccc = alpha*dt2
    cccs = ccc*ccc
    bbb = 1.0-alpha
    bb1 = 1.0/alpha
    nn = torch.arange(0,mw).reshape(mw, 1).double()
    nn = nn.expand(mw,zw)
    aaa = nn * (nn + 1)/(ae*ae)
    #
    ### compute rh2 = amtrx @ store2 for each wave number
    #
    store2 = tmn1+ccc*tmn3
    rr = torch.from_numpy(amtrx).double()
    tmp_r = torch.einsum('kl,lji->kji',[rr,store2.real])
    tmp_i = torch.einsum('kl,lji->kji',[rr,store2.imag])
    rh2 = torch.complex(tmp_r, tmp_i)
    ###
    rh1 = dmn1
    xx = qmn1 + ccc*qmn3
    rhs = rh1 + ccc*(dmn3+aaa*(phismn+rh2+xx*287.05*300.0))
    #
    ##### rhs = emtrx @ rhs
    #
    tmp_r = torch.einsum('jikl,lji->kji',[emtrx,rhs.real])
    tmp_i = torch.einsum('jikl,lji->kji',[emtrx,rhs.imag])
    rhs = torch.complex(tmp_r, tmp_i)
    #
    ##### rh2 = cmtrx @ rhs
    #
    qq = torch.from_numpy(cmtrx).double()
    tmp_r = torch.einsum('kl,lji->kji',[qq,rhs.real])
    tmp_i = torch.einsum('kl,lji->kji',[qq,rhs.imag])
    rh2 = torch.complex(tmp_r, tmp_i)
    #
    xx = torch.einsum('kij,k->kij',dmn2,torch.tensor(delsig)).sum(dim=0) # dbar
    #
    dzdt = zmn3
    zmn3=zmn1+dt2*dzdt
    dtdt = tmn3
    tmn3=tmn1+dt2*(dtdt-rh2)
    wmn1=wmn3 #### This is just the heating for output
    dmn3=bb1*(rhs-bbb*rh1)
    dqdt = qmn3
    qmn3 = qmn1 + dt2*(dqdt - xx)
    #
    zmn1,zmn2,zmn3 = tfilt(andree,zmn1,zmn2,zmn3)
    dmn1,dmn2,dmn2 = tfilt(andree,dmn1,dmn2,dmn3)
    tmn1,tmn2,tmn3 = tfilt(andree,tmn1,tmn2,tmn3)
    qmn1,qmn2,qmn3 = tfilt(andree,qmn1,qmn2,qmn3)
    return zmn1,zmn2,zmn3,dmn1,dmn2,dmn3,tmn1,tmn2,tmn3,qmn1,qmn2,qmn3
#


# In[15]:


#
# Explicit time differencing 
#
def explicit(dt,amtrx,cmtrx,dmtrx,emtrx,zmn1,zmn2,zmn3,dmn1,dmn2,dmn3,tmn1,tmn2,             tmn3,wmn1,wmn2,wmn3,qmn1,qmn2,qmn3,phismn,delsig,kmax,mw,zw):
    ae = 6.371E+06
    andree = 2.0e-02
    alpha = 0.5
    dt2 = 2.0*dt
    ccc = alpha*dt2
    cccs = ccc*ccc
    bbb = 1.0-alpha
    bb1 = 1.0/alpha
    nn = torch.arange(0,mw).reshape(mw, 1).double()
    nn = nn.expand(mw,zw)
    aaa = nn * (nn + 1)/(ae*ae)
    yy = torch.zeros((mw,zw),dtype=torch.float64)
    xx = torch.complex(yy,yy)
    #
    ### compute rh2 = amtrx @ tmn2 for each wave number
    #
    rr = torch.from_numpy(amtrx).double()
    tmp_r = torch.einsum('kl,lji->kji',[rr,tmn2.real])
    tmp_i = torch.einsum('kl,lji->kji',[rr,tmn2.imag])
    rh2 = torch.complex(tmp_r, tmp_i) # this is added to divergence
                                      # tendency
    ###
    rr = torch.from_numpy(cmtrx).double()
    tmp_r = torch.einsum('kl,lji->kji',[rr,dmn2.real]) #dmn2???- Sela uses dmn1
    tmp_i = torch.einsum('kl,lji->kji',[rr,dmn2.imag])
    rh3 = torch.complex(tmp_r, tmp_i) # this is added to the
                                      # temperature tendency
    ###
    dddt = dmn3 + aaa*(phismn+rh2+qmn2*287.05*300.0)
    #
    #
    for k in range(kmax):
        xx = xx + delsig[k]*dmn2[k] # dbar
    #
    dzdt = zmn3
    zmn3=zmn1+dt2*dzdt
    dtdt = tmn3 - rh3
    tmn3=tmn1+dt2*dtdt
    wmn1=wmn3 #### This is just the heating for output
    dmn3 = dmn1 + dt2*dddt
    dqdt = qmn3 - xx
    qmn3 = qmn1 + dt2*dqdt
    #
    zmn1,zmn2,zmn3 = tfilt(andree,zmn1,zmn2,zmn3)
    dmn1,dmn2,dmn2 = tfilt(andree,dmn1,dmn2,dmn3)
    tmn1,tmn2,tmn3 = tfilt(andree,tmn1,tmn2,tmn3)
    qmn1,qmn2,qmn3 = tfilt(andree,qmn1,qmn2,qmn3)
    return zmn1,zmn2,zmn3,dmn1,dmn2,dmn3,tmn1,tmn2,tmn3,qmn1,qmn2,qmn3
#


# In[16]:


#
# Robert time filtering
#
def tfilt(filt,fmn1,fmn2,fmn3):
    xxx = fmn2 + filt*(fmn1 - 2*fmn2 + fmn3)
    fmn1 = xxx
    fmn2 = fmn3
    return fmn1,fmn2,fmn3


# In[17]:


#
# Convert spectral vorticity and divergence into u & v on
# Gaussian grid. Note this routine assumes that the relative
# vorticity is input
#
def uv(divsht,vort,div,mw,zw,kmax,imax,jmax):
    ae = 6.371E+06
    nn = torch.arange(0,mw).reshape(mw, 1).double()
    nn = nn.expand(mw,zw)
    invlap = -(ae*ae)/nn/(nn+1)
    invlap[0]=0.
    streamf = invlap.unsqueeze(0)*vort/ae
    vp = invlap.unsqueeze(0)*div/ae
    vordiv = torch.stack((streamf,vp), 1)
    uvgrid = divsht(vordiv)
    u = uvgrid[:,0,:,:]
    v = uvgrid[:,1,:,:]
    return u,v
#


# In[18]:


#
# Calculate grad(lnPs)
#
def gradq(divsht,qmn,mw,zw,imax,jmax):
    ae = 6.371E+06
    qmna = qmn/ae
    zeroq = torch.stack((torch.zeros_like(qmna),qmna))
    gradqgrid = divsht(zeroq)
    dxq = gradqgrid[0]
    dyq = gradqgrid[1]
    return dxq,dyq
#


# In[19]:


#
# Convert u and v on Gaussian grid to spectral vorticity
# and divergence
#
def vortdivspec(vsht,u,v,kmax,mw,zw):
    ae = 6.371E+06
    nn = torch.arange(0,mw).reshape(mw, 1).double()
    nn = nn.expand(mw,zw)
    lap = -(nn*(nn+1))/(ae*ae)
    uvgrid = torch.stack((u,v), 1)
    xy = (lap*vsht(uvgrid))*ae 
    vort = xy[:,0,:,:]
    div = xy[:,1,:,:]
    return vort,div


# In[20]:


#
# Convert from grid to spectral and then apply laplacian
#
def lap_sht(dsht,e,mw,zw):
    ae = 6.371E+06
    nn = torch.arange(0,mw).reshape(mw, 1).double()
    nn = nn.expand(mw,zw)
    lap = -(nn*(nn+1))/(ae*ae)
    spec_lap = (lap*dsht(e))         
    return spec_lap


# In[21]:


#
# Get gridded lnps and geopotential
#
def get_geo_ps(disht,tmn,qmn,phismn,amtrx,kmax,mw,zw,jmax,imax):
    lnps = disht(qmn)
    qq = tmn
    rr = torch.from_numpy(amtrx)
    tmp_r = torch.einsum('kl,lji->kji',[rr,qq.real])
    tmp_i = torch.einsum('kl,lji->kji',[rr,qq.imag])
    rh2 = torch.complex(tmp_r, tmp_i)
    #
    geosmn = torch.stack([phismn + rh2[k] for k in range(kmax)])
    geo = disht(geosmn)
    return lnps,geo


# In[22]:


#
# Calcucluate Potential Temperature in sigma coordinates on the
# Gaussian Grid
#
#
def potential_temp(temp,sigma,lnps,kmax):
    pressure = temp*0.0
    surfp = (np.exp(lnps))*1000.0
    for k in range(kmax):
        pressure[k] = sigma[k]*surfp
    #
    r = 287.5
    cp = 1004.0
    gamma = r/cp
    theta = temp*((1000.0/pressure)**gamma)
    return theta


# In[23]:


#
# Convert spectral data to Gaussian grid and write to disk
# as netcdf data
#
def postprocessing(disht,divsht,zmnt,dmnt,tmnt,qmnt,wmnt,heat,              phismn,amtrx,                   times,mw,zw,kmax,imax,jmax,sl,lats,lons,tl,                  datapath):
    u = torch.zeros((tl,kmax,jmax,imax),dtype=torch.float64)
    v = torch.zeros((tl,kmax,jmax,imax),dtype=torch.float64)
    vort = torch.zeros((tl,kmax,jmax,imax),dtype=torch.float64)
    div = torch.zeros((tl,kmax,jmax,imax),dtype=torch.float64)
    temp = torch.zeros((tl,kmax,jmax,imax),dtype=torch.float64)
    geo = torch.zeros((tl,kmax,jmax,imax),dtype=torch.float64)
    lnps = torch.zeros((tl,jmax,imax),dtype=torch.float64)
    for it in range(tl):
        u[it],v[it] = uv(divsht,zmnt[it],dmnt[it],mw,zw,kmax,imax,jmax)
        for k in range(kmax):
#            vort[it,k] = disht(zmnt[it,k]) # uncomment if vort wanted
#            div[it,k] = disht(dmnt[it,k]) # uncomment of div wanted
            temp[it,k] = disht(tmnt[it,k])
        lnps[it],geo[it] = get_geo_ps(disht,tmnt[it],qmnt[it],phismn,amtrx,kmax,mw,zw,                         jmax,imax)
    #
    tstamp_start = str(times[0])[0:10]
    tstamp_end = str(times[tl-1])[0:10]
    stamp = tstamp_start+'_'+tstamp_end
    du = xr.Dataset({'u': (['time','lev','lat','lon'],u.numpy())},                        coords={'time': times,'lev':sl, 'lat': lats, 'lon': lons})
    dv = xr.Dataset({'v': (['time','lev','lat','lon'],v.numpy())},                        coords={'time': times,'lev':sl, 'lat': lats, 'lon': lons})
#    dvort = xr.Dataset({'vort': (['time','lev','lat','lon'],vort.numpy())},\
#                        coords={'time': times,'lev':sl, 'lat': lats, 'lon': lons}) # uncomment of vort wanted
#    ddiv = xr.Dataset({'div': (['time','lev','lat','lon'],div.numpy())},\
#                        coords={'time': times,'lev':sl, 'lat': lats, 'lon': lons}) # uncomment of div wanted
    dtemp = xr.Dataset({'t': (['time','lev','lat','lon'],temp.numpy())},                        coords={'time': times,'lev':sl, 'lat': lats, 'lon': lons})
    dheat = xr.Dataset({'heat': (['time','lev','lat','lon'],heat.numpy())},                     coords={'time': times,'lev':sl, 'lat': lats, 'lon': lons})
    dgeo = xr.Dataset({'geo': (['time','lev','lat','lon'],geo.numpy())},                        coords={'time': times,'lev':sl, 'lat': lats, 'lon': lons})
    dps = xr.Dataset({'lnps': (['time','lat','lon'],lnps.numpy())},                        coords={'time': times,'lat': lats, 'lon': lons})
    datasets = list([du, dv, dtemp, dgeo, dps, dheat])
    filename_paths = list([datapath+'uvel_'+stamp+'.nc'
                           ,datapath+'vvel_'+stamp+'.nc'
                           ,datapath+'temp_'+stamp+'.nc'
                           ,datapath+'geo_'+stamp+'.nc'
                           ,datapath+'lnps_'+stamp+'.nc'
                           ,datapath+'heat_'+stamp+'.nc'])
    xr.save_mfdataset(datasets, filename_paths)
    return


# In[ ]:

####
# latent heating routine
#
#
# Now we have a stochastic parametetric model for the rainfall
# Here we explore possible ways of including a feedback from the atmospheric circulation
#
#
from random import seed
from random import gauss
from random import uniform
#
def latent_heat_release(dsht,disht,shape,scale,imax,jmax,kmax,Lat,delsig,div,iseed):
    ijmax = imax*jmax
    precip = np.zeros(ijmax)
    div = np.reshape(div,ijmax)
    shape = np.reshape(shape,ijmax)
    scale = np.reshape(scale,ijmax)
    #####scale = np.ones(ijmax)*4.6 ##### Testing constant scale
    #####l_prec = ( div < 0.0 ) # where low level convergence = 1 otherwise 0
    #####l_prec = 1*l_prec
    #####feedback = l_prec*2.0 + (1 - l_prec)*0.5
    precip = np.random.gamma(shape,scale,ijmax) # using the non-interactive Gamma distribution to predict precip
    #####precip = np.random.gamma(shape,scale,ijmax)*feedback # using the non-interactive Gamma distribution to predict #                                         precip interactively (i.e., feedback term)
#    precip = (l_prec)*div*(-5e+05) # using low level converge to predict rainfall - positive definite only
#
    precip = np.reshape(precip,(jmax,imax))
#
#
# Convert rainfall anomaly into a hearting rate that can
# applied to the temperature equation (i.e., Q/Cp) and distribute
# in the vertical
#
    vert_struc = np.zeros(kmax) # whatever user wants
#                                 kmax is lowest level and
#                                 0 is the upper most level of
#                                 the atmosphere
    heat = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
#
    vert_struc[0] = 0.0
    vert_struc[1] = 0.0
    vert_struc[2] = 0.0
    vert_struc[3] = 0.0
    vert_struc[4] = 0.0
    vert_struc[5] = 0.25
    vert_struc[6] = 0.5
    vert_struc[7] = 0.75
    vert_struc[8] = 1.0
    vert_struc[9] = 1.5
    vert_struc[10] = 1.75
    vert_struc[11] = 1.75
    vert_struc[12] = 1.75
    vert_struc[13] = 2.0
    vert_struc[14] = 2.0
    vert_struc[15] = 2.0
    vert_struc[16] = 2.0
    vert_struc[17] = 1.75
    vert_struc[18] = 1.75
    vert_struc[19] = 1.5
    vert_struc[20] = 1.25
    vert_struc[21] = 0.75
    vert_struc[22] = 0.5
    vert_struc[23] = 0.25
    vert_struc[24] = 0.0
    vert_struc[25] = 0.0
#
# Need to ensure that vertical integral normalizes to 1.0
    rnorm = (vert_struc*delsig).sum()
    vert_struc = vert_struc/rnorm
#
#
# convert to heating and multiple by vertical structure
#
    Lv = 2.5e+06
    rhow = 1000.0
    Cp = 1005.0
    Ps = 101325.0
    grav = 9.8
    beta = (Lv*rhow/Cp)*(grav/Ps)/(1000.0*86400.0)
    tropics = np.exp((-Lat*Lat)/1000.0) # limit forcing to tropics
    tmp = tropics*precip
#
# Transform forward and backward to reduce forcing at unresolved scales
#
    tmpspec = dsht(torch.from_numpy(tmp))
    tmp = disht(tmpspec)
#
    for k in range(kmax):
        heat[k,:,:] = (tmp[:,:]*vert_struc[k]*beta) # in K/sec
    
    return heat




