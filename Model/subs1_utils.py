#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import xarray
import pandas as pd
from netCDF4 import Dataset
import dask
import torch.nn as nn
import torch.fft as fft

from sht_utils import *


# In[2]:


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
    delsig = np.zeros(kmax) #### The following parameter must be set
    si = np.zeros(kmaxp1)   #### and are returned
    sl = np.zeros(kmax)
    sikap = np.zeros(kmaxp1)
    slkap = np.zeros(kmax)
    cth1 = np.zeros(kmax)
    cth2 = np.zeros(kmax)
    r1b = np.zeros(kmax)
    r2b = np.zeros(kmax)
    #### Parameter setting above are required
    #
    #### This example below corresponds to the LBM
    sigLBM = np.zeros(kmax)
    siglLBM = np.zeros(kmaxp1)
    sigLBM[0]=0.02075
    sigLBM[1]=0.09234
    sigLBM[2]=0.2025
    sigLBM[3]=0.3478
    sigLBM[4]=0.5133
    sigLBM[5]=0.6789
    sigLBM[6]=0.8146
    sigLBM[7]=0.8999
    sigLBM[8]=0.9499
    sigLBM[9]=0.9800
    sigLBM[10]=0.9950
    #
    siglLBM[0] = sigLBM[0]/2.0
    for k in np.arange(1, kmax, 1, dtype=int):
        siglLBM[k] = (sigLBM[k] + sigLBM[k-1])/2.0
    siglLBM[kmax] = 1.0
    #
    for kk in range(kmax):
        k = kmax - kk - 1
        delsig[k] = siglLBM[k+1]-siglLBM[k]
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
    t0h = np.zeros(kmax+1)
    th = np.zeros(kmax+1)
    rkappa = 287.05/1005.0 #### R/Cp
    si[0]=0.0
    si[kmax]=1.0
    sikap[0]=0.0
    sikap[kmax]=1.0
    for k in range(kmax-1):
        si[k+1]=si[k]+delsig[k]
        sikap[k+1]=si[k+1]**rkappa
    rk1=1.0+rkappa
    #
    for k in range(kmax):
        slkap[k]=(si[k+1]**rk1-si[k]**rk1)/(rk1*(si[k+1]-si[k]))
        sl[k]=slkap[k]**(1.0/rkappa)
    #
    for k in range(kmax-1):
        cth1[k+1]=sikap[k+1]/(2.0*slkap[k+1])
        cth2[k]=sikap[k+1]/(2.0*slkap[k])
    #
    t0h[0]=0.0
    t0h[kmax]=300.0
    for k in range(kmax-1):
        t0h[k+1]=cth1[k+1]*300.0+cth2[k]*300.0
    #
    for k in range(kmax):
        if ( k > 0 ):
            r2b[k]=t0h[k]*slkap[k]/sikap[k]-300.0
        if ( k < kmax-1 ):
            r1b[k]=300.0-slkap[k]/sikap[k+1]*t0h[k+1]
    return delsig, si, sl, sikap, slkap, cth1, cth2, r1b, r2b


# In[3]:


##
## This matrix in version is used every time step in the implicit scheme
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


# In[4]:


###
### AMTRX, CMTRX and DMTRX used to calculate geopotential
### height from the temperature and used in the semi-implicit
### time differencing.
###
def mcoeff(kmax,si,sl,slkap,r1b,r2b,delsig):
    amtrx = np.zeros((kmax,kmax))
    cmtrx = np.zeros((kmax,kmax))
    dmtrx = np.zeros((kmax,kmax))
    #
    # local variables
    b = np.zeros((kmax,kmax))
    h = np.zeros((kmax,kmax))
    cm = np.zeros((kmax,kmax))
    am = np.zeros((kmax,kmax))
    aa = np.zeros(kmax)
    bb = np.zeros(kmax)
    mu = np.zeros(kmax)
    nu = np.zeros(kmax)
    lamda = np.zeros(kmax)
    for k in range(kmax-1):
        aa[k]=0.5*1005.0*(slkap[k]-slkap[k+1])/slkap[k+1]
        bb[k]=0.5*1005.0*(slkap[k]-slkap[k+1])/slkap[k]
#
    for k in range(kmax):
        lamda[k]=(287.05/1005.0)*300.0-(si[k]*r2b[k]+si[k+1]*r1b[k])/delsig[k]
        mu[k]=lamda[k]+r1b[k]/delsig[k]
        nu[k]=mu[k]+r2b[k]/delsig[k]
#
    for k in range(kmax-1):
        h[k,k]=-1.0
        h[k,k+1]=1.0
        b[k,k]=bb[k]
        b[k,k+1]=aa[k]
#
    for k in range(kmax):
        h[kmax-1,k]=delsig[k]
        b[kmax-1,k]=287.05*delsig[k]
#
    for i in range(kmax-1):
        cmtrx[i,i]=mu[i]*delsig[i]
        for j in np.arange(i, kmax-1, 1, dtype=int):
            cmtrx[i,j+1]=lamda[i]*delsig[j+1]
        for j in np.arange(0, i+1, 1, dtype=int):
            cmtrx[i+1,j]=nu[i+1]*delsig[j]
#
    cmtrx[kmax-1,kmax-1]=mu[kmax-1]*delsig[kmax-1]
    hinv = np.linalg.inv(h)
    amtrx = hinv @ b
    cm = amtrx @ cmtrx
    for k in range(kmax):
        for l in range(kmax):
            am[k,l]=287.05*300.0*delsig[l]
#
    dmtrx = cm + am
    return amtrx,cmtrx,dmtrx


# In[5]:


#
#
# Horizontal Diffusion del*4 
#
def diffsn(zmn1,zmn3,dmn1,dmn3,tmn1,tmn3,kmax,mw,zw):
    ae = 6.371E+06
    a4 = ae*ae*ae*ae
    dkh = a4/(mw*mw*(mw+1)*(mw+1)*21*60*60)
    ekh = a4/(mw*mw*(mw+1)*(mw+1)*28*60*60)
    dkha4 = dkh/a4
    ekha4 = ekh/a4
    #
    nn = torch.arange(0,mw).reshape(mw, 1).double()
    nn = nn.expand(mw,zw)
    nn2 = nn * ( nn + 1 )
    nn4 = nn2 * nn2
    for k in range(kmax):
        dddt = dmn3[k] - dkha4*nn4*dmn1[k]
        dmn3[k] = dddt
        dzdt = zmn3[k] - ekha4*nn4*zmn1[k]
        zmn3[k] = dzdt
        dtdt = tmn3[k] - ekha4*nn4*tmn1[k]
        tmn3[k] = dtdt
    #
    #
    return zmn3,dmn3,tmn3
#


# In[6]:


def damp(zmn1,zmn3,dmn1,dmn3,tmn1,tmn3,qmn1,qmn3,f_spec,tclim,lnpsclim,kmax):
    newton = torch.zeros((kmax),dtype=torch.float64) + 1/(20*24*60*60)
    ray = torch.zeros((kmax),dtype=torch.float64) + 1/(10*24*60*60)
    ray[kmax-1] = 1/(2*24*60*60)
    newton[kmax-1] = 1/(2*24*60*60)
    for k in range(kmax):
        xxx = zmn3[k] - ray[k]*(zmn1[k]-f_spec)
        yyy = dmn3[k] - ray[k]*dmn1[k]
        zzz = tmn3[k] - newton[k]*(tmn1[k]-tclim[k])
        zmn3[k] = xxx
        dmn3[k] = yyy
        tmn3[k] = zzz
    #
    qmn3 = qmn3 - newton[0]*(qmn1-lnpsclim)
    return zmn3,dmn3,tmn3,qmn3


# In[7]:


#
# Calculate non-linear products on the Gaussian grid
# and the vertical derivatives
#
def nlprod(u,v,vort,div,temp,dxq,dyq,heat,delsig,si,sikap,slkap,          r1b,r2b,cth1,cth2,cost_lg,kmax,imax,jmax):
    c = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    cbs = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64)
    dbs = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64)
    cbar = torch.zeros((jmax,imax),dtype=torch.float64)
    dbar = torch.zeros((jmax,imax),dtype=torch.float64)
    sd = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64) # sigma dot - vertical vel.
    th = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64)
    cs = torch.zeros((jmax,imax),dtype=torch.float64)
    for i in range(imax):
        mu2 = np.sqrt(1.0-cost_lg[:]*cost_lg[:])
        cs[:,i] = torch.from_numpy(mu2[:])
    sd2d = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sd2d1 = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    r1p = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sduk1 = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sdvk1 = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sdwk1 = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    r2p = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
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
    # Remove mean temperature (300.0) from temp
    #
    temp = temp - 300.0
    #
    # Compute c=V.del(q), cbs, dbs, cbar, dbar (AFGL Documentation)
    #
    for k in range(kmax):
        c[k] = u[k]*dxq + v[k]*dyq
    for k in range(kmax):
        cbs[k+1]=cbs[k] + c[k]*delsig[k]
        dbs[k+1]=dbs[k] + div[k]*delsig[k]
    cbar=cbs[kmax]
    dbar=dbs[kmax]
    #
    # Compute sd = si*(cbar+dbar)-cbs-dbs
    #
    sd[0] = 0.0
    for k in np.arange(1, kmax, 1, dtype=int):
        sd[k]=si[k]*(cbar + dbar) - cbs[k] - dbs[k]
    sd[kmax] = 0.0
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
    for k in range(kmax):
        a[k]=(vort[k]*u[k] + 287.05*temp[k]*dyq)*cs
        b[k]=(vort[k]*v[k] - 287.05*temp[k]*dxq)*cs
        e[k]=(u[k]*u[k] + v[k]*v[k])/2.0
        ut[k]=u[k]*temp[k]*cs
        vt[k]=v[k]*temp[k]*cs
    #
    # Vertical Advection
    #
    for k in range(kmax):
        sd2d[k]=sd[k]/(2.*delsig[k])
        sd2d1[k]=sd[k+1]/(2.*delsig[k])
    for k in range(kmax-1):
        r1p[k]=temp[k]-(th[k+1]*slkap[k])/sikap[k+1]
        sduk1[k]=sd2d1[k]*(u[k+1]-u[k])*cs
        sdvk1[k]=sd2d1[k]*(v[k+1]-v[k])*cs
        ###sdwk1[k]=sd2d1[k]*(w[k+1]-w[k]) # no moisture equation
    r1p[kmax-1] = 0.0
    sduk1[kmax-1]=0.0
    sdvk1[kmax-1]=0.0
    sdwk1[kmax-1]=0.0
    #
    r2p[0]=0.0
    sduk[0]=0.0
    sdvk[0]=0.0
    sdwk[0]=0.0
    for k in np.arange(1, kmax, 1, dtype=int):
        r2p[k]=((th[k]*slkap[k])/sikap[k])-temp[k]
        sduk[k]=sd2d[k]*(u[k]-u[k-1])*cs
        sdvk[k]=sd2d[k]*(v[k]-v[k-1])*cs
        ###sdwk[k]=sd2d[k]*(w[k]-w[k-1]) # no moisture equation
    #
    # Update a, b and ri for the temperature equation
    #
    for k in range(kmax):
        xx = a[k] + sdvk[k] + sdvk1[k]
        a[k] = xx
        xx = b[k] - sduk[k] - sduk1[k]
        b[k] = xx
        ri[k]=temp[k]*div[k]+(sd[k+1]*r1p[k]                +sd[k]*r2p[k]+r1b[k]*(si[k+1]*cbar-cbs[k+1])                +r2b[k]*(si[k]*cbar-cbs[k]))/delsig[k]                +(287.05/1005.0)*((temp[k]+300.0)*(c[k]-cbar)-temp[k]*dbar)
        ri[k]=ri[k]+heat[k]
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


# In[ ]:


#
# Calculate non-linear products on the Gaussian grid
# and the vertical derivatives
#
def nlprod_prescribed_mean(u,v,vort,div,temp,dxq,dyq,heat,delsig,si,sikap,slkap,          r1b,r2b,cth1,cth2,cost_lg,kmax,imax,jmax):
    #### Using stacked variables [0] corresponds to the prescribed mean
    #### and [1] corresponds to the perturbation
    c = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    cbs = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64)
    dbs = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64)
    cbar = torch.zeros((jmax,imax),dtype=torch.float64)
    dbar = torch.zeros((jmax,imax),dtype=torch.float64)
    sd = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64) # sigma dot - vertical vel.
    th = torch.zeros((kmax+1,jmax,imax),dtype=torch.float64)
    cs = torch.zeros((jmax,imax),dtype=torch.float64)
    for i in range(imax):
        mu2 = np.sqrt(1.0-cost_lg[:]*cost_lg[:])
        cs[:,i] = torch.from_numpy(mu2[:])
    sd2d = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sd2d1 = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    r1p = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sduk1 = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sdvk1 = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    sdwk1 = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    r2p = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
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
    # Compute a,b,e,ut,vt - see afgl documentation
    #
    for k in range(kmax):
        a[k]=(vort[0,k]*u[1,k] + 287.05*temp[0,k]*dyq[1])*cs+             (vort[1,k]*u[0,k] + 287.05*temp[1,k]*dyq[0])*cs+             (vort[1,k]*u[1,k] + 287.05*temp[1,k]*dyq[1])*cs # non-linear term
        b[k]=(vort[0,k]*v[1,k] - 287.05*temp[0,k]*dxq[1])*cs+             (vort[1,k]*v[0,k] - 287.05*temp[1,k]*dxq[0])*cs+             (vort[1,k]*v[1,k] - 287.05*temp[1,k]*dxq[1])*cs
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
        r1p[k]=temp[k]-(th[k+1]*slkap[k])/sikap[k+1]
        sduk1[k]=sd2d1[k]*(u[k+1]-u[k])*cs
        sdvk1[k]=sd2d1[k]*(v[k+1]-v[k])*cs
        ###sdwk1[k]=sd2d1[k]*(w[k+1]-w[k]) # no moisture equation
    r1p[kmax-1] = 0.0
    sduk1[kmax-1]=0.0
    sdvk1[kmax-1]=0.0
    sdwk1[kmax-1]=0.0
    #
    r2p[0]=0.0
    sduk[0]=0.0
    sdvk[0]=0.0
    sdwk[0]=0.0
    for k in np.arange(1, kmax, 1, dtype=int):
        r2p[k]=((th[k]*slkap[k])/sikap[k])-temp[k]
        sduk[k]=sd2d[k]*(u[k]-u[k-1])*cs
        sdvk[k]=sd2d[k]*(v[k]-v[k-1])*cs
        ###sdwk[k]=sd2d[k]*(w[k]-w[k-1]) # no moisture equation
    #
    # Update a, b and ri for the temperature equation
    #
    for k in range(kmax):
        xx = a[k] + sdvk[k] + sdvk1[k]
        a[k] = xx
        xx = b[k] - sduk[k] - sduk1[k]
        b[k] = xx
        ri[k]=temp[k]*div[k]+(sd[k+1]*r1p[k]                +sd[k]*r2p[k]+r1b[k]*(si[k+1]*cbar-cbs[k+1])                +r2b[k]*(si[k]*cbar-cbs[k]))/delsig[k]                +(287.05/1005.0)*((temp[k]+300.0)*(c[k]-cbar)-temp[k]*dbar)
        ri[k]=ri[k]+heat[k]
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


# In[8]:


#
# Implicit time differencing 
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
    xx = torch.complex(torch.tensor([0.0]).double(),torch.tensor([0.0]).double())
    for k in range(kmax):
        xx = xx + delsig[k]*rhs[k] # rh_tmp is rhs in AFGL
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


# In[9]:


#
# Explicit time differencing ##### Not working yet ...
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
    #
    ### compute rh2 = amtrx @ store2 for each wave number
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
    xx = torch.complex(torch.tensor([0.0]).double(),torch.tensor([0.0]).double())
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


# In[10]:


#
# Robert time filtering
#
def tfilt(filt,fmn1,fmn2,fmn3):
    xxx = fmn2 + filt*(fmn1 - 2*fmn2 + fmn3)
    fmn1 = xxx
    fmn2 = fmn3
    return fmn1,fmn2,fmn3


# In[11]:


#
# Convert spectral vorticity and divergence into u & v on
# Gaussian grid. Note that the total vorticity is passed
# so coriolis parameter needs to be removed
#
def uv(ivsht,vort,div,f_spec,mw,zw,kmax,imax,jmax):
    ae = 6.371E+06
    u = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    v = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    nn = torch.arange(0,mw).reshape(mw, 1).double()
    nn = nn.expand(mw,zw)
    invlap = -(ae*ae)/nn/(nn+1)
    invlap[0]=0.
    for k in range(kmax):
        streamf = invlap*(vort[k] - f_spec)/ae
        vp = invlap*div[k]/ae
        vordiv = torch.stack((streamf,vp))
        uvgrid = ivsht(vordiv).cpu()
        u[k] = uvgrid[0]
        v[k] = uvgrid[1]
    return u,v
#


# In[12]:


#
# Calculate grad(lnPs)
#
def gradq(ivsht,qmn,mw,zw,kmax,imax,jmax):
    ae = 6.371E+06
    zerospec = torch.zeros((mw,zw),dtype=torch.complex128)
    dxq = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    dyq = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    qmna = qmn/ae
    zeroq = torch.stack((zerospec,qmna))
    gradqgrid = ivsht(zeroq).cpu()
    dxq = gradqgrid[0]
    dyq = gradqgrid[1]
    return dxq,dyq
#


# In[13]:


#
# Convert u and v on Gaussian grid to spectral vorticity
# and divergence
#
def vortdivspec(vsht,u,v,kmax,mw,zw):
    vort = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
    div = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
    ae = 6.371E+06
    nn = torch.arange(0,mw).reshape(mw, 1).double()
    nn = nn.expand(mw,zw)
    lap = -(nn*(nn+1))/(ae*ae)
    for k in range(kmax):
        uvgrid = torch.stack((u[k],v[k]))
        xy = (lap*vsht(uvgrid).cpu())*ae        
        vort[k] = xy[0]
        div[k] = xy[1]
    return vort,div


# In[14]:


#
# Convert from grid to spectral and then apply laplacian
#
def lap_sht(sht,e,mw,zw):
    spec_lap = torch.zeros((mw,zw),dtype=torch.complex128)
    ae = 6.371E+06
    nn = torch.arange(0,mw).reshape(mw, 1).double()
    nn = nn.expand(mw,zw)
    lap = -(nn*(nn+1))/(ae*ae)
    spec_lap = (lap*sht(e).cpu())        
    return spec_lap


# In[15]:


#
# Get gridded lnps and geopotential
#
def get_geo_ps(isht,tmn,qmn,phismn,amtrx,kmax,mw,zw,jmax,imax):
    geosmn = torch.zeros((kmax,mw,zw),dtype=torch.complex128)
    geo = torch.zeros((kmax,jmax,imax),dtype=torch.float64)
    lnps = isht(qmn).cpu()
    qq = tmn
    rr = torch.from_numpy(amtrx)
    tmp_r = torch.einsum('kl,lji->kji',[rr,qq.real])
    tmp_i = torch.einsum('kl,lji->kji',[rr,qq.imag])
    rh2 = torch.complex(tmp_r, tmp_i)
    for k in range(kmax):
        geosmn[k] = phismn + rh2[k]
    for k in range(kmax):
        geo[k] = isht(geosmn[k]).cpu()
    #
    return lnps,geo


# In[16]:


#
# Convert spectral data to Gaussian grid and write to disk
# as netcdf data
#
def postprocessing(isht,ivsht,zmnt,dmnt,tmnt,qmnt,wmnt,                   phismn,f_spec,amtrx,                   times,mw,zw,kmax,imax,jmax,sl,lats,lons,tl,                  datapath):
    u = torch.zeros((tl,kmax,jmax,imax),dtype=torch.float64)
    v = torch.zeros((tl,kmax,jmax,imax),dtype=torch.float64)
    vort = torch.zeros((tl,kmax,jmax,imax),dtype=torch.float64)
    div = torch.zeros((tl,kmax,jmax,imax),dtype=torch.float64)
    temp = torch.zeros((tl,kmax,jmax,imax),dtype=torch.float64)
    geo = torch.zeros((tl,kmax,jmax,imax),dtype=torch.float64)
    lnps = torch.zeros((tl,jmax,imax),dtype=torch.float64)
    for it in range(tl):
        u[it],v[it] = uv(ivsht,zmnt[it],dmnt[it],f_spec,mw,zw,kmax,imax,jmax)
        for k in range(kmax):
#            vort[it,k] = isht(zmnt[it,k]).cpu() # uncomment if vort wanted
#            div[it,k] = isht(dmnt[it,k]).cpu() # uncomment of div wanted
            temp[it,k] = isht(tmnt[it,k]).cpu()
        lnps[it],geo[it] = get_geo_ps(isht,tmnt[it],qmnt[it],phismn,amtrx,kmax,mw,zw,                         jmax,imax)
    #
    tstamp_start = str(times[0])[0:10]
    tstamp_end = str(times[tl-1])[0:10]
    stamp = tstamp_start+'_'+tstamp_end
    du = xarray.Dataset({'u': (['time','lev','lat','lon'],u.numpy())},                        coords={'time': times,'lev':sl, 'lat': lats, 'lon': lons})
    dv = xarray.Dataset({'v': (['time','lev','lat','lon'],v.numpy())},                        coords={'time': times,'lev':sl, 'lat': lats, 'lon': lons})
#    dvort = xarray.Dataset({'vort': (['time','lev','lat','lon'],vort.numpy())},\
#                        coords={'time': times,'lev':sl, 'lat': lats, 'lon': lons}) # uncomment of vort wanted
#    ddiv = xarray.Dataset({'div': (['time','lev','lat','lon'],div.numpy())},\
#                        coords={'time': times,'lev':sl, 'lat': lats, 'lon': lons}) # uncomment of div wanted
    dtemp = xarray.Dataset({'t': (['time','lev','lat','lon'],temp.numpy())},                        coords={'time': times,'lev':sl, 'lat': lats, 'lon': lons})
    dgeo = xarray.Dataset({'geo': (['time','lev','lat','lon'],geo.numpy())},                        coords={'time': times,'lev':sl, 'lat': lats, 'lon': lons})
    dps = xarray.Dataset({'lnps': (['time','lat','lon'],lnps.numpy())},                        coords={'time': times,'lat': lats, 'lon': lons})
    du.to_netcdf(datapath+'uvel_'+stamp+'.nc')
    dv.to_netcdf(datapath+'vvel_'+stamp+'.nc')
    dtemp.to_netcdf(datapath+'temp_'+stamp+'.nc')
    dgeo.to_netcdf(datapath+'geo_'+stamp+'.nc')
    dps.to_netcdf(datapath+'lnps_'+stamp+'.nc')
#    dvort.to_netcdf(datapath+'vort_'+stamp+'.nc') # uncomment of vort wanted
#    ddiv.to_netcdf(datapath+'div_'+stamp+'.nc') # uncomment of div wanted
    return

