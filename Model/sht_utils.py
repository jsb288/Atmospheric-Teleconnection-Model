# coding=utf-8
# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# Note: All required torch-harmonics functions assembled in this script by Leo Siqueira (lsiqueira@earth.miami.edu)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

def legendre_gauss_weights(n, a=-1.0, b=1.0):
    r"""
    Helper routine which returns the Legendre-Gauss nodes and weights
    on the interval [a, b]
    """

    xlg, wlg = np.polynomial.legendre.leggauss(n)
    xlg = (b - a) * 0.5 * xlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5

    return xlg, wlg

def lobatto_weights(n, a=-1.0, b=1.0, tol=1e-16, maxiter=100):
    r"""
    Helper routine which returns the Legendre-Gauss-Lobatto nodes and weights
    on the interval [a, b]
    """

    wlg = np.zeros((n,))
    tlg = np.zeros((n,))
    tmp = np.zeros((n,))

    # Vandermonde Matrix
    vdm = np.zeros((n, n))
  
    # initialize Chebyshev nodes as first guess
    for i in range(n): 
        tlg[i] = -np.cos(np.pi*i / (n-1))
    
    tmp = 2.0
    
    for i in range(maxiter):
        tmp = tlg
       
        vdm[:,0] = 1.0 
        vdm[:,1] = tlg
       
        for k in range(2, n):
            vdm[:, k] = ( (2*k-1) * tlg * vdm[:, k-1] - (k-1) * vdm[:, k-2] ) / k
       
        tlg = tmp - ( tlg*vdm[:, n-1] - vdm[:, n-2] ) / ( n * vdm[:, n-1]) 
        
        if (max(abs(tlg - tmp).flatten()) < tol ):
            break 
    
    wlg = 2.0 / ( (n*(n-1))*(vdm[:, n-1]**2))

    # rescale
    tlg = (b - a) * 0.5 * tlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5
    
    return tlg, wlg

def clm(l, m):
    """
    defines the normalization factor to orthonormalize the Spherical Harmonics
    """
    return np.sqrt((2*l + 1) / 4 / np.pi) * np.sqrt(np.math.factorial(l-m) / np.math.factorial(l+m))


def precompute_legpoly(mmax, lmax, t, norm="ortho", inverse=False, csphase=True):
    r"""
    Computes the values of (-1)^m c^l_m P^l_m(\cos \theta) at the positions specified by x (theta)
    The resulting tensor has shape (mmax, lmax, len(x)).
    The Condon-Shortley Phase (-1)^m can be turned off optionally

    method of computation follows
    [1] Schaeffer, N.; Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Rapp, R.H.; A Fortran Program for the Computation of Gravimetric Quantities from High Degree Spherical Harmonic Expansions, Ohio State University Columbus; report; 1982;
        https://apps.dtic.mil/sti/citations/ADA123406
    [3] Schrama, E.; Orbit integration based upon interpolated gravitational gradients
    """

    # compute the tensor P^m_n:
    nmax = max(mmax,lmax)
    pct = np.zeros((nmax, nmax, len(t)), dtype=np.float64)

    sint = np.sin(t)
    cost = np.cos(t)
        
    norm_factor = 1. if norm == "ortho" else np.sqrt(4 * np.pi)
    norm_factor = 1. / norm_factor if inverse else norm_factor

    # initial values to start the recursion
    pct[0,0,:] = norm_factor / np.sqrt(4 * np.pi)

    # fill the diagonal and the lower diagonal
    for l in range(1, nmax):
        pct[l-1, l, :] = np.sqrt(2*l + 1) * cost * pct[l-1, l-1, :]
        pct[l, l, :] = np.sqrt( (2*l + 1) * (1 + cost) * (1 - cost) / 2 / l ) * pct[l-1, l-1, :]

    # fill the remaining values on the upper triangle and multiply b
    for l in range(2, nmax):
        for m in range(0, l-1):
            pct[m, l, :] = cost * np.sqrt((2*l - 1) / (l - m) * (2*l + 1) / (l + m)) * pct[m, l-1, :] \
                            - np.sqrt((l + m - 1) / (l - m) * (2*l + 1) / (2*l - 3) * (l - m - 1) / (l + m)) * pct[m, l-2, :]

    if norm == "schmidt":
        for l in range(0, nmax):
            if inverse:
                pct[:, l, : ] = pct[:, l, : ] * np.sqrt(2*l + 1)
            else:
                pct[:, l, : ] = pct[:, l, : ] / np.sqrt(2*l + 1)

    pct = pct[:mmax, :lmax]

    if csphase:
        for m in range(1, mmax, 2):
            pct[m] *= -1

    return torch.from_numpy(pct)

def precompute_dlegpoly(mmax, lmax, x, norm="ortho", inverse=False, csphase=True):
    r"""
    Computes the values of the derivatives $\frac{d}{d \theta} P^m_l(\cos \theta)$
    at the positions specified by x (theta), as well as $\frac{1}{\sin \theta} P^m_l(\cos \theta)$,
    needed for the computation of the vector spherical harmonics. The resulting tensor has shape
    (2, mmax, lmax, len(x)).

    computation follows
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    pct = precompute_legpoly(mmax+1, lmax+1, x, norm=norm, inverse=inverse, csphase=False)

    dpct = torch.zeros((2, mmax, lmax, len(x)), dtype=torch.float64)

    # fill the derivative terms wrt theta
    for l in range(0, lmax):

        # m = 0
        dpct[0, 0, l] = - np.sqrt(l*(l+1)) * pct[1, l]

        # 0 < m < l
        for m in range(1, min(l, mmax)):
            dpct[0, m, l] = 0.5 * ( np.sqrt((l+m)*(l-m+1)) * pct[m-1, l] - np.sqrt((l-m)*(l+m+1)) * pct[m+1, l] )

        # m == l
        if mmax > l:
            dpct[0, l, l] = np.sqrt(l/2) * pct[l-1, l]

        # fill the - 1j m P^m_l / sin(phi). as this component is purely imaginary,
        # we won't store it explicitly in a complex array
        for m in range(1, min(l+1, mmax)):
            # this component is implicitly complex
            # we do not divide by m here as this cancels with the derivative of the exponential
            dpct[1, m, l] = 0.5 * np.sqrt((2*l+1)/(2*l+3)) * \
                ( np.sqrt((l-m+1)*(l-m+2)) * pct[m-1, l+1] + np.sqrt((l+m+1)*(l+m+2)) * pct[m+1, l+1] )

    if csphase:
        for m in range(1, mmax, 2):
            dpct[:, m] *= -1

    return dpct

# Single Thread Transforms:

class RealSHT(nn.Module):
    r"""
    Defines a module for computing the forward (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):
        r"""
        Initializes the SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        nlat: input grid resolution in the latitudinal direction
        nlon: input grid resolution in the longitudinal direction
        grid: grid in the latitude direction (for now only tensor product grids are supported)
        """

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # TODO: include assertions regarding the dimensions

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
            # cost, w = fejer2_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        tq = np.flip(np.arccos(cost))

        # determine the dimensions 
        self.mmax = mmax or self.nlon // 2 + 1

        # combine quadrature weights with the legendre weights
        weights = torch.from_numpy(w)
        pct = precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        weights = torch.einsum('mlk,k->mlk', pct, weights)

        # remember quadrature weights
        self.register_buffer('weights', weights, persistent=False)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        assert(x.shape[-2] == self.nlat)
        assert(x.shape[-1] == self.nlon)

        # apply real fft in the longitudinal direction
        x = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")
        
        # do the Legendre-Gauss quadrature
        x = torch.view_as_real(x)
        
        # distributed contraction: fork
        out_shape = list(x.size())
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax
        xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
        
        # contraction
        xout[..., 0] = torch.einsum('...km,mlk->...lm', x[..., :self.mmax, 0], self.weights.to(x.dtype) )
        xout[..., 1] = torch.einsum('...km,mlk->...lm', x[..., :self.mmax, 1], self.weights.to(x.dtype) )
        x = torch.view_as_complex(xout)
        
        return x

class InverseRealSHT(nn.Module):
    r"""
    Defines a module for computing the inverse (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    nlat, nlon: Output dimensions
    lmax, mmax: Input dimensions (spherical coefficients). For convenience, these are inferred from the output dimensions

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        t = np.flip(np.arccos(cost))

        # determine the dimensions 
        self.mmax = mmax or self.nlon // 2 + 1

        pct = precompute_legpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)

        # register buffer
        self.register_buffer('pct', pct, persistent=False)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        assert(x.shape[-2] == self.lmax)
        assert(x.shape[-1] == self.mmax)
        
        # Evaluate associated Legendre functions on the output nodes
        x = torch.view_as_real(x)
        
        rl = torch.einsum('...lm, mlk->...km', x[..., 0], self.pct.to(x.dtype) )
        im = torch.einsum('...lm, mlk->...km', x[..., 1], self.pct.to(x.dtype) )
        xs = torch.stack((rl, im), -1)

        # apply the inverse (real) FFT
        x = torch.view_as_complex(xs)
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        return x


class RealVectorSHT(nn.Module):
    r"""
    Defines a module for computing the forward (real) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last three dimensions of the input.

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):
        r"""
        Initializes the vector SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        nlat: input grid resolution in the latitudinal direction
        nlon: input grid resolution in the longitudinal direction
        grid: type of grid the data lives on
        """

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
            # cost, w = fejer2_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        tq = np.flip(np.arccos(cost))

        # determine the dimensions 
        self.mmax = mmax or self.nlon // 2 + 1

        weights = torch.from_numpy(w)
        dpct = precompute_dlegpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        
        # combine integration weights, normalization factor in to one:
        l = torch.arange(0, self.lmax)
        norm_factor = 1. / l / (l+1)
        norm_factor[0] = 1.
        weights = torch.einsum('dmlk,k,l->dmlk', dpct, weights, norm_factor)
        # since the second component is imaginary, we need to take complex conjugation into account
        weights[1] = -1 * weights[1]

        # remember quadrature weights
        self.register_buffer('weights', weights, persistent=False)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        assert(len(x.shape) >= 3)

        # apply real fft in the longitudinal direction
        x = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")
        
        # do the Legendre-Gauss quadrature
        x = torch.view_as_real(x)
        
        # distributed contraction: fork
        out_shape = list(x.size())
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax
        xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # contraction - spheroidal component
        # real component
        xout[..., 0, :, :, 0] =   torch.einsum('...km,mlk->...lm', x[..., 0, :, :self.mmax, 0], self.weights[0].to(x.dtype)) \
                                - torch.einsum('...km,mlk->...lm', x[..., 1, :, :self.mmax, 1], self.weights[1].to(x.dtype)) 

        # iamg component
        xout[..., 0, :, :, 1] =   torch.einsum('...km,mlk->...lm', x[..., 0, :, :self.mmax, 1], self.weights[0].to(x.dtype)) \
                                + torch.einsum('...km,mlk->...lm', x[..., 1, :, :self.mmax, 0], self.weights[1].to(x.dtype)) 

        # contraction - toroidal component
        # real component
        xout[..., 1, :, :, 0] = - torch.einsum('...km,mlk->...lm', x[..., 0, :, :self.mmax, 1], self.weights[1].to(x.dtype)) \
                                - torch.einsum('...km,mlk->...lm', x[..., 1, :, :self.mmax, 0], self.weights[0].to(x.dtype)) 
        # imag component
        xout[..., 1, :, :, 1] =   torch.einsum('...km,mlk->...lm', x[..., 0, :, :self.mmax, 0], self.weights[1].to(x.dtype)) \
                                - torch.einsum('...km,mlk->...lm', x[..., 1, :, :self.mmax, 1], self.weights[0].to(x.dtype)) 

        return torch.view_as_complex(xout)


class InverseRealVectorSHT(nn.Module):
    r"""
    Defines a module for computing the inverse (real-valued) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    
    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """
    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        t = np.flip(np.arccos(cost))

        # determine the dimensions 
        self.mmax = mmax or self.nlon // 2 + 1

        dpct = precompute_dlegpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)

        # register weights
        self.register_buffer('dpct', dpct, persistent=False)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        assert(x.shape[-2] == self.lmax)
        assert(x.shape[-1] == self.mmax)
        
        # Evaluate associated Legendre functions on the output nodes
        x = torch.view_as_real(x)

        # contraction - spheroidal component
        # real component
        srl =   torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 0], self.dpct[0].to(x.dtype)) \
              - torch.einsum('...lm,mlk->...km', x[..., 1, :, :, 1], self.dpct[1].to(x.dtype)) 
        # iamg component
        sim =   torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 1], self.dpct[0].to(x.dtype)) \
              + torch.einsum('...lm,mlk->...km', x[..., 1, :, :, 0], self.dpct[1].to(x.dtype)) 

        # contraction - toroidal component
        # real component
        trl = - torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 1], self.dpct[1].to(x.dtype)) \
              - torch.einsum('...lm,mlk->...km', x[..., 1, :, :, 0], self.dpct[0].to(x.dtype)) 
        # imag component
        tim =   torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 0], self.dpct[1].to(x.dtype)) \
              - torch.einsum('...lm,mlk->...km', x[..., 1, :, :, 1], self.dpct[0].to(x.dtype)) 
        
        # reassemble
        s = torch.stack((srl, sim), -1)
        t = torch.stack((trl, tim), -1)
        xs = torch.stack((s, t), -4)

        # apply the inverse (real) FFT
        x = torch.view_as_complex(xs)
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        return x
        
# Distributed Transforms:

# those need to be global
_POLAR_PARALLEL_GROUP = None
_AZIMUTH_PARALLEL_GROUP = None
_IS_INITIALIZED = False

def polar_group():
    return _POLAR_PARALLEL_GROUP

def azimuth_group():
    return _AZIMUTH_PARALLEL_GROUP

def init(polar_process_group, azimuth_process_group):
    global _POLAR_PARALLEL_GROUP
    global _AZIMUTH_PARALLEL_GROUP
    _POLAR_PARALLEL_GROUP = polar_process_group
    _AZIMUTH_PARALLEL_GROUP = azimuth_process_group
    _IS_INITIALIZED = True

def is_initialized() -> bool:
    return _IS_INITIALIZED

def is_distributed_polar() -> bool:
    return (_POLAR_PARALLEL_GROUP is not None)

def is_distributed_azimuth() -> bool:
    return (_AZIMUTH_PARALLEL_GROUP is not None)

def polar_group_size() -> int:
    if not is_distributed_polar():
        return 1
    else:
        return dist.get_world_size(group = _POLAR_PARALLEL_GROUP)

def azimuth_group_size() -> int:
    if not is_distributed_azimuth():
        return 1
    else:
        return dist.get_world_size(group = _AZIMUTH_PARALLEL_GROUP)

def polar_group_rank() -> int:
    if not is_distributed_polar():
        return 0
    else:
        return dist.get_rank(group = _POLAR_PARALLEL_GROUP)

def azimuth_group_rank() -> int:
    if not is_distributed_azimuth():
        return 0
    else:
        return dist.get_rank(group = _AZIMUTH_PARALLEL_GROUP)

class DistributedRealSHT(nn.Module):
    """
    Defines a module for computing the forward (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="legendre-gauss", norm="ortho", csphase=True):
        """
        Initializes the SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        nlat: input grid resolution in the latitudinal direction
        nlon: input grid resolution in the longitudinal direction
        grid: grid in the latitude direction (for now only tensor product grids are supported)
        """

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # TODO: include assertions regarding the dimensions

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
            # cost, w = fejer2_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        tq = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # spatial paddings
        latdist = (self.nlat + self.comm_size_polar - 1) // self.comm_size_polar
        self.nlatpad = latdist * self.comm_size_polar - self.nlat
        londist = (self.nlon + self.comm_size_azimuth - 1) // self.comm_size_azimuth
        self.nlonpad = londist * self.comm_size_azimuth - self.nlon

        # frequency paddings
        ldist = (self.lmax + self.comm_size_polar - 1) // self.comm_size_polar
        self.lpad = ldist * self.comm_size_polar - self.lmax
        mdist = (self.mmax + self.comm_size_azimuth - 1) // self.comm_size_azimuth
        self.mpad = mdist * self.comm_size_azimuth - self.mmax

        # combine quadrature weights with the legendre weights
        weights = torch.from_numpy(w)
        pct = precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        weights = torch.einsum('mlk,k->mlk', pct, weights)

        # we need to split in m, pad before:
        weights = F.pad(weights, [0, 0, 0, 0, 0, self.mpad], mode="constant")
        weights = torch.split(weights, (self.mmax+self.mpad) // self.comm_size_azimuth, dim=0)[self.comm_rank_azimuth]

        # compute the local pad and size
        # spatial
        self.nlat_local = min(latdist, self.nlat - self.comm_rank_polar * latdist)
        self.nlatpad_local = latdist - self.nlat_local
        self.nlon_local = min(londist, self.nlon - self.comm_rank_azimuth * londist)
        self.nlonpad_local = londist - self.nlon_local

        # frequency
        self.lmax_local = min(ldist, self.lmax - self.comm_rank_polar * ldist)
        self.lpad_local = ldist - self.lmax_local
        self.mmax_local	= min(mdist, self.mmax - self.comm_rank_azimuth * mdist)
        self.mpad_local	= mdist	- self.mmax_local

        # remember quadrature weights
        self.register_buffer('weights', weights, persistent=False)

    def extra_repr(self):
        """
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        # we need to ensure that we can split the channels evenly
        assert(x.shape[1] % self.comm_size_polar == 0)
        assert(x.shape[1] % self.comm_size_azimuth == 0)

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            xt = distributed_transpose_azimuth.apply(x, (1, -1))
        else:
            xt = x

        # apply real fft in the longitudinal direction: make sure to truncate to nlon
        xtf = 2.0 * torch.pi * torch.fft.rfft(xt, n=self.nlon, dim=-1, norm="forward")

        # truncate
        xtft = xtf[..., :self.mmax]

        # pad the dim to allow for splitting
        xtfp = F.pad(xtft, [0, self.mpad], mode="constant")

        # transpose: after this, m is split and c is local
        if self.comm_size_azimuth > 1:
            y = distributed_transpose_azimuth.apply(xtfp, (-1, 1))
        else:
            y = xtfp

        # transpose: after this, c is split and h is local
        if self.comm_size_polar > 1:
            yt = distributed_transpose_polar.apply(y, (1, -2))
        else:
            yt = y

        # the input data might be padded, make sure to truncate to nlat:
        ytt = yt[..., :self.nlat, :]

        # do the Legendre-Gauss quadrature
        yttr = torch.view_as_real(ytt)

        # contraction
        yor = torch.einsum('...kmr,mlk->...lmr', yttr, self.weights.to(yttr.dtype)).contiguous()

        # pad if required, truncation is implicit
        yopr = F.pad(yor, [0, 0, 0, 0, 0, self.lpad], mode="constant")
        yop = torch.view_as_complex(yopr)

        # transpose: after this, l is split and c is local
        if self.comm_size_polar	> 1:
            y = distributed_transpose_polar.apply(yop, (-2, 1))
        else:
            y = yop

        return y


class DistributedInverseRealSHT(nn.Module):
    """
    Defines a module for computing the inverse (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    nlat, nlon: Output dimensions
    lmax, mmax: Input dimensions (spherical coefficients). For convenience, these are inferred from the output dimensions

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="legendre-gauss", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        t = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # spatial paddings
        latdist = (self.nlat + self.comm_size_polar - 1) // self.comm_size_polar
        self.nlatpad = latdist * self.comm_size_polar - self.nlat
        londist = (self.nlon + self.comm_size_azimuth - 1) // self.comm_size_azimuth
        self.nlonpad = londist * self.comm_size_azimuth - self.nlon

        # frequency paddings
        ldist = (self.lmax + self.comm_size_polar - 1) // self.comm_size_polar
        self.lpad = ldist * self.comm_size_polar - self.lmax
        mdist = (self.mmax + self.comm_size_azimuth - 1) // self.comm_size_azimuth
        self.mpad = mdist * self.comm_size_azimuth - self.mmax

        # compute legende polynomials
        pct = precompute_legpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)

        # split in m
        pct = F.pad(pct, [0, 0, 0, 0, 0, self.mpad], mode="constant")
        pct = torch.split(pct, (self.mmax+self.mpad) // self.comm_size_azimuth, dim=0)[self.comm_rank_azimuth]

        # compute the local pads and sizes
        # spatial
        self.nlat_local = min(latdist, self.nlat - self.comm_rank_polar * latdist)
        self.nlatpad_local = latdist - self.nlat_local
        self.nlon_local = min(londist, self.nlon - self.comm_rank_azimuth * londist)
        self.nlonpad_local = londist - self.nlon_local

        # frequency
        self.lmax_local = min(ldist, self.lmax - self.comm_rank_polar * ldist)
        self.lpad_local = ldist - self.lmax_local
        self.mmax_local = min(mdist, self.mmax - self.comm_rank_azimuth * mdist)
        self.mpad_local = mdist - self.mmax_local

        # register
        self.register_buffer('pct', pct, persistent=False)

    def extra_repr(self):
        """
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        # we need to ensure that we can split the channels evenly
        assert(x.shape[1] % self.comm_size_polar == 0)
        assert(x.shape[1] % self.comm_size_azimuth == 0)

        # transpose: after that, channels are split, l is local:
        if self.comm_size_polar > 1:
            xt = distributed_transpose_polar.apply(x, (1, -2))
        else:
            xt = x

        # remove padding in l:
        xtt = xt[..., :self.lmax, :]

        # Evaluate associated Legendre functions on the output nodes
        xttr = torch.view_as_real(xtt)

        # einsum
        xs = torch.einsum('...lmr, mlk->...kmr', xttr, self.pct.to(xttr.dtype)).contiguous()
        x = torch.view_as_complex(xs)

        # transpose: after this, l is split and channels are local
        xp = F.pad(x, [0, 0, 0, self.nlatpad])

        if self.comm_size_polar > 1:
            y = distributed_transpose_polar.apply(xp, (-2, 1))
        else:
            y = xp

        # transpose: after this, channels are split and m is local
        if self.comm_size_azimuth > 1:
            yt = distributed_transpose_azimuth.apply(y, (1, -1))
        else:
            yt = y

        # truncate
        ytt = yt[..., :self.mmax]

        # apply the inverse (real) FFT
        x = torch.fft.irfft(ytt, n=self.nlon, dim=-1, norm="forward")

        # pad before we transpose back
        xp = F.pad(x, [0, self.nlonpad])

        # transpose: after this, m is split and channels are local
        if self.comm_size_azimuth > 1:
            out = distributed_transpose_azimuth.apply(xp, (-1, 1))
        else:
            out = xp

        return out


class DistributedRealVectorSHT(nn.Module):
    """
    Defines a module for computing the forward (real) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last three dimensions of the input.

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="legendre-gauss", norm="ortho", csphase=True):
        """
        Initializes the vector SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        nlat: input grid resolution in the latitudinal direction
        nlon: input grid resolution in the longitudinal direction
        grid: type of grid the data lives on
        """

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
            # cost, w = fejer2_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        tq = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # spatial paddings
        latdist = (self.nlat + self.comm_size_polar - 1) // self.comm_size_polar
        self.nlatpad = latdist * self.comm_size_polar - self.nlat
        londist = (self.nlon + self.comm_size_azimuth - 1) // self.comm_size_azimuth
        self.nlonpad = londist * self.comm_size_azimuth - self.nlon

        # frequency paddings
        ldist = (self.lmax + self.comm_size_polar - 1) // self.comm_size_polar
        self.lpad = ldist * self.comm_size_polar - self.lmax
        mdist = (self.mmax + self.comm_size_azimuth - 1) // self.comm_size_azimuth
        self.mpad = mdist * self.comm_size_azimuth - self.mmax

        weights = torch.from_numpy(w)
        dpct = precompute_dlegpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)

        # combine integration weights, normalization factor in to one:
        l = torch.arange(0, self.lmax)
        norm_factor = 1. / l / (l+1)
        norm_factor[0] = 1.
        weights = torch.einsum('dmlk,k,l->dmlk', dpct, weights, norm_factor)
        # since the second component is imaginary, we need to take complex conjugation into account
        weights[1] = -1 * weights[1]

        # we need to split in m, pad before:
        weights = F.pad(weights, [0, 0, 0, 0, 0, self.mpad], mode="constant")
        weights = torch.split(weights, (self.mmax+self.mpad) // self.comm_size_azimuth, dim=1)[self.comm_rank_azimuth]

        # remember quadrature weights
        self.register_buffer('weights', weights, persistent=False)

        # compute the local pad and size
        # spatial
        self.nlat_local = min(latdist, self.nlat - self.comm_rank_polar * latdist)
        self.nlatpad_local = latdist - self.nlat_local
        self.nlon_local = min(londist, self.nlon - self.comm_rank_azimuth * londist)
        self.nlonpad_local = londist - self.nlon_local

        # frequency
        self.lmax_local = min(ldist, self.lmax - self.comm_rank_polar * ldist)
        self.lpad_local = ldist - self.lmax_local
        self.mmax_local = min(mdist, self.mmax - self.comm_rank_azimuth * mdist)
        self.mpad_local = mdist - self.mmax_local

    def extra_repr(self):
        """
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        assert(len(x.shape) >= 3)
        assert(x.shape[1] % self.comm_size_polar == 0)
        assert(x.shape[1] % self.comm_size_azimuth == 0)

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            xt = distributed_transpose_azimuth.apply(x, (1, -1))
        else:
            xt = x

        # apply real fft in the longitudinal direction: make sure to truncate to nlon
        xtf = 2.0 * torch.pi * torch.fft.rfft(xt, n=self.nlon, dim=-1, norm="forward")

        # truncate
        xtft = xtf[..., :self.mmax]

        # pad the dim to allow for splitting
        xtfp = F.pad(xtft, [0, self.mpad], mode="constant")

        # transpose: after this, m is split and c is local
        if self.comm_size_azimuth > 1:
            y = distributed_transpose_azimuth.apply(xtfp, (-1, 1))
        else:
            y = xtfp

        # transpose: after this, c is split and h is local
        if self.comm_size_polar > 1:
            yt = distributed_transpose_polar.apply(y, (1, -2))
        else:
            yt = y

        # the input data might be padded, make sure to truncate to nlat:
        ytt = yt[..., :self.nlat, :]

        # do the Legendre-Gauss quadrature
        yttr = torch.view_as_real(ytt)

        # create output array
        yor = torch.zeros_like(yttr, dtype=yttr.dtype, device=yttr.device)

        # contraction - spheroidal component
        # real component
        yor[..., 0, :, :, 0] =   torch.einsum('...km,mlk->...lm', yttr[..., 0, :, :, 0], self.weights[0].to(yttr.dtype)) \
                               - torch.einsum('...km,mlk->...lm', yttr[..., 1, :, :, 1], self.weights[1].to(yttr.dtype))
        # iamg component
        yor[..., 0, :, :, 1] =   torch.einsum('...km,mlk->...lm', yttr[..., 0, :, :, 1], self.weights[0].to(yttr.dtype)) \
                               + torch.einsum('...km,mlk->...lm', yttr[..., 1, :, :, 0], self.weights[1].to(yttr.dtype))

        # contraction - toroidal component
        # real component
        yor[..., 1, :, :, 0] = - torch.einsum('...km,mlk->...lm', yttr[..., 0, :, :, 1], self.weights[1].to(yttr.dtype)) \
                               - torch.einsum('...km,mlk->...lm', yttr[..., 1, :, :, 0], self.weights[0].to(yttr.dtype))
        # imag component
        yor[..., 1, :, :, 1] =   torch.einsum('...km,mlk->...lm', yttr[..., 0, :, :, 0], self.weights[1].to(yttr.dtype)) \
                               - torch.einsum('...km,mlk->...lm', yttr[..., 1, :, :, 1], self.weights[0].to(yttr.dtype))

        # pad if required
        yopr = F.pad(yor, [0, 0, 0, 0, 0, self.lpad], mode="constant")
        yop = torch.view_as_complex(yopr)

        # transpose: after this, l is split and c is local
        if self.comm_size_polar > 1:
            y = distributed_transpose_polar.apply(yop, (-2, 1))
        else:
            y = yop

        return y


class DistributedInverseRealVectorSHT(nn.Module):
    """
    Defines a module for computing the inverse (real-valued) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """
    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="legendre-gauss", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        t = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # spatial paddings
        latdist = (self.nlat + self.comm_size_polar - 1) // self.comm_size_polar
        self.nlatpad = latdist * self.comm_size_polar - self.nlat
        londist = (self.nlon + self.comm_size_azimuth - 1) // self.comm_size_azimuth
        self.nlonpad = londist * self.comm_size_azimuth - self.nlon

        # frequency paddings
        ldist = (self.lmax + self.comm_size_polar - 1) // self.comm_size_polar
        self.lpad = ldist * self.comm_size_polar - self.lmax
        mdist = (self.mmax + self.comm_size_azimuth - 1) // self.comm_size_azimuth
        self.mpad = mdist * self.comm_size_azimuth - self.mmax

        # compute legende polynomials
        dpct = precompute_dlegpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)

        # split in m
        pct = F.pad(pct, [0, 0, 0, 0, 0, self.mpad], mode="constant")
        pct = torch.split(pct, (self.mmax+self.mpad) // self.comm_size_azimuth, dim=0)[self.comm_rank_azimuth]

        # register buffer
        self.register_buffer('dpct', dpct, persistent=False)

        # compute the local pad and size
        # spatial
        self.nlat_local = min(latdist, self.nlat - self.comm_rank_polar * latdist)
        self.nlatpad_local = latdist - self.nlat_local
        self.nlon_local = min(londist, self.nlon - self.comm_rank_azimuth * londist)
        self.nlonpad_local = londist - self.nlon_local

        # frequency
        self.lmax_local = min(ldist, self.lmax - self.comm_rank_polar * ldist)
        self.lpad_local = ldist - self.lmax_local
        self.mmax_local = min(mdist, self.mmax - self.comm_rank_azimuth * mdist)
        self.mpad_local = mdist - self.mmax_local

    def extra_repr(self):
        """
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        assert(x.shape[1] % self.comm_size_polar == 0)
        assert(x.shape[1] % self.comm_size_azimuth == 0)

        # transpose: after that, channels are split, l is local:
        if self.comm_size_polar > 1:
            xt = distributed_transpose_polar.apply(x, (1, -2))
        else:
            xt = x

        # remove padding in l:
        xtt = xt[..., :self.lmax, :]

        # Evaluate associated Legendre functions on the output nodes
        xttr = torch.view_as_real(xtt)

        # contraction - spheroidal component
        # real component
        srl =   torch.einsum('...lm,mlk->...km', xttr[..., 0, :, :, 0], self.dpct[0].to(xttr.dtype)) \
              - torch.einsum('...lm,mlk->...km', xttr[..., 1, :, :, 1], self.dpct[1].to(xttr.dtype))
        # imag component
        sim =   torch.einsum('...lm,mlk->...km', xttr[..., 0, :, :, 1], self.dpct[0].to(xttr.dtype)) \
              + torch.einsum('...lm,mlk->...km', xttr[..., 1, :, :, 0], self.dpct[1].to(xttr.dtype))

        # contraction - toroidal component
        # real component
        trl = - torch.einsum('...lm,mlk->...km', xttr[..., 0, :, :, 1], self.dpct[1].to(xttr.dtype)) \
              - torch.einsum('...lm,mlk->...km', xttr[..., 1, :, :, 0], self.dpct[0].to(xttr.dtype))
        # imag component
        tim =   torch.einsum('...lm,mlk->...km', xttr[..., 0, :, :, 0], self.dpct[1].to(xttr.dtype)) \
              - torch.einsum('...lm,mlk->...km', xttr[..., 1, :, :, 1], self.dpct[0].to(xttr.dtype))

        # reassemble
        s = torch.stack((srl, sim), -1)
        t = torch.stack((trl, tim), -1)
        xs = torch.stack((s, t), -4)

        # convert to complex
        x = torch.view_as_complex(xs)

        # transpose: after this, l is split and channels are local
        xp = F.pad(x, [0, 0, 0, self.nlatpad])

        if self.comm_size_polar > 1:
            y = distributed_transpose_polar.apply(xp, (-2, 1))
        else:
            y = xp

        # transpose: after this, channels are split and m is local
        if self.comm_size_azimuth > 1:
            yt = distributed_transpose_azimuth.apply(y, (1, -1))
        else:
            yt = y

        # truncate
        ytt = yt[..., :self.mmax]

        # apply the inverse (real) FFT
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        # pad before we transpose back
        xp = F.pad(x, [0, self.nlonpad])

        # transpose: after this, m is split and channels are local
        if self.comm_size_azimuth > 1:
            out = distributed_transpose_azimuth.apply(xp, (-1, 1))
        else:
            out = xp

        return out

# general helpers
def get_memory_format(tensor):
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    else:
        return torch.contiguous_format

def split_tensor_along_dim(tensor, dim, num_chunks):
    assert dim < tensor.dim(), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"
    assert (tensor.shape[dim] % num_chunks == 0), f"Error, cannot split dim {dim} evenly. Dim size is \
                                                   {tensor.shape[dim]} and requested numnber of splits is {num_chunks}"
    chunk_size = tensor.shape[dim] // num_chunks
    tensor_list = torch.split(tensor, chunk_size, dim=dim)
    
    return tensor_list

def _transpose(tensor, dim0, dim1, group=None, async_op=False):
    # get input format
    input_format = get_memory_format(tensor)
    
    # get comm params
    comm_size = dist.get_world_size(group=group)

    # split and local transposition
    split_size = tensor.shape[dim0] // comm_size
    x_send = [y.contiguous(memory_format=input_format) for y in torch.split(tensor, split_size, dim=dim0)]
    x_recv = [torch.empty_like(x_send[0]).contiguous(memory_format=input_format) for _ in range(comm_size)]
    
    # global transposition
    req = dist.all_to_all(x_recv, x_send, group=group, async_op=async_op)
    
    return x_recv, req 


class distributed_transpose_azimuth(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dim):
        xlist, _ = _transpose(x, dim[0], dim[1], group=azimuth_group())
        x = torch.cat(xlist, dim=dim[1])
        ctx.dim = dim
        return x

    @staticmethod
    def backward(ctx, go):
        dim = ctx.dim
        gilist, _ = _transpose(go, dim[1], dim[0], group=azimuth_group())
        gi = torch.cat(gilist, dim=dim[0])
        return gi, None

    
class distributed_transpose_polar(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dim):
        xlist, _ = _transpose(x, dim[0], dim[1], group=polar_group())
        x = torch.cat(xlist, dim=dim[1])
        ctx.dim = dim
        return x

    @staticmethod
    def backward(ctx, go):
        dim = ctx.dim
        gilist, _ = _transpose(go, dim[1], dim[0], group=polar_group())
        gi = torch.cat(gilist, dim=dim[0])
        return gi, None