"""
iceearthmani.py
Author: Samuel B. Kachuck
Date: Nov 8, 2017

Visualize the model manifold for earth and ice parameterization. 
"""

import numpy as np
import sys, os, subprocess, time
from jinja2 import Template
import emcee
import pickle
import datetime


import giapy
from giapy.earth_tools.earthSphericalLap import SphericalEarth
from giapy.map_tools import haversine
from giapy.icehistory import PersistentIceHistory
from giapy.plot_tools.interp_path import map_path
from giapy.giamc import sampleOut
import giapy.numTools.dfridr as giader
from giapy.numTools.minimize import lm_minimize, geolm_minimize

from geodesic import geodesic, InitialVelocity

# Each case is a tuple of (theta_0, phi_0, h_0)
#                         (cent colat, lon, height)
ice_spatial_cases = {'L1': (0, 0, 1500), 
                     'L2': (25, 75, 1500),
                     'L3': (25, 75, 500)}

topo_spatial_cases = {'B0': (0, 0, 0, 0),
                      'B1': (100, 320, 760, 1200),
                      'B2': (35, 25, 760, 1200),
                      'B3': (35, 25, 3800, 6000)}
NLON, NLAT = 360, 360

TRUE_MODEL = np.array([np.log10(2.), np.log10(0.1), 70, 1500, np.log(15-10.0)])

SIG=15.

# TABOO variables
with open('tabooconfig','r') as f: 
    DRCTRY = f.readline()
TBTEMPLATE = DRCTRY+'tb_template.F90'
DATATEMPLATE = DRCTRY+'data_template.inc'

def sphericalload(Lons, Lats, lonc, latc, h0, alpha=10):
    """
    Return an ice load according to case spatial/evolution.

    Parameters
    ----------
    spatial - code for spatioal distribution
        Spherical cap of height h0, centered at theta_0, phi_0
    evolution - code for time evolution
    """

    #theta0, phi0, h0 = ice_spatial_cases[spatial]
    
    alpha = np.radians(alpha)
    if alpha <1e-2: return np.zeros_like(Lons)

    # delta is angular distance from center of load
    delta = haversine(Lats, latc, Lons, lonc, r=1, radians=True)

    load = h0*np.sqrt((np.cos(delta) - np.cos(alpha)) /
                      (1. - np.cos(alpha))* (delta <= alpha)) 

    return load

def gen_icehistory(spatial='L1', evolution='T1', tstep=0.02, **kwargs):

    Lons, Lats = np.meshgrid(np.linspace(-np.pi, np.pi, NLON),
                             np.linspace(-np.pi/2, np.pi/2, NLAT))

    colatc, lonc, h0 = ice_spatial_cases[spatial]
    h0 = kwargs.get('h0', h0)
    t2 = kwargs.get('t2', 5)
    t2even = tstep*np.round(t2/tstep)

    # convert colat to lat
    latc = 90 - colatc
    # convert degrees to radians
    latc, lonc = np.radians([latc, lonc])

    if evolution=='T1':
        load = np.zeros((2, NLAT, NLON))
        load[1,:,:] = sphericalload(Lons, Lats, lonc, latc, h0)
        times = np.arange(0, 10.04, tstep)[::-1]

        metadata = {'Lon'               : Lons,
                    'Lat'               : Lats,
                    'nlat'              : NLAT,
                    'shape'             : Lons.shape,
                    '_alterationMask'   : np.zeros_like(Lats),
                    'areaProps'         : {},
                    'areaVerts'         : {},
                    'times'             : times,
                    'stageOrder'        : [0]+[1]*(len(times)-1),
                    'path'              : '',
                    'fnames'            : ['','']}

    if evolution=='T2':
        t1 = 15 

        nloadsteps = int((t1-t2even)/tstep)
        load = np.zeros((nloadsteps+1, NLAT, NLON))
        times = np.arange(0, t1+tstep, tstep)[::-1]

        for i, t in enumerate(np.arange(t2even, t1+tstep, tstep)[::-1]):
            alpha = (t1-t)/(t1-t2)*10
            h = (t1-t)/(t1-t2)*h0
            load[i] = sphericalload(Lons, Lats, lonc, latc, h, alpha)
        
        metadata = {'Lon'               : Lons,
                    'Lat'               : Lats,
                    'nlat'              : NLAT,
                    'shape'             : Lons.shape,
                    '_alterationMask'   : np.zeros_like(Lats),
                    'areaProps'         : {},
                    'areaVerts'         : {},
                    'times'             : times,
                    'stageOrder'        : range(nloadsteps)+[nloadsteps-1]*(len(times)-nloadsteps),
                    'path'              : '',
                    'fnames'            : ['','']}
                    
    if evolution=='T3':
        t1 = 15



        nloadsteps = int((t1-t2even)/tstep)
        load = np.zeros((nloadsteps+3, NLAT, NLON))
        times = np.arange(min(0,t2even), t1+tstep, tstep)[::-1] 
       
        for i, t in enumerate(np.arange(t2even, t1+tstep, tstep)[::-1]):
            alpha = (t2-t)/(t2-t1)*10
            h = (t2-t)/(t2-t1)*h0
            load[i] = sphericalload(Lons, Lats, lonc, latc, h, alpha)

        stageOrder = range(nloadsteps)+[-1]*(len(times)-nloadsteps)
        
        #times = np.r_[np.arange(16, 101)[::-1], times]
        #stageOrder = [-1]+([0]*(len(np.arange(16,101))-1))+stageOrder
        
        metadata = {'Lon'               : Lons,
                    'Lat'               : Lats,
                    'nlat'              : NLAT,
                    'shape'             : Lons.shape,
                    '_alterationMask'   : np.zeros_like(Lats),
                    'areaProps'         : {},
                    'areaVerts'         : {},
                    'times'             : times,
                    'stageOrder'        : stageOrder,
                    'path'              : '',
                    'fnames'            : ['','']}

    return PersistentIceHistory(load, metadata)

def gen_sstopo(spatial='B0', sigb=26):
    Lons, Lats = np.meshgrid(np.linspace(-np.pi, np.pi, NLON),
                             np.linspace(-np.pi/2, np.pi/2, NLAT))
    thetab, phib, bmax, b0 = topo_spatial_cases[spatial]
    # convert colat to lat
    thetab = 90 - thetab
    # convert degrees to radians
    thetab, phib, sigb = np.radians([thetab, phib, sigb])

    # delta is angular distance from center of load
    delta = haversine(Lats, thetab, Lons, phib, r=1, radians=True)

    sstopo = bmax - b0*np.exp(-delta**2/2./sigb**2)
    return sstopo

def generative_model(logv2, logv1, lith, h0, logt2):
    earth = gen_taboo_earth(10**logv1, 10**logv2, lith)
    t2 = 15 - np.exp(logt2)
    ice  = gen_icehistory(spatial='L2', evolution='T3', h0=h0, t2=t2, tstep=.2) 

    
    sim = giapy.giasim.GiaSimGlobal(earth, ice, topo=topoB3)
    result = sim.performConvolution(out_times=ice.times, ntrunc=128)

    return sim, result

def observations(sim, result, smooth=False):
    sstopoloc = result.sstopo[:,sinds[0],sinds[1]]

    if not smooth:
        return result.sstopo[-1,sinds[0],sinds[1]] - sstopoloc[tinds]
    else: 
        return result.sstopo[-1,sinds[0],sinds[1]] - sstopoloc[::2]

def predictions(sim, result):
    uplrate = (result.sstopo[-1] - result.sstopo[-2]) / 200.

    avgoceurate = (result.esl[-1] - result.esl[-2]) / 200.

    newrsl = (result.sstopo[-1] - result.sstopo[tinds])[:, pinds[0], pinds[1]]

    preds =  np.r_[newrsl.flatten(), 
                    uplrate[sinds[0], sinds[1]], 
                    uplrate[pinds[0], pinds[1]], 
                    avgoceurate]

    return preds

def obsandpreds(sim, result):
    return np.r_[observations(sim, result).T.flatten(), predictions(sim, result)]

def gen_taboo_earth(v1, v2, lith):


    # make temporary directory. Use hashed time and random number to avoid
    # collision between directories.
    tmpdir = '_tmptaboo'+str(hash(time.time()))+str(np.random.randint(100))

    os.mkdir(tmpdir)
    os.chdir(tmpdir)
    os.mkdir('VSC')

    # Write out TABOO with correct lithosphere.
    with open(TBTEMPLATE, 'r') as f:
        template = Template(f.read())
    with open('tb.F90', 'w') as f:
        f.write(template.render(rlith=6371-float(lith)))
    
    # Write out viscoelastic parameters.
    with open('./VSC/vsc_param_2layer.dat', 'w') as f:
        template = Template("'{{ lith }}',\n'{{ v2 }}',\n'{{ v1 }}',\n'{{ v1 }}'")
        f.write(template.render(lith=lith,v1=v1,v2=v2))

    # Write out viscoelastic parameters path.
    with open(DATATEMPLATE, 'r') as f:
        template = Template(f.read())
    with open('data.inc', 'w') as f:
        viscpath = os.path.abspath('./VSC/vsc_param_2layer.dat')
        f.write(template.render(viscpath=viscpath, viscpathlen=len(viscpath)))


    subprocess.call('gfortran tb.F90 -o taboo_run.out', shell=True)
    subprocess.call('./taboo_run.out', shell=True)

    earth = SphericalEarth()
    earth.loadTabooNumbers()

    os.chdir('..')
    subprocess.call('rm -r {}'.format(tmpdir), shell=True)

    return earth

def loglikelihood(params, data, smooth=False, full=False, prior=True):
    # Prior range
    boxcarprior = params[0] < -8 or params[0] > 3 or params[1] < -3 or params[1] > 3
    if boxcarprior and priors:
        return -np.inf, np.empty(25)

    if full:
        fullparams = params.copy()
    else:
        fullparams = TRUE_MODEL.copy()
        fullparams[[1,4]] = params 
   

    sim, result = generative_model(*fullparams)
    sstopoloc = result.sstopo[:,sinds[0],sinds[1]]

    problocs = result.sstopo[-1,sinds[0], sinds[1]] - sstopoloc[tinds]

    prob = -0.5*np.sum(((problocs.T.flatten()- data)/SIG)**2)

    if smooth:
        obsers = (result.sstopo[-1,sinds[0], sinds[1]] -
                        sstopoloc[::2]).T.flatten()
    else:
        obsers = problocs.T.flatten()

    preds = predictions(sim, result)

    return prob, np.r_[obsers, preds]

def full_jacobian(params, data, sig):
    jac = giader.jacfridr(residuals, x=params, h=np.array([.01, 0.01, 1, 10,
                                0.01]), ndim=15,
                                fargs=(data,sig,False,True))
    return jac.T

def jacobian(params, data, sig):
    jac = giader.jacfridr(residuals, x=params, h=np.array([.01, 0.01]), 
                            ndim=15, fargs=(data,sig))
    return jac.T

def residuals(params, data, sig):
    return (loglikelihood(params, data, prior=False)[1][:15] - data)/sig

# Directional second derivative
def Avv(params,v, data, sig):
    h = 1e-4
    ret = (residuals(params + h*v, data, sig) + 
                residuals(params - h*v, data, sig) - 2*residuals(params, data, sig))/h/h
    return ret

def neg_loglike(x, *args):
    return -loglikelihood(x, *args)[0]

class pickleable_geodesic(geodesic):
    def results(self):
        res = {}
        res['ts'] = self.ts
        res['xs'] = self.xs
        res['vs'] = self.vs
        res['rs'] = self.rs
        res['vels'] = self.vels
        return res

       COOLPOINT1 = {'POS' : np.array([ 0.71428571,  9.142857  ]),
                    'TDAT': np.array([  1.37094450e+02,   1.76909730e+02,   
                                        6.78583025e+01,
                        4.36532927e+01,   1.51940899e+01,  -1.17675056e+01,
                          4.44809017e+01,   2.08024131e+01,   1.24532265e+01,
                                   3.91999968e+00,  -1.36939880e+02,
                                   -4.27034566e+01,
                                            1.73466719e-01,   5.44272543e-01,
                                            3.62926892e-01]),
                    'ERR': np.array([  4.43769216, -30.42118628,  -7.18123917, -10.93562241,
                -0.85016914,   2.17680239,   5.50808059,   0.8097613 ,
                         7.45813956,  -0.60783704,   2.05831261,  11.35377398,
                                  0.04953255,  -0.07640478,  -0.18069191]),
                    'SIG' = np.array([  2.74188900e+01,   3.53819460e+01,   1.35716605e+01,
                    8.73065854e+00,   3.03881799e+00,   2.35350112e+00,
                          8.89618035e+00,   4.16048263e+00,   2.49064530e+00,
                                   7.83999935e-01,   2.73879761e+01,
                                   8.54069132e+00,
                                            3.46933438e-02,   1.08854509e-01,
                                            7.25853785e-02])
                    'MIN': np.array([  0.84844117,  10.19599704]),
                    'FULL': np.array([np.log10(2.), np.log10(0.71428571), 70,
                                        1500, np.log(15-9.142857)])}

        COOLPOINT4 = {'TDAT': np.array([  3.04299790e+02,   2.98824122e+02,   1.12634787e+02,
                           2.16411098e+01,   9.23738144e-01,   2.99699735e+01,
                           1.64125310e+01,   1.34444550e-02,   1.89174560e+00,
                           5.54921798e-01,  -1.34515341e+02,  -7.80804640e+01,
                          -4.46250522e+00,  -5.17397459e-01,
                          -4.07247878e-02]),

        'ERR' : np.array([ -8.35774799,  10.55384355, -14.83290158,   8.08810353,
               -11.84851821,   3.05463451,  12.29802327,  -4.73890058,
                       -4.53192798,   3.9131337 ,   7.10964655,   0.31344921,
                                3.94560317,   5.33219108,   7.29739278]),

        'SIG': 10}

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sample a heuristic model manifold')
    parser.add_argument('typ', type=str, nargs='?', default=None, 
                    help='type of manifold exploration, see overall description',
                    choices=['grid', 'min', 'geomin', 'mcmc', 'uni',
                    'tighttime', 'jacgrid', 'geodesic'])
    parser.add_argument('fname', type=str, nargs='?', default=None)
    parser.add_argument('nsteps', type=int, nargs='?', default=50)
    parser.add_argument('--s', dest='smooth', default=False,
                            action='store_const', const=True)
    parser.add_argument('--params', nargs='+', default=['v1', 't0'])

    comargs = parser.parse_args()
    typ, fname, nsteps = comargs.typ, comargs.fname, comargs.nsteps
    smooth = comargs.smooth

    assert len(comargs.params)>=2, 'Must sample at least 2 parameters'
    p1name, p2name = comargs.params[:2]

    ilatc, ilonc, h = ice_spatial_cases['L2']
    tlatc, tlonc, bmax, b0 = topo_spatial_cases['B2']
    transect = map_path((ilonc, 90-ilatc), (tlonc, 90-tlatc),
                    m=giapy.map_tools.Basemap(), lonlat=True, n=5)

    grid = giapy.map_tools.GridObject(giapy.map_tools.Basemap(), shape=(NLAT,NLON))
    sinds = np.array([grid.closest_grid_ind(transect.xs[i], transect.ys[i]) for i in [0,1,4]]).T
    tinds = [0,10,40,50,65]
    pinds = np.array([grid.closest_grid_ind(lon, lat) for lon, lat in
                                            zip([70], [55])]).T
    topoB3 = gen_sstopo('B3')

    # Parameter Ranges
    paramranges = {'t0': (0, 14, 4),
                   'logt0': (-3, 3, 4),
                   'v1': (-3, 3, 1),
                   'h': (500, 2500, 3),
                   'l': (30, 250, 2)}
   
    # Check the output file and create if not present - OVERWRITES!
    if fname is not None:
        try:
            with open(fname, 'r') as f:
                pass
        except:
           with open(fname, 'w') as f:
                pass

    # UNIFORM SAMPLING PARAMETER SPACE
    if typ == 'uni':
        params = (np.random.rand(5, nsteps) * 
                        np.array([2-(-1), logv1ran, 250,1500, t0ran])[:,None] + 
                        np.array([-1, logv1min, 30, 500, t0min])[:,None]) 
        
        #params[0] = np.log10(2)
        #params[2] = 70
        #params[3] = 1500
        #params[4] = 0.2*np.round(params[4]/0.2)
        for param in params.T:
            obsers = obsandpreds(*generative_model(*param))
            output = np.r_[param, obsers]

            with open(fname, 'a') as f:
                f.write('{}\n'.format('\t'.join([str(n) for n in output])))

    # OUTPUT FROM MODEL ON GRID
    if typ == 'grid':
        p1min, p1max, p1ind = paramranges[p1name]
        p2min, p2max, p2ind = paramranges[p2name] 

        # Reverse the order to make read-in logical
        for p2 in np.linspace(p2min, p2max, nsteps):
            for p1 in np.linspace(p1min, p1max, nsteps):
                full_params = TRUE_MODEL.copy()
                full_params[[p1ind,p2ind]] = p1, p2
  
                obsers = obsandpreds(*generative_model(*full_params))
                output = np.r_[p1, p2, obsers]

                with open(fname, 'a') as f:
                    f.write('{}\n'.format('\t'.join([str(n) for n in output])))

    # MCMC THE POSTERIOR
    elif typ == 'mcmc':
        nwalkers, ndim = 10, 2

        DATA = TDAT + ERR

        sampler = emcee.EnsembleSampler(nwalkers, ndim, loglikelihood, 
                                        args=(DATA,smooth), threads=4)

        #pos = np.repeat(TRUE_MODEL[[1,4]][None,:], nwalkers, axis=0)
        #pos = np.repeat(POS[None, :],
        #                nwalkers, axis=0)
        #scat = np.random.randn(nwalkers, ndim)
        #scat[:,0] *= 1
        #scat[:,1] *= 2.5
        #pos += scat

        pos = np.loadtxt(fname)[-nwalkers:,[2,3]]
        #params = np.random.multivariate_normal(mean=np.array([-0.89995028, 5.21344428]), 
        #                              cov=np.array([[0.092039901222281395,  0.],
        #                                            [0., 0.0625]]), 
        #                            size=nwalkers)

        sampleOut(sampler, pos, None, None, 
                        fname, nsteps, verbose=True)
    
    # SAMPLE INDEPENDENT PARAMETERS, TIGHT TIME
    elif typ == 'tighttime':
        params = np.random.multivariate_normal(mean=np.array([-0.89995028, 5.21344428]), 
                                      cov=np.array([[0.092039901222281395,  0.],
                                                    [0., 0.0625]]), 
                                    size=nsteps)
        
        fullparams = np.repeat(TRUE_MODEL[None,:], nsteps, axis=0)
      
        fullparams[:,[1,4]] = params

        for param in fullparams:
            #sim, result = generative_model(*param)
            #obsers = observations(sim, result)
            obsers = observations(*generative_model(*param), smooth=smooth).T.flatten()
            output = np.r_[param, obsers]

            with open(fname, 'a') as f:
                f.write('{}\n'.format('\t'.join([str(n) for n in output])))

    # MINIMIZE
    elif typ in ['min', 'geomin']:

        DATA = TDAT + ERR

        geo = False if typ == 'min' else True
        pos = TRUE_MODEL[[1,4]]
        pos = np.array([1, 2])

        pos = TRUE_MODEL.copy()
        pos[[1,4]] = np.array([ 0.71428571,  np.log(15-9.142857)  ])


        posminres = geolm_minimize(residuals, pos, jac=jacobian,
        fargs=(DATA,SIG),
                        jargs=(DATA,SIG), keep_steps=True, geo=geo)
        pickle.dump(posminres, open(fname, 'w'))

    elif typ == 'geodesic':
        coolpoint = COOLPOINT1

        DATA = coolpoint['TDAT']+ coolpoint['ERR']


        def r(x):
            return residuals(x, DATA, SIG)

        def j(x):
            return jacobian(x, DATA, SIG)

        def A(x,v):
            return Avv(x, v, DATA, SIG)

        x = coolpoint['POS']
        x = coolpoint['MIN']

        v = np.array([1., 0])
        v /= np.linalg.norm(v)

        # Callback function used to monitor the geodesic after each step
        def callback(geo):
            with open(geo.fname, 'w') as f:
                pickle.dump(geo.results(), f)
            # Integrate until the norm of the velocity has grown by a factor of 10
            # and print out some diagnotistic along the way
            print("{}: Iteration: {:d}, tau: {:f}, |v| = {:f}".format(
                    datetime.datetime.now(), len(geo.vs), geo.ts[-1], np.linalg.norm(geo.vs[-1])))
            return np.linalg.norm(geo.vs[-1]) < 10.0

        geo = pickleable_geodesic(r, j, A, 15, 2, x, v, atol = 1e-2, rtol = 1e-2,
                        callback = callback)
        geo.fname = fname
                        
        with open(fname, 'w') as f:
            pickle.dump(geo.results(), f)
        try:
            geo.integrate(25.0)
        except:
            print('KEYBOARD INTERRUPT')

#        del geo.r, geo.j, geo.Avv
        with open(fname, 'w') as f:
            pickle.dump(geo.results(), f)
        

    # COMPUTE THE JACOBIAN ON A GRID
    elif typ == 'jacgrid':

        for t0 in np.linspace(t0min, t0max, nsteps):
            for v1 in np.linspace(-4, 1.5, nsteps):
                full_params = TRUE_MODEL.copy()
                full_params[[1,4]] = v1, t0
                
                res = residuals(np.array([v1,t0]), DATA)
                jac = jacobian(np.array([v1,t0]), DATA)
                output = np.r_[v1, t0, res, jac.flatten()]
                with open(fname, 'a') as f:
                    f.write('{}\n'.format('\t'.join([str(n) for n in output])))
        
    # SAMPLE A GAUSSIAN PARAMETER DISTRIBUTION
    elif typ == 'gauss':
        params = np.random.multivariate_normal(mean=np.array([-0.89995028, 5.21344428]), 
                                      cov=np.array([[ 0.05434718,  0.16358428],
                                                    [0.16358428, 0.64113101]]), 
                                    size=nsteps)
        
        fullparams = np.repeat(TRUE_MODEL[None,:], nsteps, axis=0)
        
       
      
        fullparams[:,[1,4]] = params

        for param in fullparams:
            #sim, result = generative_model(*param)
            #obsers = observations(sim, result)
            obsers = observations(*generative_model(*param), smooth=smooth).T.flatten()
            output = np.r_[param, obsers]

            with open(fname, 'a') as f:
                f.write('{}\n'.format('\t'.join([str(n) for n in output])))
