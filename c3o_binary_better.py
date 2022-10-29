from scipy.integrate import solve_ivp, cumtrapz, quad
import argparse
import numpy as np
import sys
from scipy.interpolate import interp1d
import pandas as pd
import multiprocessing
import fcntl
from tqdm import tqdm

# Assuming Planck 2018 flat cosmology
# and cosmological units.
# OmegaM = 0.315
# # Planck 2015?
#OmegaM = 0.308

# Astropy Planck2018??
OmegaM = 0.30966

OmegaL = 1 - OmegaM

# CODATA G and c in mks, converted to MSol, AU, and H0^{-1}
# 67.4 km/(Mpc s) from Planck 2018
# H0_km_per_Mpc_s = 67.4

# Astropy Planck 2018??
H0_km_per_Mpc_s = 67.66

# # Planck 2015?
# H0_km_per_Mpc_s = 67.8

# From the void
km_per_Mpc = 3.0856775814913673e19
secs_per_year = 31557600

# Compute me
secs_per_H0inv = 1. / (H0_km_per_Mpc_s / km_per_Mpc)
RecipH0ToMyr = secs_per_H0inv / secs_per_year / 1e6
MyrToRecipH0 = 1./RecipH0ToMyr

G_MKS = 6.67408e-11
c_MKS = 299792458 # m/s
G_over_c2_MKS =  G_MKS/c_MKS**2  # m/kg
AU_per_meter = 1/1.496e11
kg_per_msol = 1.989e30 
#secs_per_H0inv = 4.5786e17

## Things in "astronomical units" (AU, Msol, and reciprocal Hubble)
G_over_c2_ASTRON = G_over_c2_MKS * AU_per_meter * kg_per_msol

c_ASTRON = c_MKS * AU_per_meter * secs_per_H0inv
G_ASTRON = G_over_c2_ASTRON * c_ASTRON**2

# So I don't compute them inline every time
G3_over_c6_ASTRON = (G_over_c2_ASTRON)**3
G3_over_c5_ASTRON = G3_over_c6_ASTRON * c_ASTRON
sqrt_G7_over_c12_ASTRON = (G3_over_c5_ASTRON**2 * G_over_c2_ASTRON)**0.5

# Peters buddy....
sqrt_G7_over_c10_ASTRON = (G3_over_c5_ASTRON**2 * G_ASTRON)**0.5
                           
# Solar radii to astronomical units
RSol_to_AU = 0.00465047

# Conversions to Astro units
#MKStoAstro_L = 1.03e-35
#MKStoAstro_R = 1/(1.496e11)

# In AU (this should be a cutoff at ~0.2 Rsol)
periastron_cutoff = 1e-4

# Madau & Dickinson 2014 stellar return fraction
madau_R = 0.27

# From Madau & Dickinson 2014.
def madau_psi(z):
    return 0.015*(1+z)**2.7/(1 + ( (1+z)/2.9 )**5.6)
    #return (1+z)**2.7

# We assumed the time unit here was H_0^{-1}, so this should be right
def H(a):
    return (OmegaM/a**3 + OmegaL)**0.5

def lookback_integrand(a):
    return 1./(a*H(a))

# Use quad() [ like astropy, hehe ] because cumtrapz() has been cancelled
# (its cumulative error was 10%, or 1Gyr anyway, so it was dumb to use anyway)
max_lookforward = 2.

def direct_lookback_interpolators(a_i, a_f=max_lookforward, N=10000):
    dom_past = np.linspace(a_i, 1, N//2)
    cumt_past = np.array([quad(lookback_integrand, 1, a)[0] for a in dom_past])

    # Drop the 1, because its already in the "past"
    dom_future = np.linspace(1, a_f, N//2)[1:]
    cumt_future = np.array([quad(lookback_integrand, 1, a)[0] for a in dom_future])

    dom = np.append(dom_past, dom_future)
    cumt = np.append(cumt_past, cumt_future)

    print(dom)
    print(cumt)
    
    # Definitely complain if we end up outside the integration bounds
    return (interp1d(dom, cumt), interp1d(cumt, dom))

# Get some interpolators for lookback time
a2t, t2a = direct_lookback_interpolators(1./(1+29))

# Locking for logging stuff
# This is a global that will store the lock.
# Though we are passed one lock to rule them all at multiprocessing setup...
log_lock = None
def setup_locking(lock):
    global log_lock
    log_lock = lock

class EarlyTermException(Exception):

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        
# We have the default time and state, and then we expect additional arguments
def c3o_binary_a(a,
                 state,
                 M_i3,
                 q_times_one_plus_q,
                 one_plus_q_over_q2,
                 k,
                 a_i,
                 q,
                 icvec):

    # q is the mass ratio
    # k is the coupling strength (i.e. [0,3])

    # Unpack the current integrator state
    R, e = state

    # If R or e is negative, the evaluation equtions will vomit
    # So just (force) bail at this timestep
    if R <= 0 or e < 0 or (e - 1.) > 0:
        tmp = EarlyTermException()
        tmp.a = a
        tmp.state = state
        tmp.icvec = icvec
        raise(tmp)
    
    # Precompute shifted mass
    shifted_mass_cubed = M_i3 * (a/a_i)**(3*k)

    # Precompute e**2
    e2 = e**2

    # Precompute 1-e**2
    one_minus_e2 = 1 - e2

    # Recip Hubble
    recipHa = lookback_integrand(a)
    
    # Except out when we try to compute something bad
    with np.errstate(all='raise'):
        try:
            # Semi-major axis
            dRda = (-64./5 * G3_over_c5_ASTRON * shifted_mass_cubed * q_times_one_plus_q * (1 + 73./24*e2 + 37./96*e2**2) / (R**3 * one_minus_e2**3.5)*recipHa - 3*k*R/a)

            # Eccentricity
            deda = -304./15 * e * (1 + 121./304 * e2) * G3_over_c5_ASTRON * shifted_mass_cubed * q_times_one_plus_q / (one_minus_e2**2.5 * R**4)*recipHa
            
            # # This was missing a shift factor and may have been off by a power of c^2!
            #dLda = (-32./5 *
            #        sqrt_G7_over_c10_ASTRON *
            #        (shifted_mass_cubed)**1.5 * q**2 *
            #        (1+q)**0.5 *
            #        (1 + 7./8 * e2) / (R**3.5 * one_minus_e2**2)) * recipHa

            # Return the values in a numpy array
            return np.array((dRda, deda))

        except Exception as derp:
            # Log that something stupid happened
            log_insanity(derp,
                         icvec,
                         pd.DataFrame(data={'z_DCO' : [1./a - 1],
                                            'M' : [shifted_mass_cubed**(1./3)],
                                            'q' : [q],
                                            'R' : [R],
                                            'e' : [e],
                                            'k' : [k]}),
                         flavor="NaN happened during computation of the evolution equations...")
                         
            # All values are undefined if something stupid happened
            return np.array((np.nan, np.nan))

def log_insanity(theexcept, icvec, fcvec, flavor=None):
                     
    # Do locking, since this is multiprocess
    log_lock.acquire()

    try:
        # Dump it
        import traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, file=integrator_insanity)

        print("The exception: ", theexcept,
              "Flavor text: ", flavor,
              "\nInitial condition at failure\n",
              icvec,
              "\nFinal condition at failure\n",
              fcvec,
              "\n",
              file=integrator_insanity)

    except Exception as e:
        print("While trying to log shit, another exception was raised", e)#, file=integrator_insanity)
        
    finally:
        # Release the posix lock
        log_lock.release()
        
    return
          
# Terminate when periastron is below cutoff
def merger(a,
           state,
           M_i3,
           q_times_one_plus_q,
           one_plus_q_over_q2,
           k,
           a_i,
           q,
           icvec):

    # state[0] is R, state[1] is e
    periastron = state[0]*(1-state[1])
    delta = periastron - periastron_cutoff
    return delta

# Terminate if we become hyperbolic (possibly integration error)
def ejection(a,
             state,
             M_i3,
             q_times_one_plus_q,
             one_plus_q_over_q2,
             k,
             a_i,
             q,
             icvec):

    return 1. - state[1]

# Terminate in all these situations
merger.terminal = True
ejection.terminal = True

#
# Wrapper on solve_ivp() that computes the initial conditions.
# Go ahead and if, and hope branch prediction knows ahead of time whatt to do
#
def characterize_inspiral(model, frame, method='LSODA', full=False):

    # These are actually one dimensional, because we
    # explicitly did not vectorize, so that we could get some sense of
    # progress during the calculation...
    
    # Pull things out of the frame (though the lookups
    # will probably get cached anyway)
    #z = frame['z_DCO']
    #a = 1./(1 + z)
    a = frame['a_DCO']
    M = frame['M']
    q = frame['q']
    k = frame['k']
    e = frame['e']
    R = frame['R']

    # Short circuit integration if we are already below integration cutoff
    if R*(1-e) <= periastron_cutoff:
        if not full:
            # Make sure we get a float and not some stupid 0dim array
            return a
        else:
            raise Exception("Not today, Mr. Bond")
        
            # # Mimimc an orbit object?
            # tmp = object()
            # tmp.t = np.array([a2t(a)])
            # tmp.nfev = 0
            # tmp.t_events = []

            # # This is an array with L,R,e
            # # (I don't care to compute the angular momentum)
            # tmp.y = np.array([[0],[R],[e]])
            # return tmp

    # Fancy trouble detection
    events = [merger, ejection]
    
    while True:
        with np.errstate(all='raise'):
            # This returns a sol object
            try:
                orbit = solve_ivp(model,
                                  # The differential equation for the orbit (it should be vectorized)
                                  [a, max_lookforward],
                                  # --> The integration domain (solve_ivp() chooses a grid)
                                  np.array([R,
                                            e]),
                                  # --> Initial conditions
                                  events=events,
                                  # --> Integration termination conditions
                                  args=[M**3,
                                        q*(1.+q),
                                        (1.+q)/q**2,
                                        k,
                                        a,
                                        q,
                                        frame],
                                  # --> Precomputed arguments to save time, and the initial condition so we can troubleshoot
                                  #dense_output=True,
                                  method=method,
                                  # --> Integration method
                                  max_step=0.01)  # --> Make sure the integrator doesn't jump into batshit
               
            except (FloatingPointError, ValueError) as uhoh:
                log_insanity(uhoh,
                             frame,
                             frame,
                             flavor="errstate got hit outside of solve_ivp().... attempting without event detection")

                # Die if it failed
                if len(events) == 0:
                    log_insanity(uhoh,
                                 frame,
                                 frame,
                                 flavor="errstate failed again outside solve_ivp().  dying.")
                    exit(4)
                
                # Kill fancy event detection one by one
                events.pop()
                continue

            except EarlyTermException as lol:
                # Return the timestamp of fuckage
                if not full:
                    return lol.a
                else:
                    raise Exception("Full reporting not supported for early termination")
            
        if not full:
            # Merged?
            if len(orbit.t_events[0]):
                return orbit.t_events[0][0]

            # "Ejected" gracefully?
            if len(orbit.t_events[1]):
                return orbit.t_events[1][0]

            # Terminated but missed the merger?
            return orbit.t[-1]
        else:
            # Otherwise, return the entire object provided by ivp
            return orbit

# In this approach, each worker will receive a chunk
# of the full data frames
integrator_insanity = open("integrator_insanity", "wt", buffering=1)

# Set the default integrator
integration_method = 'BDF'

def c3o_binary_a_worker(frames):

    # For return and reporting
    response = []
    total = len(frames)
    n = 0
    
    # I guess I should be vectorizing the actual integrations here
    # but then I can't get sweet progress reports...

    # 1) Vectorize (no, want output!)
    # 2) List comp (no, want output!!)
    # 3) Loop (yeah!!!)

    # Cool C-like behaviour with walrus operator!
    for index,frame in (pbar := tqdm(frames.iterrows(), total=len(frames))):

        # Get a merger scale factor
        response.append(characterize_inspiral(c3o_binary_a, frame, method=integration_method))

        ## Give some output every 255 systems
        #if not (n & 0xff):
        pbar.set_description("PID %d: a_DCO = %.2e, M = %.2e, q = %.2e, R = %.2e, e = %.2e" % (multiprocessing.current_process().pid, frame['a_DCO'], frame['M'], frame['q'], frame['R'], frame['e']))

        # Keep track
        n += 1
            
    frames['a_f'] = response

    # Return only the new column (save memory), but with the indices intact!
    return frames[['a_f']]
    
# Make a binary worker for multiprocessing
# We're going to make this more memory efficient now
# The worker will just compute the 
