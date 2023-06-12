
# Thank you @Joe Kington (https://stackoverflow.com/a/6238859/3124688) for
# pointing out the utility of map_coordinates!

from scipy.ndimage import map_coordinates, spline_filter
import numpy as np
import pandas as pd
import c3o_binary_better as c3o
import multiprocessing
import sys
from scipy.stats import uniform, loguniform
import itertools
from tqdm import tqdm

# Precompute a bunch of values for use with ndimage.map_coordinates
# Assumes k = 3 because this is the measured value in SMBH

class approxMergerTime(object):

    def __init__(self, params):

        # Establish the column names
        self.column_names = ['a_DCO', 'M', 'M2', 'R', 'e']

        # Establish it
        self.params = params
        
        # Convert all parameters, which can be converted to numbers, to numbers
        for key,val in params.items():
            try:
                self.params[key] = float(val)
            except TypeError:
                self.params[key] = val

        # Ranges
        self.ranges = np.array([[self.params['a_min'], self.params['a_max']],  # Position 0
                                np.log10([self.params['M_min'], self.params['M_max']]),  # Position 1 (primary mass)
                                np.log10([self.params['M_min'], self.params['M_max']]),  # Position 2 (secondard mass)
                                #[self.params['q_min'], self.params['q_max']],  # Position 2
                                np.log10([self.params['R_min'], self.params['R_max']]),  # Position 3
                                [self.params['e_min'], self.params['e_max']]]) # Position 4

        # Column 1 - Column 2
        widths = self.ranges[:,1] - self.ranges[:,0]

        # Gridpoints
        self.N = np.array([self.params['N_a'],
                           self.params['N_M'],
                           self.params['N_M'],
                           self.params['N_R'],
                           self.params['N_e']], dtype=int)

        # Grids
        self.individual_grids = { col : (np.logspace if col in ['R', 'M', 'M2'] else np.linspace)(x[0], x[1], N) for x,col,N in zip(self.ranges,
                                                                                                                                    self.column_names,
                                                                                                                                    self.N)}
        
        # These map the requested values into gridspace
        # after subtracting off the low end
        self.scalings = (self.N - 1)/widths 

        # Did the user specify a non-standard coupling?
        if 'k' in self.params:
            self.k = self.params['k']
        else:
            # By default, assume theoretically and observationally
            # most well-motivated value
            self.k = 3
            
        # How aggressive do we want to approximate?
        if 'order' in self.params:
            self.order = int(self.params['order'])
        else:
            # Default linear
            self.order = 1

        # Adjust the maximum order based on N
        self.maxorder = min(5, min(self.N))
        
        # Things go fast af if you can get the entire table
        # in RAM.
        if 'usefloats' in self.params and self.params['usefloats']:
            self.dtype = np.float32
        else:
            self.dtype = np.float64
            
        # The precomputation grid (in straight up numpy, and then in pandas)
        self.data = None
        self.cute_results = None
        
        # The splined data for fast interpolation
        self.splined_data = None

        # The mask used for avoiding NaN's in filter data
        self.data_mask = None

    # Print out the expected table size
    def getTableByteSize(self):
        # KC 6/11/23
        # XXX For some reason, this size estimate is completely wrong...
        return np.prod(self.N)*(4 if isinstance(self.dtype, np.float32) else 8)

    # Population the interpolation table (the slow step)
    def compute(self, Nkids=1):
        
        # If there is a table, don't blindly trample it
        if not self.data is None:
            raise Exception("Looks like a table has already been computed")

        # Make sure we're being called reasonably
        #if not __name__ == '__main__':
        #    raise Exception("Gotta spawn from a __main__ module")

        # Make a dataframe storing the cartesian product
        cartesian = pd.DataFrame(np.array([x for x in itertools.product(*self.individual_grids.values())]),
                                 columns=self.column_names)

        # Give some output
        print("About to compute at %d grid positions..." % len(cartesian))
        
        ## OOO This can be 2x optimized here, but it really causes
        # complications to reconstruct the final map.        
        # # Take the subset where q <= 1
        
        # Integrator only knows how to handle q...
        cartesian['q'] = cartesian['M2'] / cartesian['M']

        # So drop illegal q
        # cartesian = cartesian[cartesian['q'] <= 1]
        cartesian.drop(index=cartesian[cartesian['q'] > 1].index, inplace=True)
        
        # Explicitly set the coupling strength
        cartesian['k'] = self.k

        # Spawn offspring
        with multiprocessing.Pool(Nkids,
                                  initializer=c3o.setup_locking,
                                  initargs=(multiprocessing.Lock(),)) as p:
            results = pd.concat(p.map(c3o.c3o_binary_a_worker, np.array_split(cartesian.copy(), Nkids)))

        # Save in times, because converting is causing me grief
        cartesian['a_f'] = results['a_f']

        # Drop k, because we already know it
        del cartesian['k']

        # Save the pandas frame for easy viewing/debugging of the grid
        self.cute_results = cartesian.copy()

        # Do this the dumb way, so I don't screw it up
        cartesian_swap = cartesian.copy()
        tmp = cartesian_swap['M']
        cartesian_swap['M'] = cartesian_swap['M2']
        cartesian_swap['M2'] = tmp
        chonker = pd.concat([cartesian, cartesian_swap], ignore_index=True)
        chonker.drop_duplicates(inplace=True, ignore_index=True)
        
        # Convert this into something that map coordinates can handle
        # This should leave 'z_f' in final position (its not in the column_names)
        self.data = (chonker.sort_values(self.column_names))['a_f'].to_numpy().reshape(*self.N)

        # Do the spline stuff
        self.setup_splines()

    # Way to adjust resolution
    def set_order(self, order):
        if isinstance(order,int) and (order >= 0 and order <= 5):
            self.order = order
        else:
            raise ValueError("order goes to ndimage.map_coordinates(), so it needs to be in [0,5]")

        # Re-establish splining
        self.setup_splines()

    # Do masking and filtering in preparation for splining
    def setup_splines(self):

        if self.data is None:
            raise Exception("No underlying data defined yet")

        # Now set up the splined data at the requested order
        if self.order > 1:
            self.splined_data = spline_filter(self.data,
                                              order=self.order)
        else:
            # Just use linear
            self.splined_data = self.data
        
    # Return the estimated merger time
    def merger_a(self, a, M, M2, R, e):

        # Make sure we have splined it out
        if self.splined_data is None:
            raise Exception("No splined data defined")

        try:
            # 'nearest' returns the edges if you go off the grid, Roy
            # It also wants columns, hence the transpose

            # 5 rows and 10 columns, so transpose to get 10 rows and 5 columns
            queries = np.asarray([a, np.log10(M), np.log10(M2), np.log10(R), e]).T
            xlations = self.ranges[:,0]

            coords = np.atleast_2d(self.scalings*(queries - xlations)).T

            # Perform the lookup
            return map_coordinates(self.splined_data,
                                   coords,
                                   order=self.order,
                                   mode='nearest',
                                   prefilter=False)
                              
        except ValueError as err:
            print("Interpolation table is probably not computed yet?")
            print(err)
            raise(err)
            return np.nan
        
    # Run N random values within the ranges
    # and do actual integrations
    # and plot some precision statistics
    def validate(self, N, Nkids=1):

        #if not __name__ == '__main__':
        #    raise Exception("Gotta spawn from a __main__ module")
            
        # Make test distributions
        coords = pd.DataFrame(np.array([(loguniform(10**x[0], 10**x[1]) if col in ['R', 'M', 'M2'] else uniform(loc=x[0], scale=(x[1] - x[0]))).rvs(size=N) for x,col in zip(self.ranges, self.column_names)]).T,
                              columns=self.column_names)

        # Add in the coupling
        coords['k'] = self.k

        # Integrator needs q
        coords['q'] = coords['M2']/coords['M']
        
        # Using pandas here is probably unnecessarily slow
        # If it sucketh exceeding, just revert back to careful numpy
        
        # # Make a set of runs to perform
        # import matplotlib.pyplot as plt
        # print(vals)

        # # Distribution verification stuff 
        # for i,col in enumerate(self.column_names):
        #     if col == 'R':
        #         plt.xscale('log')
        #         bins=np.logspace(self.ranges[i][0], self.ranges[i][1], 30)
        #         print(bins)
        #     else:
        #         bins=None
                
        #     plt.hist(vals[col], bins=bins, edgecolor='grey')
        #     plt.show()

        # Spawn offspring
        # Notice that we use a copy because the underlying integrator mutates it
        with multiprocessing.Pool(Nkids,
                                  initializer=c3o.setup_locking,
                                  initargs=(multiprocessing.Lock(),)) as p:
            results = pd.concat(p.map(c3o.c3o_binary_a_worker, np.array_split(coords.copy(), Nkids)))

        # Rename it
        results['a_f_direct'] = results['a_f']

        # Remove k and q
        del coords['k']
        del coords['q']
        
        # Make data for all orders
        for order in range(1, self.maxorder+1):

            # Make new spline filters
            self.set_order(order)
            
            # Now iterate over the coordinates (displaying progress)
            mergers = [self.merger_a(*row) for idx,row in tqdm(coords.iterrows(), total=len(coords))]
            
            results['a_f_table_%d' % order] = pd.DataFrame(mergers,
                                                           columns=['a_f'],
                                                           index=coords.index)

        # Concatenate the columns, order should be correct ;)
        # No idea how to do this quickly, Im sure its easy
        for column in coords.columns:
            results[column] = coords[column]
            
        return results
            
# # Testing stub
# import argparse
# import pickle
# import os

# parser = argparse.ArgumentParser(description="Precompute merger redshift table of cosmologically coupled compact remnant binaries",
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser.add_argument("fname",
#                     help="File to output or read in")

# parser.add_argument("--psize",
#                     help="How many children to spawn during computation",
#                     type=int,
#                     default=1)

# parser.add_argument("--validate",
#                     help="Number of random validation pulls to examine",
#                     type=int,
#                     default=1000)

# parser.add_argument("--order",
#                     help="Order at which to use map_coordinates()",
#                     type=int,
#                     default=1)

# args = parser.parse_args()

# if os.path.exists(args.fname):
#     print("Found file, opening")
#     derp = pickle.load(open(args.fname, 'rb'))

#     if not 'maxorder' in dir(derp):
#         derp.maxorder = min(5, min(derp.N))

#     if not 'individual_grids' in dir(derp):
#         derp.individual_grids = { col : (np.logspace if col == 'R' else np.linspace)(x[0], x[1], N) for x,col,N in zip(derp.ranges,
#                                                                                                                        derp.column_names,
#                                                                                                                        derp.N)}
        
#     # Set the requested order
#     derp.set_order(args.order)
    
# else:    
#     derp = approxMergerTime({ 'a_min' : 1/(1+25),
#                               'a_max' : 0.95,
#                               'M_min' : 2.7,
#                               'M_max' : 15,
#                               'R_min' : 1e-2,
#                               'R_max' : 1e3,
#                               'e_min' : 0,
#                               'e_max' : 1. - 1e-2,
#                               'N_a' : 40,
#                               'N_M' : 10,
#                               'N_R' : 10,
#                               'N_e' : 20,
#                               'order' : args.order,
#                               'usefloats' : False})

#     # Eventual size of table (typical max L2 cache is 8MB)
#     # Not sure how much speed will depend on the size of the
#     # table itself though, as map_coordinates() needs to play
#     # its own games....
#     print("Table will be: ", derp.getTableByteSize()/1e6, "MB")

#     # # Sanity check scalings
#     # print((derp.ranges[:,1] - derp.ranges[:,0])*derp.scalings)
    
#     # Generate the table
#     derp.compute(args.psize)

#     # Write out the pickle
#     pickle.dump(derp, open(args.fname, "wb"))

# # Spot check the results
# print(derp.cute_results)

# # Print some stuf
# print("Table supports maximum order of: ", derp.maxorder)
# print("Table is roughly: ", derp.getTableByteSize()/1e6, "MB")

# # Now plot some graphs
# import matplotlib.pyplot as plt

# # Look at some slices through the data
# #fig, axs = plt.subplots(ncols=derp.N[derp.column_names.index('e')],
# #                        nrows=derp.N[derp.column_names.index('q')])

# # # Turn this into a movie that can be played looking for shitty behaviour
# # # Make huge graphs, one for each z
# # for k,z_DCO in enumerate(derp.individual_grids['z_DCO']):
# #     for i,M in enumerate(derp.individual_grids['M']):
# #         for j,M2 in enumerate(derp.individual_grids['M'][:i+1]):
# #             q = M2/M
# #             plt.xlabel("log10(R)")
# #             plt.ylabel("e")
# #             plt.title("a_DCO = %f, M = %f, q = %f" % (1./(1+z_DCO),M,q))
# #             plt.imshow(derp.data[k,i,j,:,:].T,
# #                        extent=(derp.ranges[3][0], derp.ranges[3][-1], derp.ranges[4][0], derp.ranges[4][1]),
# #                        aspect='equal',
# #                        origin='lower',
# #                        vmin=1/(1+derp.ranges[0][-1]),
# #                        vmax=c3o.max_lookforward)
# #             plt.colorbar(label='a merger')
# #             plt.show()
            
# # # See some statistics
# results = derp.validate(args.validate, args.psize)

# print(results)



# bins = np.linspace(-1,0.5,100)
# fig = plt.figure()
# gs = fig.add_gridspec(derp.maxorder, hspace=0)
# axs = gs.subplots(sharex=True) #, sharey=True)

# for order in range(1, derp.maxorder+1):

#     ax = axs[order-1]
    
#     deltas = (results['a_f_direct'] - results['a_f_table_%d' % order])/results['a_f_direct']
#     bad = np.isfinite(deltas)
#     ax.hist(deltas[bad], label='order %d' % order, bins=bins)
#     ax.legend()
#     ax.set_yscale('log')
    
# #plt.xlabel("Delta merger time, direct - table, in 1/H0's")
# #plt.ylabel("Counts")
# #plt.yscale('log')
# plt.show()

                        
