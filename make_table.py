# Testing stub
import argparse
import pickle
import os

from precomputation import approxMergerTime

parser = argparse.ArgumentParser(description="Precompute merger redshift table of cosmologically coupled compact remnant binaries",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("fname",
                    help="File to output or read in")

parser.add_argument("--psize",
                    help="How many children to spawn during computation",
                    type=int,
                    default=1)

parser.add_argument("--validate",
                    help="Number of random validation pulls to examine",
                    type=int,
                    default=1000)

parser.add_argument("--order",
                    help="Order at which to use map_coordinates()",
                    type=int,
                    default=1)

parser.add_argument("--k",
                    help="The coupling strength",
                    type=float,
                    default=3)

args = parser.parse_args()

if os.path.exists(args.fname):
    print("Found file, opening")
    derp = pickle.load(open(args.fname, 'rb'))

    if not 'maxorder' in dir(derp):
        derp.maxorder = min(5, min(derp.N))

    if not 'individual_grids' in dir(derp):
        derp.individual_grids = { col : (np.logspace if col == 'R' else np.linspace)(x[0], x[1], N) for x,col,N in zip(derp.ranges,
                                                                                                                       derp.column_names,
                                                                                                                       derp.N)}
        
    # Set the requested order
    derp.set_order(args.order)
    
else:
    print("Producing table for coupling strength", args.k)
    
    derp = approxMergerTime({ 'k' : args.k,
                              'a_min' : 1/(1+25),
                              'a_max' : 0.95,
                              'M_min' : 2.7,
                              'M_max' : 15,
                              'R_min' : 1e-2,
                              'R_max' : 1e3,
                              'e_min' : 0,
                              'e_max' : 1. - 1e-2,
                              'N_a' : 40,
                              'N_M' : 10,
                              'N_R' : 10,
                              'N_e' : 20,
                              'order' : args.order,
                              'usefloats' : False})

    # Eventual size of table (typical max L2 cache is 8MB)
    # Not sure how much speed will depend on the size of the
    # table itself though, as map_coordinates() needs to play
    # its own games....
    print("Table will be: ", derp.getTableByteSize()/1e6, "MB")

    # # Sanity check scalings
    # print((derp.ranges[:,1] - derp.ranges[:,0])*derp.scalings)
    
    # Generate the table
    derp.compute(args.psize)

    # Write out the pickle
    pickle.dump(derp, open(args.fname, "wb"))
