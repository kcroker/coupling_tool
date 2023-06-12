# Testing stub
import argparse
import pickle
import os

from precomputation import approxMergerTime

parser = argparse.ArgumentParser(description="Precompute merger redshift table of cosmologically coupled compact remnant binaries",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("fname",
                    help="File to output or read in")

parser.add_argument("--R",
                    help="Semi-major axis (AU) grid range, will be uniform in log space.  Third number is samples.",
                    default="1e-2:1e3:10")

parser.add_argument("--M",
                    help="Mass (Msol) grid range, will be uniform in log space. Third number is samples",
                    default="1e-1:20:10")

parser.add_argument("--e",
                    help="Eccentricity grid range, will be uniform in log space.  Third number is samples",
                    default="0:0.99:10")

parser.add_argument("--z",
                    help="Redshift grid range, will be converted to linear grid in scale factor.  Third number is samples",
                    default="26:1e-2:10")

parser.add_argument("--k",
                    help="The coupling strength",
                    type=float,
                    default=3)

parser.add_argument("--validate",
                    help="Number of random validation pulls to examine",
                    type=int,
                    default=1000)

parser.add_argument("--bad",
                    help="Display fractional error ranges in logspace outside of 10%",
                    action='store_true')

parser.add_argument("--psize",
                    help="How many children to spawn during computation",
                    type=int,
                    default=1)

parser.add_argument("--order",
                    help="Order at which to use map_coordinates()",
                    type=int,
                    default=1)

def parseParams(args):
    derp = {}
    for field in ['R', 'z', 'e', 'M']:
        dakine = [float(x) if n < 2 else int(x) for n,x in enumerate(eval("args.%s" % field).split(':'))]

        # Switch to scale factor if necessary
        if field == 'z':
            for i in range(2):
                dakine[i] = 1./(1 + dakine[i])
            field = 'a'
            
        # Add the dict entries
        derp['%s_min' % field] = dakine[0]
        derp['%s_max' % field] = dakine[1]
        derp['N_%s' % field] = dakine[2]


    # Add remaining parameterse
    derp['order'] = args.order
    derp['useFloats'] = False
    derp['k'] = args.k
    
    return derp
    
args = parser.parse_args()

if os.path.exists(args.fname):
    print("Found file, opening")
    derp = pickle.load(open(args.fname, 'rb'))

    print(derp.ranges)
    
    if not 'maxorder' in dir(derp):
        derp.maxorder = min(5, min(derp.N))

    if not 'individual_grids' in dir(derp):
        derp.individual_grids = { col : (np.logspace if col == ['R', 'M', 'M2'] else np.linspace)(x[0], x[1], N) for x,col,N in zip(derp.ranges,
                                                                                                                                    derp.column_names,
                                                                                                                                    derp.N)}
        
    # Set the requested order
    derp.set_order(args.order)

    # Run validation
    if os.path.exists('%s_validations.dat' % args.fname):
        import pandas as pd
        print("Found existing validation set for this table, using that instead of running new ones.")
        results = pd.read_csv('%s_validations.dat' % args.fname)
    else:
        results = derp.validate(args.validate, Nkids=args.psize)

        # Save the results
        results.to_csv("%s_validations.dat" % args.fname)

    # Visualize them
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    
    # My screen is too fancy
    matplotlib.rcParams['figure.dpi'] = 300

    # Compute some new things
    results['q'] = results['M2']/results['M']
    results['Mtot'] = results['M'] + results['M2']

    print(results)
    
    results['frac_error'] = results['a_f_table_1']/results['a_f_direct'] - 1.

    if args.bad:
        hires_bins = np.linspace(-0.1, 0.1, 12)
        low_bin = np.min(results['frac_error'])
        high_bin = np.max(results['frac_error'])

        bins = np.concatenate( (np.linspace(low_bin, hires_bins[0], 10),
                                hires_bins[1:-1],
                                np.linspace(hires_bins[-1], high_bin, 10)) )
        
        plt.xscale('symlog', linthresh=0.1)
        plt.gca().axvline(-0.1, linestyle=':', linewidth=1, color='k')
        plt.gca().axvline(0.1, linestyle=':', linewidth=1, color='k')
    else:
        bins = 40
    
    plt.hist(results['frac_error'], bins=bins, alpha=0.5)

    plt.yscale('log')
    plt.title('%s performance, random loguniform sampling' % args.fname)
    plt.ylabel('Counts')
    plt.xlabel('Fractional error in linear interpolation')
    plt.show()

    # Now show which systems show poor performance
    badboiz = results[np.abs(results['frac_error']) > 0.01]

    cols = ['a_DCO', 'q', 'Mtot', 'R', 'e']
    logcols = ['q', 'R', 'Mtot']
    fig, axs = plt.subplots(len(cols), 1)
    fig.set_size_inches(10, 24)
    
    for ax,col in zip(axs, cols):

        if col in logcols:
            ax.set_xscale('log')
            bins = np.logspace(np.log10(np.min(badboiz[col])),
                               np.log10(np.max(badboiz[col])),
                               20)
        else:
            bins = 20
            
        ax.hist(badboiz[col], bins=bins)
        ax.set_yscale('log')
        ax.set_xlabel(col)
        ax.set_ylabel("Counts")
        
    plt.savefig("detailed_errors.pdf", dpi=300)
        
else:
    params = parseParams(args)
    print("Producing table with parameters", params)

    derp = approxMergerTime(params)
    
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
