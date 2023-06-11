#
# This version will randomly draw N points in binary orbital parameter
# space, that plausibly merge within a Hubble time
#
import argparse
from scipy.stats import loguniform, uniform
import numpy as np
import pandas as pd
import multiprocessing
import c3o_binary_better as c3o
import tensorflow as tf
from tensorflow import keras

parser = argparse.ArgumentParser(description="Keras ML model of adiabatically extended Peters' equations for cosmological coupling",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("model",
                    help="Name of model to train")

parser.add_argument("--psize",
                   help="How many children to spawn during computation",
                   type=int,
                   default=1)

parser.add_argument("--batchsize",
                    help="Size of each training, validation, and testing batch",
                    type=int,
                    default=1000)

parser.add_argument("--existing",
                    help="Use a library of existing training data until exhausted")

#parser.add_argument("--validate",
#                    help="Number of random validation pulls to examine",
#                    type=int,
#                    default=1000)

#parser.add_argument("--order",
#                    help="Order at which to use map_coordinates()",
#                    type=int,
#                    default=1)

parser.add_argument("--k",
                    help="Value of k for this table",
                    type=float,
                    default=0.0)

parser.add_argument("--q",
                    help="Range of mass ratio q in q_min:q_max format",
                    default='1e-1:1')

parser.add_argument("--e",
                    help="Range of eccentricity e in e_min:e_max format",
                    default='0:0.99')

parser.add_argument("--a_DCO",
                    help="Range of double compact object scale factors a in a_min:a_max format",
                    default='0.01:0.9999')

# These variables all have pretty large ranges, so we should be using loguniform
# distributions

parser.add_argument("--R",
                    help="Range of semi-major axis R in R_min:R_max format",
                    default='1e-3:1e4')

parser.add_argument("--M",
                    help="Range of primary mass M in M_min:M_max format",
                    default='0.1:55')

parser.add_argument("--log",
                    help="Variables from which to pull samples uniformly in logspace",
                    default='R,M')

args = parser.parse_args()

# Parse out the ranges
logrange = args.log.split(',')
params = ['q', 'e', 'a_DCO', 'R', 'M']
dists = {}

for param in params:
    try:
        low,high = [float(x) for x in eval('args.%s' % param).split(':')]

        # Replace the argument with a distribution from which random variates can be drawn
        dists[param] = loguniform(low, high) if param in logrange else uniform(low, high - low)
    except:
        print("ded")

# Make initial values
    
# Establish the Datasets for TensorFlow
#train_dataset = tf.data.Dataset.from_tensors(datasets[:args.batchsize])

#for element in train_dataset:
#    print(element)

#quit()
    
#val_dataset = tf.data.Dataset.from_tensors(datasets[args.batchsize:2*args.batchsize])
#test_dataset = tf.data.Dataset.from_tensors(datasets[2*args.batchsize:])

stash_count = 0
def get_dataset(size, psize=1, label=None, existing=None):

    # Count of batches for this run
    global stash_count
    
    # If its a new model, we can train it on already existing data
    dataset = None
    for fname in existing:
        try:
            dataset = pd.read_hdf(fname, key="binaries")

            if 'periastron' not in dataset.columns:
                dataset['periastron'] = dataset['R']*(1. - dataset['e'])

            break
        except Exception as e:
            print("Failed to get data from", fname, e)
            continue

    if dataset is None:
        # Generate initial conditions
        dataset = pd.DataFrame(np.asarray([dist.rvs(size=size) for dist in dists.values()]).T, columns=params)
        
        # Set the coupling strength to be the same across everything
        dataset['k'] = args.k

        # Spawn offspring to compute exact results
        with multiprocessing.Pool(args.psize,
                                  initializer=c3o.setup_locking,
                                  initargs=(multiprocessing.Lock(),)) as p:

            # Take the first 2/3.  The final 1/3 is for testing.
            dataset['a_merger'] = pd.concat(p.map(c3o.c3o_binary_a_worker, np.array_split(dataset.copy(), psize)))

        # Remove the k column, because its unnecessary for fitting
        dataset.drop(columns=['k'], inplace=True)

        # Add periastron
        dataset['periastron'] = dataset['R']*(1. - dataset['e'])
        
        # If stash, save it, so that we can train off of it
        # later, and compare model performance
        if args.existing:
            # If no label is given, use the time
            if label is None:
                import time
                label = str(time.time())
            dataset.to_hdf("%s/%s_%.10d.hdf5" % (args.existing, label, stash_count), "binaries")
            stash_count += 1

    # Return the datasets
    return dataset

# Now build a Keras model
from tensorflow.keras import layers

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='path/to/my/model_{epoch}',
        save_freq='epoch'),
    keras.callbacks.TensorBoard(log_dir='./logs')
]

# Open a strategy scope.
#with strategy.scope():

try:
    model = keras.models.load_model(args.model)
    print("Loaded model from", args.model)
except:
    print("Did not find model at", args.model, "\nMaking a new one...")

    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    # The model takes 5 input parameters, goes through 4 dense layers, and outputs a single number
    inputs = keras.Input(shape=(len(params)+1,), name='BinarySpec')
    categorizer = layers.Dense(5, activation="relu")(inputs)
    valler = layers.Dense(32, activation="linear")(inputs)
    x = layers.Concatenate()([categorizer, valler])
    x = layers.Dense(32, activation="linear")(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    outputs = layers.Rescaling(2)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    loss = tf.keras.losses.MeanAbsolutePercentageError(
        reduction="auto", name="mean_absolute_percentage_error"
    )

    model.compile(
    optimizer='adam',
    loss=loss)
    #metrics=[tf.keras.metrics.RootMeanSquaredError()])

    #model.compile(optimizer='adam',
    #              loss=loss)

model.summary()


# List all existing data if present
if args.existing:
    import glob
    existing_files = glob.glob("%s/*.hdf5" % args.existing)
    if len(existing_files):
        print("Successfully found", len(existing_files), "data files.  Will pull from these first.")
        existing_files = iter(existing_files)
    else:
        print("No data found.")

# Label these batches with the start time
import time
label = str(time.time())

while True:

    # Get a dataset
    training = get_dataset(args.batchsize,
                           args.psize,
                           label=label,
                           existing=existing_files)

    print(training)

    # Get targets
    targets = training['a_merger']
    
    print(targets)

    # Get inputs
    inputs = training.drop(columns=['a_merger'])

    # Now train on this batch
    model.train_on_batch(inputs[100:].to_numpy(),
                         targets[100:].to_numpy())

    # How did we do?
    print("Performance: ", model.test_on_batch(inputs[:100].to_numpy(),
                                               targets[:100].to_numpy()))

    # Visualize performance
    derp = training[:100].copy();
    derp['estimates'] = model.predict(inputs[:100].to_numpy())
    print(derp)

    model.save(args.model)

# import time

# # Get a test run
# t1 = time.time()
# dataset, mergers = get_dataset(1000, args.psize)
# t2 = time.time()
# print("Generation took: ", (t2-t1)*args.psize/1000, "s per binary")
      
# durs = []
# for i in range(10):
    
#     t1 = time.time()
#     model.evaluate(datasets[i*100:(i+1)*100].to_numpy())
#     t2 = time.time()
#     delta = (t2-t1)/100.
#     print("Eval took: ", delta)
#     durs.append(delta)

# # Final stats, dropping the bootstrap one that has to cache it
# durs = np.asarray(durs[1:])
# print(durs)
# print("Delta:", np.average(durs), "+/-", np.sqrt(np.var(durs)))

