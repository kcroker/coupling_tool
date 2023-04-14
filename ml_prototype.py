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

parser.add_argument("fname",
                    help="File to output or read in")

parser.add_argument("--psize",
                   help="How many children to spawn during computation",
                   type=int,
                   default=1)

parser.add_argument("--batchsize",
                    help="Size of each training, validation, and testing batch",
                    type=int,
                    default=1000)

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

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='path/to/my/model_{epoch}',
        save_freq='epoch'),
    keras.callbacks.TensorBoard(log_dir='./logs')
]

# Now build a Keras model
from tensorflow.keras import layers

# Create a MirroredStrategy.
# strategy = tf.distribute.MirroredStrategy()

# Open a strategy scope.
#with strategy.scope():

# Everything that creates variables should be under the strategy scope.
# In general this is only model construction & `compile()`.
# The model takes 5 input parameters, goes through 2 dense layers, and outputs a single number
inputs = keras.Input(shape=(len(params),), name='BinarySpec')
x = layers.Dense(64, activation="relu")(inputs)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(16, activation="relu")(x)
x = layers.Dense(8, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid", name='Magic')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
    
loss = tf.keras.losses.MeanAbsolutePercentageError(
    reduction="auto", name="mean_absolute_percentage_error"
)

model.compile(optimizer='adam',
              loss=loss)
model.summary()

# Train the model on all available devices.
while True:

    # Generate initial conditions
    datasets = pd.DataFrame(np.asarray([dist.rvs(size=args.batchsize) for dist in dists.values()]).T, columns=params)

    # Set the coupling strength to be the same across everything
    datasets['k'] = args.k

    # Spawn offspring to compute exact results
    with multiprocessing.Pool(args.psize,
                              initializer=c3o.setup_locking,
                              initargs=(multiprocessing.Lock(),)) as p:

        # Take the first 2/3.  The final 1/3 is for testing.
        mergers = pd.concat(p.map(c3o.c3o_binary_a_worker, np.array_split(datasets.copy(), args.psize)))

    # Remove the k column, because its unnecessary for fitting
    datasets.drop(columns=['k'], inplace=True)

    # Now train on this batch
    model.train_on_batch(datasets[100:].to_numpy(),
                         mergers[100:].to_numpy())

    # How did we do?
    print("Performance: ", model.test_on_batch(datasets[:100].to_numpy(),
                                               mergers[:100].to_numpy()))



