# Cosmologically Coupled BBH Evolution
Copyright (c) 2022 Kevin Croker, GNU GPL v3

---

## Overview
Thank you for taking an interest in cosmologically coupled black hole dynamics!
For cosmologically coupled black holes, the merger masses are not representative of the birth masses, because BH masses grow in time due to a purely strong-field GR effect.
Cosmological coupling strongly impacts the evolution of the binary: adiabatically collapsing even very distant orbits into the GW-radiative capture regime within a Hubble time, while preserving eccentricity.

This package allows you to produce interpolation table python pickles that enable rapid (~1e-6s) computation of merger time, given initial binary orbital parameters: eccentricity, semi-major axis, primary mass, mass ratio, and birth mass.  This enables MC parameter inference over simulated cosmologically coupled BBH populations.

## Usage

Default settings are appropriate for data-driven birth masses as found in Globular Clusters[^1], Gaia DR3[^2] and Cygnus-X1[^3] BHs.
To produce a table across 14 processors in parallel, run

> $ python3 make_table.py --psize 14 sample_table_k3.p

The name of the table will be `sample_table_k3.p`.  This pickle can be loaded, and its `approxMergerTime()` method can then be called.
Example usage of this object is provided in `demo.py`.

Generation of more specific tables can be done with command line flags.  For example, tables suitable for BBH populations produced by default configuration COSMIC runs, which assume a TOV limit lower bound on BH mass, and a coupling of _k_=0.5 consistent with this assumption:

> $ python3 make_table.py --psize 14 khalf_cosmic.p --M 2.2:55:10 --R 0.01:1e3:10 --k 0.5

The syntax is "lower bound:upper bound:mesh samples."
The range for mass _M_ is applied for both primary and secondary masses, so mass ratio _q_ = 1 can be approximated at low and high ends.
Semi-major axis and mass are meshed in logspace.
Eccentricity and scale factor are meshed linearly, between the bounding redshifts specified with `--z`.
For a full list of options, along with detailed values, use the `--help` flag.

Note that approximation of binaries with parameters outside table ranges will use the nearest binary configuration that is tabulated.
Based on coupling strength, this can lead to large fractional errors.
Always validate your tables, as described below.

## Validation

Tables can be validated by direct integration of randomly sampled binary parameters:

> $ python3 make_table.py --psize 14 --validate 100000 sample_table_k3.p

This will write out a CSV to `sample_table_k3.p_validations.dat` and then show overall performances.
The data in this file should be suitable for estimating systematic errors from the use of the table in MC inference.

Fractional error in merger scale factor will be displayed as a histogram.
Distributions for individual binary parameters of systems with fractional error performance worse than 1% will be saved
to `detailed_errors.pdf`.
For cases that show large fractional errors, you can add the `--bad` flag to a validation run and get a linear scale around +/- 10% fractional error, and logarithmic otherwise.

## Caveats and Todo

- The tables can perform badly as *k* approaches 0.  For *k*=3, I have found very good performance with the current parameters in `make_table.py`, e.g. only 1% of systems have fractional error in merger scale factor greater than 1%, and all but 0.1% of systems have fractional error less than 5%.  The resulting table is ~4.2M and fits in L2 or L3 cache.
- The current code does not use `astropy`, but should.  It will be ported soon.
- The current code computes binary evolution via the extended Peters Equations assuming *conformal time*.
See Croker & Weiner (2019), Section 3 on why working in conformal time may be more appropriate for extending SR predictions across cosmological timescales.  (Contrast this to the usual intuition that one should work in proper (or "self") time, which would correspond to timelike arc for RW comoving observers.)
I've not formed a strong opinion on this, and I probably won't reach one without a fully GR treatment of the coupled binary problem.
Working in "cosmic" proper time can be achieved by modifying `c3o_binary_better.py` appropriately.
Email me if you can't figure it out.
- I will stop using pickles, because getting the `precomputation.py` file to load as a module/package is annoying and hacky.

## How to Cite

If you find this package useful, please feel free to cite the following papers in your work:
- Tracking of radiative losses and adiabatic orbital decay simultaneously[^4]
- First precisely LIGO detector-selected population study with COSMIC[^5]

## References

[^1]: Rodriguez, C. L., “Constraints on the Cosmological Coupling of Black Holes from the Globular Cluster NGC 3201”, <i>The Astrophysical Journal Letters</i>, vol. 947, no. 1, 2023. [doi:10.3847/2041-8213/acc9b6](https://doi.org/10.3847/2041-8213/acc9b6).

[^2]: Andrae, R. and El-Badry, K., “Constraints on the cosmological coupling of black holes from Gaia”, <i>Astronomy and Astrophysics Letters</i>, vol. 673, 2023. [doi:10.1051/0004-6361/202346350](https://doi.org/10.1051/0004-6361/202346350).

[^3]: Miller-Jones, J. C. A., “Cygnus X-1 contains a 21-solar mass black hole—Implications for massive star winds”, <i>Science</i>, vol. 371, no. 6533, pp. 1046–1049, 2021. [doi:10.1126/science.abb3363](https://doi.org/10.1126/science.abb3363).

[^4]: Croker, K. S., Nishimura, K. A., and Farrah, D., “Implications of Symmetry and Pressure in Friedmann Cosmology. II. Stellar Remnant Black Hole Mass Function”, <i>The Astrophysical Journal</i>, vol. 889, no. 2, 2020. [doi:10.3847/1538-4357/ab5aff](https://doi.org/10.3847/1538-4357/ab5aff).

[^5]: Croker, K. S., Zevin, M., Farrah, D., Nishimura, K. A., and Tarlé, G., “Cosmologically Coupled Compact Objects: A Single-parameter Model for LIGO-Virgo Mass and Redshift Distributions”, <i>The Astrophysical Journal Letters</i>, vol. 921, no. 2, 2021. [doi:10.3847/2041-8213/ac2fad](https://doi.org/10.3847/2041-8213/ac2fad).