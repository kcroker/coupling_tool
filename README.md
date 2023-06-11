# Cosmologically Coupled BBH Evolution
Copyright (c) 2022 Kevin Croker, GNU GPL v3

---

## Overview
Thank you for taking an interest in cosmologically coupled black hole dynamics!
For cosmologically coupled black holes, the merger masses are not representative of the birth masses, because BH masses grow in time due to a purely strong-field GR effect.
Cosmological coupling strongly impacts the evolution of the binary: adiabatically collapsing even very distant orbits into the GW-radiataive capture regime within a Hubble time, while preserving eccentricity.

This package allows you to produce interpolation table python pickles that enable rapid (~1e-6s) computation of merger time, given initial binary orbital parameters: eccentricity, semi-major axis, primary mass, mass ratio, and birth mass.  This enables MC parameter inference over simulated cosmologically coupled BBH populations.

## Usage

To make a basic table on 14 processors, with sub-percent precision at maximal coupling, across the range of typical COSMIC output remnant binaries:
> $ python3 make_table.py --psize 14 sample_table_k3.p

The name of the table will be `sample_table_k3.p`.  This pickle can be loaded, and its `approxMergerTime()` method can then be called.
This generator hardcodes the table ranges at present, but can easily be modified to pull these parameters from the command line, or hardcode them as you wish.

Example usage of this object is provided in `demo.py`.

## Validation

Tables can be validated by direct integration of randomly sampled binary parameters:

> $ python3 make_table.py --psize 14 --validate 100000 sample_table_k3.p

This will write out a CSV to `sample_table_k3.p_validations.dat` and then show overall performances.
Fractional error in merger scale factor will be displayed as a historgram.
Distributions for individual binary parameters of systems with fractional error performance worse than 1% will be saved
to `detailed_errors.pdf`.

## Caveats and Todo

- The tables can perform very badly as *k* approaches 0.  For *k*=3, I have found very good performance with the current parameters in `make_table.py`, e.g. only 1% of systems have fractional error in merger scale factor greater than 1%, and all systems have fractional error less than 5%.  The resulting table is ~4.2M and fits in L2 or L3 cache.
- The current code does not use `astropy`, but should.  It will be ported soon.
- The current code computes binary evolution via the modified Peters Equations assuming *conformal time*.
See Croker & Weiner (2019), Section 3 on why working in conformal time may be more appropriate for extending SR predictions across cosmological timescales.  (Contrast this to the usual intuition that one should work in proper (or "self") time, which would correspond to timelike arc for RW comoving observers.)
I've not formed a strong opinion on this, and I probably won't reach one without a fully GR treatment of the coupled binary problem.
Working in "cosmic" proper time can be achieved by modifying `c3o_binary_better.py` appropriately.
Email me if you can't figure it out.

## References

If you find this package useful, please feel free to cite the following papers in your work:
- [Derivation using adiabatic invariants, Section 3](https://iopscience.iop.org/article/10.3847/1538-4357/ab5aff#apjab5affs3)
- [First real detector-selected population study](https://iopscience.iop.org/article/10.3847/2041-8213/ac2fad)

