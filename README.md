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

## Caveats and Todo

- The current code does not use `astropy`, but should.  It will be ported soon.
- The current code computes binary evolution via the modified Peters Equations assuming non-conformal time.
This might not be correct, but can be easily adjusted and investigated.

## References

If you find this package useful, please feel free to cite the following papers in your work:
- [Derivation using adiabatic invariants, Section 3](https://iopscience.iop.org/article/10.3847/1538-4357/ab5aff#apjab5affs3)
- [First real detector-selected population study](https://iopscience.iop.org/article/10.3847/2041-8213/ac2fad)

