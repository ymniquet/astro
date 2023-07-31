Various simulation codes used in my notes on astronomy and astrophysics (see https://astro.ymniquet.fr; the notes are in French, but the codes are documented in English). Author: Yann-Michel Niquet (ymniquet@gmail.com).

1) Helpers:
- constants.py: Physical constants.
- utils.py: Miscellaneous utilities used by the codes below.
- marplotlibrc: A sample matplotlib configuration file.

2) Codes:
- degenerescence.py: Density of states and particle distribution in an ideal Fermi gas.
- gazparfait.py: Equation of state of a non-relativistic ideal Fermi gas in the classical and degenerate regimes.
- corpsnoir.py: Black-body spectrum.
- opacite.py: Comparison between various opacity models (Kramers, OPAL...).
- PPCNO.py: Proton-proton chains and CNO nuclear reaction rates.
- etoile.py: Solution of the hydrostatic model of a star.
- Chandrasekhar.py: Radius of a degenerate, relativistic object and the Chandrasekhar mass limit.

3) Others:
- OPAL/: The OPAL opacity tables and Fortran bindings (see https://opalopacity.llnl.gov/existing.html). See OPAL/opal.f for information on how to compile (used by opacite.py and etoile.py).
