#!/usr/bin/python

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 2023.09

from math import pi, sqrt
import constants as cst
import sys
import utils
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from collections import namedtuple

try: # Try import the OPAL opacity bindings.
  import opal
  _OPAL = True
except:
  _OPAL = False

"""Solve the hydrostatic model of a star."""

#############################################
### A container for the star composition. ###
#############################################

# X is the mass fraction of hydrogen, Y the mass fraction of helium, and Z = 1-X-Y the mass fraction of "metals".
# XCNO is the mass fraction of C/N/O for CNO fusion cycle.

StarComposition = namedtuple("StarComposition", ["X", "Y", "Z", "XCNO"])

############################################
### Hydrostatic main sequence star class ###
### with Kramers' opacity laws.          ###
############################################

class KramersHydrostaticMainSequenceStar:
  """Hydrostatic main sequence star class with Kramers' opacity laws."""

  def __init__(self, M, ngrid = 2**10, rgrid = 1.):
    """Initialize star with total mass M (kg).
       ngrid is the number of points on the geometric finite-differences grid,
       and rgrid is the ratio between the first and last grid steps."""
    if M <= 0.: raise ValueError("Error, M <= 0.")
    self.M = M
    self.set_grid(ngrid, rgrid) # Set-up grid.
    self.set_composition(0.7346, 0.0165, 0.0165) # Default composition = Sun.

  def set_composition(self, X, Z = 0., XCNO = 0.):
    """Set star composition.
       X is the hydrogen mass fraction, and Z the metallicity (default Z = 0).
       The Helium mass fraction is thus Y = 1-X-Z.
       X, Z and XCNO can either be scalars (homogeneous composition) or arrays on the grid."""
    if np.any(X < 0.) or np.any(X > 1.): raise ValueError("Error, X < 0 or X > 1.")
    if np.any(Z < 0.) or np.any(Z > 1.): raise ValueError("Error, Z < 0 or Z > 1.")
    if np.any(X+Z > 1.): raise ValueError("Error, X+Z > 1.")
    if np.any(XCNO < 0.) or np.any(XCNO > 1.): raise ValueError("Error, XCNO < 0 or XCNO > 1.")
    Y = 1.-X-Z
    self.comp = StarComposition(X, Y, Z, XCNO)
    self.mu = 4.e-3/(6.*X+Y+2.) # Average molecular mass in kg/mol.
    self.set_cp() # Set default isobaric heat capacity per particle.
    self.set_opacity_parameters() # Set default opacity parameters.
    self.set_surface_mass_density() # Set default mass density at the surface of the star.
    self.enable_radiative_pressure() # Account for radiative pressure.

  def set_cp(self, cp = 5.*cst.kb/2.):
    """Set isobaric heat capacity *per particle*.
       (cp = 5kb/2 for an ideal gas with no internal degrees of freedom)."""
    if cp <= cst.kb: raise ValueError("Error, cp <= kb.")
    self.Cp = cst.Na*cp/self.mu # Isobaric heat capacity per unit of mass (J/K/kg).
    self.gamma = cp/(cp-cst.kb) # Adiabatic coefficient.
    self.reset_solution()

  def set_surface_mass_density(self, rhos = 3.e-4):
    """Define the hydrostatic surface of the star as the radius R at which the
       mass density matches the input rhos (kg/m^3)."""
    if rhos <= 0.: raise ValueError("Error, rhos <= 0.")
    self.ns = rhos/np.atleast_1d(self.mu)[-1]
    self.BC = "density"
    self.reset_solution()

  def set_surface_pressure(self, Ps = cst.atm):
    """Define the hydrostatic surface of the star as the radius R at which the
       pressure matches the input Ps (Pa)."""
    if Ps <= 0.: raise ValueError("Error, Ps <= 0.")
    self.Ps = Ps
    self.BC = "pressure"
    self.reset_solution()

  def enable_radiative_pressure(self, enable = True):
    """Account for radiative pressure if enable = True."""
    self.prad = enable
    self.reset_solution()

  def set_grid(self, ngrid, rgrid = 1.):
    """Set-up a geometric finite-differences grid.
       ngrid is the number of points and rgrid is the ratio between the first and last grid steps."""
    if ngrid < 2: raise ValueError("Error, ngrid < 2.")
    if rgrid <= 0.: raise ValueError("Error, rgrid <= 0.")
    if rgrid == 1.:
      r = np.linspace(0., 1., ngrid)
    else:
      r = (1.-(1./rgrid)**(np.arange(0, ngrid, dtype = int)/(ngrid-1)))/(1.-1./rgrid)
      r[ 0] = 0.
      r[-1] = 1.
    self.init_grid(r)

  def init_grid(self, r):
    """Initialize the finite-differences grid.
       r is a mesh of the [0, 1] interval such that r[0] = 0 and r[-1] = 1."""
    try:
      self.r = np.array(r, dtype = float)
      if self.r.ndim != 1: raise
    except:
      raise TypeError("Error, r can not be converted into a 1D array of floats.")
    self.ngrid = r.size
    if r[ 0] != 0.: raise ValueError("Error, r[0] != 0.")
    if r[-1] != 1.: raise ValueError("Error, r[-1] != 1.")
    if any(self.r != np.sort(self.r)): raise ValueError("Error, r is not sorted.")
    self.dr = self.r[1:]-self.r[:-1] # Grid steps.
    rc = (self.r[:-1]+self.r[1:])/2. # Mid-points.
    self.dVl = np.empty(self.ngrid) # Volume of the half-shell on the left of each grid point.
    self.dVl[1:] = 4.*pi*(self.r[1:]**3-rc**3)/3.
    self.dVl[ 0] = 0.
    self.dVr = np.empty(self.ngrid) # Volume of the half-shell on the right of each grid point.
    self.dVr[:-1] = 4.*pi*(rc**3-self.r[:-1]**3)/3.
    self.dVr[ -1] = 0.
    self.dVc = self.dVl+self.dVr # Volume of the shell centered on each grid point.
    self.reset_solution()

  def to_grid(self, x):
    """Expand the quantity x on the grid.
       x must either be a scalar or a 1D-like object with size ngrid."""
    n = np.ndim(x)
    if n == 0:
      return np.full(self.ngrid, x)
    elif n == 1:
      if len(x) != self.ngrid:
        raise TypeError("Error, x must either be a scalar or a 1D-like object with size ngrid.")
      return np.array(x)
    else:
      raise TypeError("Error, x must either be a scalar or a 1D-like object with size ngrid.")

  #####################################################
  # Equations of state of a non-degenerate ideal gas. #
  #####################################################

  # Pressure as a function of density & temperature.

  def pressure_gas(self, n, T):
    """Return the (non-degenerate) gas pressure P = n*R*T (Pa).
       n is the density (mol/m^3) and T the temperature (K)."""
    return n*cst.R*T

  def pressure_rad(self, n, T):
    """Return the radiative pressure P = 4*sigma*T**4/(3*c) (Pa).
       n is the density (mol/m^3) and T the temperature (K)."""
    return 4.*cst.sigma*self.prad*T**4/(3.*cst.c)

  def pressure(self, n, T):
    """Return the total gas & radiative pressure (Pa) as a function of
       density n (mol/m^3) and temperature T (K)."""
    return self.pressure_gas(n, T)+self.pressure_rad(n, T)

  # Density as a function of pression & temperature.

  def density(self, P, T):
    """Return the density n (mol/m^3) as a function of
       pressure P (Pa) and temperature T (K)."""
    return (P/T-4.*cst.sigma*self.prad*T**3/(3.*cst.c))/cst.R

  # Temperature as a function of density & pressure.

  # Vectorize the solution of P = 4.*cst.sigma*T**3/(3.*cst.c)+n*cst.R*T.
  _vectorized_temperature_prad = np.vectorize(lambda n, P: np.real(np.roots([4.*cst.sigma/(3.*cst.c), 0., 0., n*cst.R, -P])[3]), otypes = [float])

  def temperature(self, n, P):
    """Return the temperature T (K) as a function of
       density n (mol/m^3) and pressure P (Pa)."""
    if self.prad:
      return self._vectorized_temperature_prad(n, P) # Radiative pressure enabled.
    else:
      return P/(n*cst.R) # Radiative pressure disabled.

  ########################################
  # Opacity.                             #
  # Override as needed in child classes. #
  ########################################

  def set_opacity_parameters(self, gbf = 1., tbf = 10., gff = 1.):
    """Set opacity parameters gbf (Gaunt factor for bound-free transitions), tbf (Guillotine factor for bound-free transitions)
       and gff (Gaunt factor for free-free transitions)."""
    self.gbf = gbf # Gaunt factor for bound-free transitions.
    self.tbf = tbf # Guillotine factor for bound-free transitions.
    self.gff = gff # Gaunt factor for free-free transitions.
    self.reset_solution()

  def kappa_bb(self, rho, T):
    """Return Kramers' opacity kappa (m^2/kg) for bound-bound transitions.
       rho is the mass density (kg/m^3) and T the temperature (K)."""
    return np.zeros_like(rho)

  def kappa_bf(self, rho, T):
    """Return Kramers' opacity kappa (m^2/kg) for bound-free transitions.
       rho is the mass density (kg/m^3) and T the temperature (K)."""
    return 4.34e21*(self.gbf/self.tbf)*(1.+self.comp.X)*self.comp.Z*rho*T**-3.5

  def kappa_ff(self, rho, T):
    """Return Kramers' opacity kappa (m^2/kg) for free-free transitions.
       rho is the mass density (kg/m^3) and T the temperature (K)."""
    return 3.68e18*self.gff*(1.+self.comp.X)*(1.-self.comp.Z)*rho*T**-3.5

  def kappa_Compton(self, rho, T):
    """Return Kramers' opacity kappa (m^2/kg) for Thomson diffusion.
       rho is the mass density (kg/m^3) and T the temperature (K)."""
    return np.full_like(rho, 0.02*(1.+self.comp.X))

  def kappa(self, rho, T):
    """Return Rosseland's average opacity kappa (m^2/kg).
       rho is the mass density (kg/m^3) and T the temperature (K)."""
    return self.kappa_bb(rho, T)+self.kappa_bf(rho, T)+self.kappa_ff(rho, T)+self.kappa_Compton(rho, T)

  #########################################################
  # Power per unit of mass produced by nuclear reactions. #
  # Override as needed in child classes.                  #
  #########################################################

  def epsilon_PP(self, rho, T):
    """Return the power per unit of mass epsilon (W/kg) for the proton-proton chain.
       rho is the mass density (kg/m^3) and T the temperature (K)."""
    one3 = 1./3.
    two3 = 2./3.
    T6 = T/1.e6
    rho3 = rho/1.e3
    X = self.comp.X
    fPP = 1.+0.133*X**2*np.sqrt((3.+X)*rho3)/T6**1.5
    psiPP = 1.+1.412e8*(1./X-1.)*np.exp(-49.98*T6**(-one3))
    CPP = 1.+0.0123*T6**one3+0.0109*T6**two3+0.000938*T6
    return 1.e-4*2.38e6*rho3*X**2*fPP*psiPP*CPP*T6**(-two3)*np.exp(-33.8*T6**(-one3))

  def epsilon_CNO(self, rho, T):
    """Return the power per unit of mass epsilon (W/kg) for the CNO cycle.
       rho is the mass density (kg/m^3) and T the temperature (K)."""
    one3 = 1./3.
    two3 = 2./3.
    T6 = T/1.e6
    rho3 = rho/1.e3
    X = self.comp.X
    XCNO = self.comp.XCNO
    CCNO = 1.+0.0027*T6**one3-0.00778*T6**two3-0.000149*T6
    return 1.e-4*8.67e27*rho3*X*XCNO*CCNO*T6**(-two3)*np.exp(-152.28*T6**(-one3))

  def epsilon_nuclear(self, rho, T):
    """Return the power per unit of mass epsilon (W/kg) produced by nuclear reactions.
       rho is the mass density (kg/m^3) and T the temperature (K)."""
    return self.epsilon_PP(rho, T)+self.epsilon_CNO(rho, T)

  ############################
  # Miscellaneous integrals. #
  ############################

  # Mass.

  def mass(self, rho, R):
    """Return the enclosed mass m (kg) as a function of radial position.
       rho is the mass density (kg/m^3) and R the radius of the star (m)."""
    return (np.cumsum(rho*self.dVc)-rho*self.dVr)*R**3

  # Gravitational field.

  def gfield(self, rho, R):
    """Return the gravitational field g (m/s^2) as a function of radial position.
       rho is the mass density (kg/m^3), and R the radius of the star (m)."""
    m = np.cumsum(rho*self.dVc)-rho*self.dVr
    m[1:] /= self.r[1:]**2
    return cst.G*m*R

  # Luminosity.

  def luminosity(self, rho, epsilon, R):
    """Return the star luminosity L (W) as a function of radial position.
       rho is the mass density (kg/m^3), epsilon the power per unit of mass produced by nuclear reactions (W/kg), and R the radius of the star (m)."""
    rhoepsilon = rho*epsilon
    return (np.cumsum(rhoepsilon*self.dVc)-rhoepsilon*self.dVr)*R**3

  #####################################
  # Solve the equations for the star. #
  #####################################

  def homogeneous_star(self):
    """Homogeneous star model (approximate initial solution).
       Return star radius R (m) as well as mass density rho (kg/m^3), pressure P (Pa) and temperature T (K) on the grid."""
    # Estimate radius R and surface temperature Tc from the relations R/Rsun = sqrt(M/Msun) and Tc/Tsun = R/RSun.
    Msun = 2.e30 # kg.
    Rsun = 696340.e3 # m.
    Tsun = 5775. # K.
    R = Rsun*np.sqrt(self.M/Msun) # Estimated radius (m).
    V = 4.*pi*R**3/3. # Estimated volume (m^3).
    rho = np.full(self.ngrid, self.M/V) # Estimated density (kg/m^3).
    Tc = Tsun*self.M/Msun # Estimated surface temperature (K).
    T = Tc+(1.-self.r**2)*cst.G*self.mu*self.M/(2.*cst.R*R) # Temperature (K).
    P = self.pressure(rho/self.mu, T) # Pressure (Pa).
    return R, rho, P, T

  def solve(self, epsilon = 1.e-6, nmax = 1024, alphamin = 0.2, alphamax = 1., plot = False, guess = None):
    """Solve the equations for the star up to relative precision epsilon with at most nmax self-consistent iterations.
       The radius, pression, temperature and luminosity are considered as function(al)s of the mass density rho.
       Starting from a given mass density rho, the equation for the pressure P is first solved, then the temperature
       T(rho, P) is calculated from the equation of state of the ideal gas, and the luminosity L(rho, T) is computed.
       The temperature T -> T_new is next updated with the energy transfer equation, as well as the density
       rho -> rho_new(P, T_new) with the equation of state. The processus is iterated until |rho_new-rho| < epsilon*rho[0].
       To prevent or smooth mass sloshing from one iteration to the next, the density is actually updated as
       rho -> alpha*rho_new+(1-alpha)*rho, where 0 < alphamin <= alpha <= alphamax <= 1 is a mixing factor ;
       decrease alphamin as well as alphamax-alphamin if the calculation does not converge. If plot is True,
       plot the solution at each self-consistent iteration. If not None, guess = (R, rho, P, T) is an initial
       guess for the solution, with R the radius of the star (m), rho the mass density (kg/m^3), P the pressure (Pa)
       and T the temperature (K) on the grid."""

    def plot_solution(n, rho, P, T, L):
      """Plot solution at iteration n."""
      for m, plots in self.lastplots[-2:]: # Change line color and style for previous iterations.
        l = "--"
        c = "blue" if m == n-1 else "gray"
        for p in plots:
          p.set_color(c)
          p.set_linestyle(l)
      if len(self.lastplots) >= 3: # Remove oldest iterations.
        m, plots = self.lastplots.pop(0)
        for p in plots: p.remove() # !! WARNING : This segfaults if the figure has been closed. !!
      plots = []
      plt.ion()
      plt.figure(11)
      plots.append(plt.semilogy(self.r, rho/1.e3, "b-")[-1])
      plt.xlabel("$r/R$")
      plt.ylabel("$\\rho$ (g/cm$^3$)")
      plt.title(f"$n={n}$")
      plt.figure(12)
      plots.append(plt.semilogy(self.r, P/1.e9, "b-")[-1])
      plt.xlabel("$r/R$")
      plt.ylabel("$P$ (GPa)")
      plt.title(utils.flatex(f"$n={n}, P(0)={P[0]/1.e9:.3e}$ GPa"))
      plt.figure(13)
      plots.append(plt.semilogy(self.r, T, "b-")[-1])
      plt.xlabel("$r/R$")
      plt.ylabel("$T$ (K)")
      plt.title(f"$n={n}, T(R)={T[-1]:.0f}$ K")
      plt.figure(14)
      plots.append(plt.plot(self.r, L, "b-")[-1])
      plt.xlabel("$r/R$")
      plt.ylabel("$L$ (W)")
      plt.title(utils.flatex(f"$n={n}, L={L[-1]:.3e}$ W"))
      plt.show()
      self.lastplots.append((n, plots))

    # Reset solution.
    self.reset_solution()
    # Set-up convergence plots.
    if plot: self.lastplots = []
    # Self-consistent iterations.
    n = 0 # Number of self-consistent iterations.
    ierr = 0 # Return code.
    alphamax = max(alphamin, alphamax)
    alpha = alphamax # Mixing factor.
    print( "+-------+"+8*(14*"-"+"+"))
    print(f"| Iter. | {'Error':>12} | {'Alpha':>12} | {'R (km)':>12} | {'L (W)':>12} | {'T(0) (K)':>12} | {'T(R) (K)':>12} | {'P(0) (GPa)':>12} | {'P(R) (GPa)':>12} |")
    print( "+-------+"+8*(14*"-"+"+"))
    # Initial guess.
    if guess is not None:
      R, rho, P, T = guess
    else:
      R, rho, P, T = self.homogeneous_star() # Use homogeneous star as initial guess.
    L = self.luminosity(rho, self.epsilon_nuclear(rho, T), R) # Luminosity.
    print(f"| {n:5d} | {np.NaN:12.5e} | {alpha:12.5e} | {R/1e3:12.5e} | {L[-1]:12.5e} | {T[0]:12.5e} | {T[-1]:12.5e} | {P[0]/1e9:12.5e} | {P[-1]/1e9:12.5e} |")
    if plot: plot_solution(n, rho, P, T, L) # Plot initial guess if apppropriate.
    while True:
      # New self-consistent iteration.
      n += 1
      if n > nmax:
        print( "+-------+"+8*(14*"-"+"+"))
        print(f"Error, could not converge within {nmax} iterations.")
        ierr = 1
        break
      # Save density.
      rhop = np.copy(rho)
      # Compute gravitational field.
      g = self.gfield(rho, R)
      # Solve pressure equation.
      dPdr = -rho*g # Compute dP/dr.
      if self.BC == "pressure":
        P[-1] = self.Ps # Set pressure at the hydrostatic surface.
      elif self.BC == "density":
        P[-1] = self.pressure(self.ns, T[-1]) # Set mass density at the hydrostatic surface.
      else:
        raise ValueError("Error, BC must be 'pressure' or 'density'.")
      dPdr = (dPdr[:-1]+dPdr[1:])/2. # Average dP/dr at mid-points.
      dP = -np.cumsum(dPdr[::-1]*self.dr[::-1])*R # Integrate dP/dr.
      P[:-1] = P[-1]+dP[::-1]
      # Get temperature from the equation of state.
      Tp = self.temperature(rho/self.mu, P)
      # Compute luminosity.
      L = self.luminosity(rho, self.epsilon_nuclear(rho, Tp), R)
      # Update temperature with the energy transfer equation.
      dTdr_rad = -3.*self.kappa(rho, Tp)*rho*L/(64.*pi*cst.sigma*Tp**3) # dT/dr for radiative transfer.
      dTdr_rad[1:] /= (self.r[1:]*R)**2
      dTdr_rad = (dTdr_rad[:-1]+dTdr_rad[1:])/2. # Average dTdr_rad at mid-points.
      dTdr_cnv = -g/self.Cp # dT/dr for convection.
      dTdr_cnv = (dTdr_cnv[:-1]+dTdr_cnv[1:])/2. # Average dTdr_cnv at mid-points.
      # Sort out radiative & convective zones.
      convection = (dTdr_cnv > dTdr_rad)
      #convection[-1] = True # Enforce convection at the surface.
      dTdr = np.where(convection, dTdr_cnv, dTdr_rad)
      # Integrate dT/dr.
      dT = -np.cumsum(dTdr[::-1]*self.dr[::-1])*R
      T[ -1] = (L[-1]/(4.*pi*cst.sigma*R**2))**(1./4.) # Surface temperature from luminosity (Stefan-Boltzmann law).
      T[:-1] = T[-1]+dT[::-1]
      # Update density from the equation of state.
      rho = abs(self.density(P, T))*self.mu # Take abs for safety.
      # Compute error.
      error = np.sqrt(np.sum(((rho-rhop)/rhop[0])**2))#+np.sum(((T-Tp)/Tp[-1])**2))
      # Compute mixing factor.
      if n > 1:
        if error > errorp:                   # The error increases -> the input and output densities spread farther and farther apart.
          alpha = max(alphamin, alphap/1.33) # The calculation *may* thus be diverging -> reduce alpha for safety.
        else:
          alpha = min(alphamax, alphap*1.25) # Otherwise, increase alpha to speed up convergence.
      print(f"| {n:5d} | {error:12.5e} | {alpha:12.5e} | {R/1e3:12.5e} | {L[-1]:12.5e} | {T[0]:12.5e} | {T[-1]:12.5e} | {P[0]/1e9:12.5e} | {P[-1]/1e9:12.5e} |")
      # Check convergence.
      if error < epsilon:
        print( "+-------+"+8*(14*"-"+"+"))
        print(f"Converged in {n} iterations.")
        break
      # Mix input & output densities.
      rho = alpha*rho+(1.-alpha)*rhop
      # Save error and mixing factor.
      errorp = error
      alphap = alpha
      # Compute total mass.
      M = np.sum(rho*self.dVc)*R**3
      # Update star radius.
      R = R*(self.M/M)**(1./3.)
      # Plot solution if apppropriate.
      if plot:
        plot_solution(n, rho, P, T, L)
        if input() == "END": sys.exit(0) # Pause.
    # Store solution within the object.
    if ierr == 0:
      #plt.figure(20) # TEST
      #rc = (self.r[:-1]+self.r[1:])/2.
      #plt.semilogy(rc, -dTdr_cnv, "b-", label = "Cnv")
      #plt.semilogy(rc, -dTdr_rad, "r-", label = "Rad")
      #plt.semilogy(rc, -dTdr    , "k-", label = "Tot")
      #plt.xlabel("$r/R$")
      #plt.ylabel("$-dT/dr$ (K/m)")
      #plt.xlim(0., 1.)
      #plt.legend(loc = "upper right")
      self.set_solution(R, rho, P, T, L, convection)
      self.post_process_solution()
    if plot: del(self.lastplots) # Delete convergence plots data.
    return ierr

  def set_solution(self, R, rho, P, T, L, convection):
    """Set solution.
       R is the star radius (m); rho is the density (kg/m^3), L the luminosity (W), P the pressure (Pa), and T the temperature (K) on the grid.
       convection[0:ngrid-1] is True where the energy is transferred by convection between grid nodes."""
    self.R = R
    self.P = np.copy(P)
    self.T = np.copy(T)
    self.L = np.copy(L)
    self.rho = np.copy(rho)
    self.convection = np.copy(convection)

  def post_process_solution(self):
    """Post-process solution.
       Override as needed in child classes."""
    return # Nothing done here.

  def reset_solution(self):
    """Reset solution."""
    self.R = None
    self.P = None
    self.T = None
    self.L = None
    self.rho = None
    self.convection = None

  def get_solution(self):
    """Return solution as (R, rho, P, T), with R the radius of the star (m),
       rho the mass density (kg/m^3), P the pressure (Pa) and T the temperature (K) on the grid."""
    return self.R, self.rho, self.P, self.T

#############################################################################
### Hydrostatic main sequence star class                                  ###
### with modified Kramers' opacity laws.                                  ###
### The guillotine factor now reads:                                      ###
###   tbf = 0.71*(rho*(1.+comp.X))**0.2                                   ###
### with rho the mass density in kg/m^3. See:                             ###
### http://spiff.rit.edu/classes/phys370/lectures/statstar/statstar.html. ###
#############################################################################

class ModKramersHydrostaticMainSequenceStar(KramersHydrostaticMainSequenceStar):
  """Hydrostatic main sequence star class with modified Kramers' opacity laws:
     the guillotine factor reads tbf = 0.71*(rho*(1.+comp.X))**0.2
     with rho the mass density in kg/m^3."""

  def set_opacity_parameters(self, gbf = 1., gff = 1.):
    """Set opacity parameters gbf (Gaunt factor for bound-free transitions) and gff (Gaunt factor for free-free transitions)."""
    self.gbf = gbf # Gaunt factor for bound-free transitions.
    self.gff = gff # Gaunt factor for free-free transitions.
    self.reset_solution()

  def kappa_bf(self, rho, T):
    """Return Kramers' opacity kappa (m^2/kg) for bound-free transitions.
       rho is the mass density (kg/m^3) and T the temperature (K)."""
    tbf = 0.71*(rho*(1.+self.comp.X))**0.2
    return 4.34e21*(self.gbf/tbf)*(1.+self.comp.X)*self.comp.Z*rho*T**-3.5

##########################################################
### Hydrostatic main sequence star class               ###
### with extended Kramers' opacity laws accounting for ###
### absorption by H-, H2O & CO near the surface,       ###
### according to:                                      ###
### https://www.astro.princeton.edu/~gk/A403/opac.pdf. ###
##########################################################

class ExtKramersHydrostaticMainSequenceStar(ModKramersHydrostaticMainSequenceStar):
  """Hydrostatic main sequence star class with extended Kramers' opacity laws
     accounting for absorption by H-, H2O & CO near the surface, according to:
     https://www.astro.princeton.edu/~gk/A403/opac.pdf."""

  def kappa_Hm(self, rho, T):
    """Return opacity kappa (m^2/kg) for H-.
       rho is the mass density (kg/m^3) and T the temperature (K)."""
    return 3.48e-28*np.sqrt(self.comp.Z*rho)*T**7.7

  def kappa_H2O_CO(self, rho, T):
    """Return opacity kappa (m^2/kg) for H2O and CO.
       rho is the mass density (kg/m^3) and T the temperature (K)."""
    return 0.01*self.comp.Z

  def kappa(self, rho, T):
    """Return Rosseland's average opacity kappa (m^2/kg).
       rho is the mass density (kg/m^3) and T the temperature (K)."""
    return self.kappa_H2O_CO(rho, T)+1./(1./self.kappa_Hm(rho, T)+1./(self.kappa_bb(rho, T)+self.kappa_bf(rho, T)+self.kappa_ff(rho, T)+self.kappa_Compton(rho, T)))

###################################################
### Hydrostatic main sequence star class        ###
### with OPAL opacity tables. See:              ###
### https://opalopacity.llnl.gov/existing.html. ###
###################################################

if _OPAL: # Define only if OPAL bindings succesfully loaded.

  class OPALHydrostaticMainSequenceStar(KramersHydrostaticMainSequenceStar):
    """Hydrostatic main sequence star class with OPAL opacity tables. See:
       https://opalopacity.llnl.gov/existing.html."""

    def set_opacity_parameters(self):
      """Set opacity parameters (None for OPAL)."""
      self.reset_solution()

    # Vectorize opal.kappa.
    vectorized_kappa_opal = np.vectorize(lambda rho, T, X, Z: opal.kappa(rho, T, X, Z), otypes = [float])

    def kappa(self, rho, T):
      """Return OPAL Rosseland average opacity kappa (m^2/kg).
         rho is the mass density (kg/m^3) and T the temperature (K)."""
      return self.vectorized_kappa_opal(rho, T, self.comp.X, self.comp.Z)

    # Vectorize opal.check_kappa_bounds.
    _vectorized_check_kappa_bounds_opal = np.vectorize(lambda rho, T, X, Z: opal.check_kappa_bounds(rho, T, X, Z), otypes = [bool])

    def post_process_solution(self, **kwargs):
      """Post-process solution.
         Check that the temperature and density are within the OPAL tables throughout the whole radiative transfer zone."""
      KramersHydrostaticMainSequenceStar.post_process_solution(self, **kwargs)
      check = self._vectorized_check_kappa_bounds_opal(self.rho, self.T, self.comp.X, self.comp.Z)
      check[:-1] = check[:-1] | self.convection
      check[ 1:] = check[ 1:] | self.convection
      if np.all(check):
        print("The temperature and density are within the OPAL tables throughout the whole radiative transfer zone.")
      else:
        print("##################################################################################################################")
        print("# WARNING : The temperature and density are out of the OPAL tables in some parts of the radiative transfer zone. #")
        print("##################################################################################################################")

###############################################
### Class factory:                          ###
### Add postprocessing & plot extensions to ###
### an existing star class.                 ###
###############################################

def AddPlotExtensions(base):

  class ExtendedHydrostaticMainSequenceStar(base):
    """Hydrostatic main sequence star class with post-processing tools."""

    panels = True # Label plots (a), (b), ...

    def post_process_solution(self, **kwargs):
      """Compute gravitational field, etc..."""
      base.post_process_solution(self, **kwargs)
      self.m = self.mass(self.rho, self.R) # Enclosed mass (kg).
      self.g = self.gfield(self.rho, self.R) # Gravitational field (m/s^2).
      self.Pgas = self.pressure_gas(self.rho/self.mu, self.T) # Gaz pressure (Pa).
      self.Prad = self.pressure_rad(self.rho/self.mu, self.T) # Radiation pressure (Pa).
      #self.P = self.Pgas+self.Prad # Total pressure (Pa).
      self.enPP = self.epsilon_PP(self.rho, self.T) # Power per unit of mass produced by the proton-proton chain (W/kg).
      self.enCNO = self.epsilon_CNO(self.rho, self.T) # Power per unit of mass produced by the CNO cycle (W/kg).
      self.en = self.enPP+self.enCNO # Total power per unit of mass produced by the nuclear reactions (W/kg).
      self.LPP = self.luminosity(self.rho, self.enPP, self.R) # Luminosity from the proton-proton chain (W).
      self.LCNO = self.luminosity(self.rho, self.enCNO, self.R) # Luminosity from the CNO cycle (W).
      #self.L = self.LPP+self.LCNO # Total luminosity (W).
      self.mfp = 1./(self.rho*self.kappa(self.rho, self.T)) # Photon mean free path (m).
      i = np.where(self.L > .99*self.L[-1])[0][0]
      self.Rc = self.r[i]*self.R # Core radius (m).
      self.Mc = self.m[i] # Core mass (kg).

    def reset_solution(self):
      """Reset solution."""
      KramersHydrostaticMainSequenceStar.reset_solution(self)
      self.m = None
      self.g = None
      self.Pgas = None
      self.Prad = None
      self.enPP = None
      self.enCNO = None
      self.en = None
      self.LPP = None
      self.LCNO = None
      self.mfp = None
      self.Rc = None
      self.Mc = None

    def print_star_parameters(self):
      """Print star parameters."""
      print(f"Mass M = {self.M:.3e} kg.")
      print(f"Radius R = {self.R/1.e3:.0f} km.")
      print(f"Surface density rho(R) = {self.rho[-1]/1.e3:.3e} g/cm^3.")
      print(f"Central density rho(0) = {self.rho[0]/1.e3:.3e} g/cm^3.")
      print(f"Average density <rho>  = {3.*self.M/(4.e3*pi*self.R**3):.3e} g/cm^3.")
      print(f"Surface gravitational field g(R) = {self.g[-1]:.3e} m/s^2 = {self.g[-1]/cst.ge:.3f} ge.")
      print(f"Surface temperature T(R) = {self.T[-1]:.0f} K.")
      print(f"Central temperature T(0) = {self.T[0]:.3e} K.")
      print(f"Central pressure P(0) = {self.P[0]/1.e9:.3e} GPa = {self.P[0]/cst.atm/1.e9:.3f} billions atm.")
      print(f"PP  luminosity LPP  = {self.LPP[-1]:.3e} W.")
      print(f"CNO luminosity LCNO = {self.LCNO[-1]:.3e} W.")
      print(f"Total luminosity L  = {self.L[-1]:.3e} W.")
      print(f"Core radius Rc = {self.Rc/1.e3:.0f} km = {self.Rc/self.R:.3f}R (@99% total luminosity).")
      print(f"Core mass Mc = {self.Mc:.3e} kg = {self.Mc/self.M:.3f}M.")
      if any(self.convection):
        if all(self.convection):
          print("The star is fully convective.")
        else:
          if self.convection[0]:
            i = np.where(~self.convection)[0][0]
            rconv = self.r[i]
            Mconv = self.m[i]
            print(f"Convective core radius Rconv = {rconv*self.R/1.e3:.0f} km = {rconv:.3f}R.")
            print(f"Convective core mass Mconv = {Mconv:.3e} kg = {Mconv/self.M:.3f}M.")
          if self.convection[-1]:
            i = np.where(~self.convection[::-1])[0][0]
            tconv = 1.-self.r[::-1][i]
            Mconv = self.M-self.m[::-1][i]
            print(f"Surface convection zone thickness Tconv = {tconv*self.R/1.e3:.0f} km = {tconv:.3f}R.")
            print(f"Surface convection zone mass Mconv = {Mconv:.3e} kg = {Mconv/self.M:.3f}M.")

    def plot_star_parameters(self):
      """Plot star parameters."""

      def plot_convection(ax):
        """Highlight convection zone(s) in axes ax."""
        rt = 0.
        convection = self.convection[0]
        for i in range(1, self.ngrid-1):
          if self.convection[i] != convection:
            if convection: ax.axvspan(rt, self.r[i], color = "gray", lw = 0., alpha = .2)
            rt = self.r[i]
            convection = self.convection[i]
        if convection: ax.axvspan(rt, 1., color = "gray", lw = 0., alpha = .2)

      plt.figure(1) # Mass density.
      ax = plt.axes()
      plot_convection(ax)
      ax.semilogy(self.r, self.rho/1.e3, "b-")
      ax.set_xlim(0., 1.)
      ax.set_xlabel("$r/R$")
      ax.set_ylabel("$\\rho$ (g/cm$^3$)")
      if self.panels: ax.text(0.05, 0.05, "(a)", fontsize = 20, ha = "left", va = "bottom", transform = ax.transAxes)
      plt.figure(2) # Pressure.
      ax = plt.axes()
      plot_convection(ax)
      if self.prad:
        ax.semilogy(self.r, self.Pgas/cst.atm, "r--", label = "Gaz")
        ax.semilogy(self.r, self.Prad/cst.atm, "m:", label = "Radiation")
        ax.semilogy(self.r, self.P/cst.atm, "b-", label = "Total")
        plt.legend(loc = "center left")
      else:
        ax.semilogy(self.r, self.P/cst.atm, "b-")
      ax.set_xlim(0., 1.)
      ax.set_xlabel("$r/R$")
      ax.set_ylabel("$P$ (atm)")
      if self.panels: ax.text(0.05, 0.05, "(b)", fontsize = 20, ha = "left", va = "bottom", transform = ax.transAxes)
      plt.figure(3) # Temperature.
      ax = plt.axes()
      plot_convection(ax)
      ax.semilogy(self.r, self.T, "b-")
      ax.set_xlim(0., 1.)
      ax.set_xlabel("$r/R$")
      ax.set_ylabel("$T$ (K)")
      if self.panels: ax.text(0.05, 0.05, "(c)", fontsize = 20, ha = "left", va = "bottom", transform = ax.transAxes)
      plt.figure(4) # Power density.
      ax = plt.axes()
      ax.plot(self.r, self.rho*self.enPP, "r--", label = "PP")
      ax.plot(self.r, self.rho*self.enCNO, "m:", label = "CNO")
      ax.plot(self.r, self.rho*self.en, "b-", label = "Total")
      ax.set_xlim(0., 1.)
      ax.set_xlabel("$r/R$")
      ax.set_ylabel("$\\rho\\varepsilon_\mathrm{n}$ (W/m$^3$)")
      plt.legend(loc = "upper right")
      if self.panels: ax.text(0.05, 0.05, "(d)", fontsize = 20, ha = "left", va = "bottom", transform = ax.transAxes)
      plt.figure(5) # Photon mean free path.
      ax = plt.axes()
      plot_convection(ax)
      mfp = self.mfp*1.e3
      radiative = np.zeros(self.ngrid, dtype = bool)
      radiative[:-1] = radiative[:-1] | ~self.convection
      radiative[ 1:] = radiative[ 1:] | ~self.convection
      mfpr = np.where(radiative, mfp, np.NaN)
      ax.semilogy(self.r, mfpr, "b-")
      ymin, ymax = ax.get_ylim()
      ax.semilogy(self.r, mfp, "b:")
      ax.set_xlim(0., 1.)
      ax.set_ylim(ymin, ymax)
      #ax.set_ylim(1.e-2, 1.e1)
      ax.set_xlabel("$r/R$")
      ax.set_ylabel("$\ell$ (mm)")
      if self.panels: ax.text(0.05, 0.05, "(e)", fontsize = 20, ha = "left", va = "bottom", transform = ax.transAxes)
      if (np.min(self.comp.X) != np.max(self.comp.X)) or (np.min(self.comp.Z) != np.max(self.comp.Z)):
        plt.figure(6) # Composition.
        ax = plt.axes()
        ax.plot(self.r, self.to_grid(self.comp.X), "b-", label = "$X$ (H)")
        ax.plot(self.r, self.to_grid(self.comp.Y), "r--", label = "$Y$ (He)")
        ax.plot(self.r, self.to_grid(self.comp.Z), "m-.", label = "$Z$ (M)")
        ax.plot(self.r, self.to_grid(self.comp.XCNO), "g:", label = "$X_\mathrm{CNO}$")
        ax.set_xlim(0., 1.)
        ax.set_ylim(0., 1.)
        ax.set_xlabel("$r/R$")
        ax.set_ylabel("Fraction massique")
        plt.legend(loc = "center right")
      #plt.figure(7) # P**(1-gamma)*T**gamma # TEST
      #ax = plt.axes()
      #plot_convection(ax)
      #ax.plot(self.r, self.P**(1.-self.gamma)*self.T**self.gamma, "b-")
      #ax.set_xlim(0., 1.)
      #ax.set_xlabel("$r/R$")
      #ax.set_ylabel("$P^{1-\gamma}T^\gamma$ (Pa$^{1-\gamma}$.K$^\gamma$)")

    def draw_star(self):
      """Draw the star."""

      def plot_convection(ax):
        """Highlight convection zone(s) in axes ax."""
        rt = 0.
        convection = self.convection[0]
        for i in range(1, self.ngrid-1):
          if self.convection[i] != convection:
            if convection: ax.fill_between(theta, rt, self.r[i], hatch = "o", fc = "None", ec = "gray", lw = 0., alpha = .66)
            rt = self.r[i]
            convection = self.convection[i]
        if convection: ax.fill_between(theta, rt, 1., hatch = "o", fc = "None", ec = "gray", lw = 0., alpha = .66)

      plt.figure(8)
      ntheta = 65
      theta = np.linspace(0., 2.*pi, ntheta)
      T = np.broadcast_to(self.T, (ntheta, self.ngrid)).T
      ax = plt.axes(projection = "polar", rasterization_zorder = -10.)
      cf = ax.contourf(theta, self.r, T, 256, cmap = "hot", zorder = -20.)
      ax.grid(False)
      ax.set_xticks([])
      ax.set_yticks([])
      plot_convection(ax)
      cb = plt.colorbar(cf, ticks = plt.MaxNLocator(10), ax = ax)
      cb.ax.set_ylabel("$T$ (K)")
      ax.plot(theta, np.full_like(theta, self.Rc/self.R), "k--") # Core radius.
      caption = utils.flatex(f"$M={self.M:.3e}$ kg, $R={self.R/1e3:.0f}$ km, $T_\mathrm{{c}}={self.T[-1]:.0f}$ K")
      ax.text(0.5, -0.1, caption, ha = "center", va = "top", transform = ax.transAxes)
      if self.panels: ax.text(-0.1, 0.975, "(f)", fontsize = 20, ha = "left", va = "top", transform = ax.transAxes)

  return ExtendedHydrostaticMainSequenceStar

#############
### Main. ###
#############

if __name__ == "__main__":

  # The star parameters.
  M = 1.99e30     # Mass (kg).
  X = 0.7346      # Hydrogen mass fraction.
  Z = 0.0169      # Metallicity.
  XCNO = 0.0115   # C/N/O mass fraction in the core.

  inhomogeneous = True # Use inhomogeneous star composition ?
  def get_X(r):   # Inhomogeneous star composition.
    """Return the hydrogen mass fraction X as a function of the reduced radial position r."""
    return X*(1.-np.exp(-(r/0.125)**2)/2) if inhomogeneous else X

  ngrid = 2**10   # Number of grid points.
  rgrid = 1.      # Ratio of last to first grid step.
  rhos = 2.5e-2   # Photosphere mass density (kg/m^3).
  Ps = 19.45      # Surface pressure (atm).
  BC = "density"  # Set BC = "pressure" to set Ps at the surface, "density" to set rhos at the surface.
  prad = True     # Enable radiative pressure.

  epsilon = 1.e-6 # Solution accuracy. Beware that OPAL interpolates opacity from a limited set of data and can not, therefore converge to epsilon <~ 1.e-6.
  nmax = 16384    # Maximum number of self-consistent iterations.
  alphamin = 0.2  # Minimal mixing parameter (decrease if does not converge).

  if _OPAL: # The model to solve.
    StarClass = OPALHydrostaticMainSequenceStar # OPAL.
    initial_guess = True # OPAL needs a very good initial guess (obtained, e.g., with Kramers' laws for the opacity).
  else:
    StarClass = ModKramersHydrostaticMainSequenceStar # Modified Kramers' laws.
    initial_guess = False

  # Plot options.
  save_plots = False # Save plots on disk ?
  plots_dir = "./Soleil" # Plots directory

  # First get an initial guess with an other model.
  # May be needed for OPAL which can only address a limited range of densities and temperatures.
  guess = None
  if initial_guess:
    star = ModKramersHydrostaticMainSequenceStar(M, ngrid, rgrid)
    star.set_composition(get_X(star.r), Z, XCNO)
    star.enable_radiative_pressure(False)
    if BC == "pressure":
      star.set_surface_pressure(Ps*cst.atm)
    elif BC == "density":
      star.set_surface_mass_density(rhos)
    else:
      raise ValueError("Error, BC must be 'pressure' or 'density'.")
    star.solve(nmax = nmax, epsilon = epsilon, alphamin = alphamin)
    guess = star.get_solution()

  # Solve the chosen model.
  ExtStarClass = AddPlotExtensions(StarClass)
  star = ExtStarClass(M, ngrid, rgrid)
  star.set_composition(get_X(star.r), Z, XCNO)
  star.enable_radiative_pressure(prad)
  if BC == "pressure":
    star.set_surface_pressure(Ps*cst.atm)
  elif BC == "density":
    star.set_surface_mass_density(rhos)
  else:
    raise ValueError("Error, BC must be 'pressure' or 'density'.")
  star.solve(nmax = nmax, epsilon = epsilon, alphamin = alphamin, guess = guess)

  # Plot data.
  star.print_star_parameters()
  star.plot_star_parameters()
  star.draw_star()

  if save_plots:
    plt.figure(1)
    plt.savefig(plots_dir+"/rho.pdf")
    plt.figure(2)
    plt.savefig(plots_dir+"/P.pdf")
    plt.figure(3)
    plt.savefig(plots_dir+"/T.pdf")
    plt.figure(4)
    plt.savefig(plots_dir+"/epsilon.pdf")
    plt.figure(5)
    plt.savefig(plots_dir+"/lpm.pdf")
    plt.figure(8)
    plt.savefig(plots_dir+"/etoile.pdf")

  plt.ioff()
  plt.show()
