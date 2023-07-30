#!/usr/bin/python

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (ymniquet@gmail.com).
# Version: 2023.05

from pylab import *
from scipy.integrate import quad
import constants as cst

"""Plot radius vs mass for a degenerate star."""

######################################################################
# Equation of state for a degenerate, non-relativistic electron gas. #
######################################################################

def Pdeg_nonrel(n):
  """Return pressure (Pa) as a function of carrier density n (m^-3) for a degenerate, non-relativistic electron gas."""
  return (cst.hbar**2/(5.*cst.me))*(3.*pi**2)**(2./3.)*n**(5./3.)
  
def dPdegdn_nonrel(n):
  """Return the derivative of the pressure (Pa.m^3) with respect to the carrier density n (m^-3) for a degenerate, non-relativistic electron gas."""
  return (cst.hbar**2/(3.*cst.me))*(3.*pi**2*n)**(2./3.)

##################################################################
# Equation of state for a degenerate, relativistic electron gas. #
##################################################################

def _Pdeg(n):
  """Return pressure (Pa) as a function of carrier density n (m^-3) for a degenerate relativistic electron gas."""
  kf = (3.*pi**2*n)**(1./3.)
  x = cst.hbar*kf/(cst.me*cst.c)
  I = quad(lambda t: t**4/sqrt(1.+t**2), 0., x)
  return cst.me**4*cst.c**5*I[0]/(3.*pi**2*cst.hbar**3)
  
Pdeg = vectorize(lambda n: _Pdeg(n), otypes = [float])

def dPdegdn(n):
  """Return the derivative of the pressure (Pa.m^3) with respect to the carrier density n (m^-3) for a degenerate relativistic electron gas."""
  kf = (3.*pi**2*n)**(1./3.)
  x = cst.hbar*kf/(cst.me*cst.c)
  I = (x**4/sqrt(1.+x**2))*(cst.hbar/(cst.me*cst.c))*(3.*pi**2)**(1./3.)*n**(-2./3.)/3.
  return cst.me**4*cst.c**5*I/(3.*pi**2*cst.hbar**3)

########################################################################
# Equation of state for a degenerate, ultra-relativistic electron gas. #
########################################################################

def Pdeg_ultra(n):
  """Return pressure (Pa) as a function of carrier density n (m^-3) for a degenerate, ultra-relativistic electron gas."""
  return (cst.hbar*cst.c/4.)*(3.*pi**2)**(1./3.)*n**(4./3.)
  
def dPdegdn_ultra(n):
  """Return the derivative of the pressure (Pa.m^3) with respect to the carrier density n (m^-3) for a degenerate, ultra-relativistic electron gas."""
  return (cst.hbar*cst.c/3.)*(3.*pi**2*n)**(1./3.)

######################
# Lame-Enden solver. #
######################

def radius_degenerate_star(M, dPdn, mue = 2., ngrid = 2**12, epsilon = 1.e-8, nmax = 1000000, alpha = .5, guess = None):
  """Compute radius R of a degenerate star with mass M (kg) by solving the hydrostatic equilibrium equations
     (self-consistent method derived from etoile.py). dPdn(n) is a function that returns the derivative
     of the pressure (Pa.m^3) with respect to the electron density n (m^-3). mu is the molecular mass per 
     electron, ngrid the number of points on the radial grid, epsilon the target accuracy of the solution, 
     nmax the maximal number of self-consistent iterations, and alpha the mixing parameter (decrease alpha 
     if not converging). If not None, guess = (R, rho) is an initial guess for the radius R (m) and the 
     mass density rho (kg/m^3) on the grid. Returns the radius R (m), the mass density rho on the grid (kg/m^3)
     and an error code ierr (= 0 if solution found, 1 if convergence is not achieved)."""
  
  # Set-up radial grid.
  r = linspace(0., 1., ngrid)
  dr = r[1:]-r[:-1] # Grid steps.
  rc = (r[:-1]+r[1:])/2. # Mid-points.
  dVl = empty(ngrid) # Volume of the half-shell on the left of each grid point.    
  dVl[1:] = 4.*pi*(r[1:]**3-rc**3)/3. 
  dVl[ 0] = 0.
  dVr = empty(ngrid) # Volume of the half-shell on the right of each grid point.    
  dVr[:-1] = 4.*pi*(rc**3-r[:-1]**3)/3. 
  dVr[ -1] = 0.   
  dVc = dVl+dVr # Volume of the shell centered on each grid point.
  # Solve the hydrostatic equilibrium equations.
  m = mue*cst.mp # Mass per electron.  
  n = 0
  ierr = 0 # Return code.
  print( "+-------+"+2*(14*"-"+"+"))
  print(f"| Iter. | {'Error':>12} | {'R (km)':>12} |")
  print( "+-------+"+2*(14*"-"+"+")) 
  # Initial guess.
  if guess is not None:
    R, rho = guess
  else:
    R = 1.e6
    rho = full(ngrid, 3.*M/(4.*pi*R**3))  
  print(f"| {n:5d} | {NaN:12.5e} | {R/1.e3:12.5e} |")    
  while True: 
    # New self-consistent iteration.
    n += 1 
    if n > nmax:
      print( "+-------+"+2*(14*"-"+"+"))
      print(f"Error, could not converge within {nmax} iterations.")
      ierr = 1
      break         
    # Save present density.
    rhop = copy(rho)    
    # Compute gravitational field.
    g = cumsum(rho*dVc)-rho*dVr
    g[1:] /= r[1:]**2
    g *= cst.G*R     
    # Solve density equation.
    dPdr = -rho*g # Compute dP/dr.
    drhodr = m*dPdr/dPdn(rho/m) # Compute drho/dr.
    drhodr = (drhodr[:-1]+drhodr[1:])/2. # Average drho/dr at mid-points.      
    drho = -cumsum(drhodr[::-1]*dr[::-1])*R # Integrate drho/dr.
    rho[-1] = 1.e-12 # Do not set to 0 to avoid divergences.
    rho[:-1] = rho[-1]+drho[::-1]
    # Compute error.
    error = sqrt(sum(((rho-rhop)/rhop[0])**2))
    print(f"| {n:5d} | {error:12.5e} | {R/1.e3:12.5e} |")            
    # Check convergence.
    if error < epsilon:
      print( "+-------+"+2*(14*"-"+"+"))
      print(f"Converged within {n} iterations.")
      break
    # Mix previous & present densities.
    rho = alpha*rho+(1.-alpha)*rhop
    # Compute total mass.
    Mt = sum(rho*dVc)*R**3
    # Update star radius.
    R = R*(M/Mt)**(1./3.)        
  return R, rho, ierr

#########  
# Main. #
#########

# Plot pressure as a function of density in a degenerate gas.

figure(1)
ax = axes()
n = 10.**linspace(6., 60., 256)
loglog(n/1.e6, Pdeg(n), "b-")
plot(n/1.e6, Pdeg_nonrel(n), "b:", label = "Non relativiste")
plot(n/1.e6, Pdeg_ultra(n), "b--", label = "Ultra-relativiste")
xlim(1.e10, 1.e52)
ylim(1.e-15, 1.e60)
xlabel("$n$ (cm$^{-3}$)")
ylabel("$P$ (Pa)")
legend(loc = "upper left")
axvline(1.25e30, linestyle = "-.", color = "gray")
#ax.text(0.975, 0.05, "(a)", fontsize = 20, ha = "right", va = "bottom", transform = ax.transAxes)

savefig("degenere_relativiste.pdf")

#show()

# Plot radius as a function of mass in a degenerate star.

Msun = 1.99e30 # Solar mass.

figure(2)
ax = axes()

# Non-relativistic case.

Mmin = 0.25 # Minimum mass (in units of Msun).
Mmax = 1.75 # Maximum mass (in units of Msun).
Ms = linspace(Mmin, Mmax, 256)
Rs = []
guess = None
for M in Ms:
  R, rho, ierr = radius_degenerate_star(M*Msun, dPdegdn_nonrel, guess = guess)
  if ierr != 0: 
    raise RuntimeError("Error, failed to solve the radius.")
  Rs.append(R)
  guess = (R, rho)
Rs = array(Rs)

plot(Ms, Rs/1.e3, "b:", label = "Non relativiste")

# Relativistic case.
 
Mmin = 0.25 # Minimum mass (in units of Msun).
Mmax = 1.434 # Maximum mass (in units of Msun).
Ms = linspace(Mmin, Mmax, 256)
Rs = []
guess = None
for M in Ms:
  R, rho, ierr = radius_degenerate_star(M*Msun, dPdegdn, guess = guess)
  if ierr != 0: 
    raise RuntimeError("Error, failed to solve the radius.")
  Rs.append(R)
  guess = (R, rho)
Rs = array(Rs)

plot(Ms, Rs/1.e3, "b-", label = "Relativiste")

# Finalize plot.
 
xlim(0.25, 1.75)
ylim(0., 14000.) 
xlabel("$M$ ($M_\odot$)")
ylabel("$R$ (km)")
legend(loc = "lower left")
axvline(1.44, linestyle = "-.", color = "gray")
#ax.text(0.975, 0.95, "(b)", fontsize = 20, ha = "right", va = "top", transform = ax.transAxes)

savefig("Chandrasekhar.pdf")

# Plot density in the star.

figure(3)
ax = axes()
plot(linspace(0., 1., len(rho)), rho/1.e3, "b-")
xlim(0., 1.)
xlabel("$r/R$")
ylabel("$\\rho$ (g/cm$^3$)")
title(f"$M={Ms[-1]:.3f}$ $M_\odot$")
#ax.text(0.975, 0.95, "(c)", fontsize = 20, ha = "right", va = "top", transform = ax.transAxes)

show()
