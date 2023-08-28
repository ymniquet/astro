#!/usr/bin/python

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (ymniquet@gmail.com).
# Version: 2023.09

from pylab import *

"""Plot nuclear fusion reaction rates."""

###############
# Parameters. #
###############

X = 0.7346 # Hydrogen mass fraction.
XCNO = 0.0115 # CNO mass fraction.
rho = 150.e3 # Density (kg/m^3).

###################

### Full models.

def epsilon_PP(rho, T):
  """Returns the power per unit of mass epsilon (W/kg) for the proton-proton chain.
     rho is the mass density (kg/m^3) and T the temperature (K).
     Full model."""
  one3 = 1./3.
  two3 = 2./3.
  T6 = T/1.e6
  rho3 = rho/1.e3
  fPP = 1.+0.133*X**2*np.sqrt((3.+X)*rho3)/T6**1.5
  psiPP = 1.+1.412e8*(1./X-1.)*np.exp(-49.98*T6**(-one3))
  CPP = 1.+0.0123*T6**one3+0.0109*T6**two3+0.000938*T6
  return 1.e-4*2.38e6*rho3*X**2*fPP*psiPP*CPP*T6**(-two3)*np.exp(-33.8*T6**(-one3))

def epsilon_CNO(rho, T):
  """Returns the power per unit of mass epsilon (W/kg) for the CNO cycle.
     rho is the mass density (kg/m^3) and T the temperature (K).
     Full model."""      
  one3 = 1./3.
  two3 = 2./3.      
  T6 = T/1.e6  
  rho3 = rho/1.e3    
  CCNO = 1.+0.0027*T6**one3-0.00778*T6**two3-0.000149*T6
  return 1.e-4*8.67e27*rho3*X*XCNO*CCNO*T6**(-two3)*np.exp(-152.28*T6**(-one3))

### Polynomial approximations.

def epsilon_PP_poly(rho, T):
  """Returns the power per unit of mass epsilon (W/kg) for the proton-proton chain.
     rho is the mass density (kg/m^3) and T the temperature (K).
     Polynomial approximation."""
  return 1.07e-36*X**2*rho*T**4

def epsilon_CNO_poly(rho, T):
  """Returns the power per unit of mass epsilon (W/kg) for the CNO cycle.
     rho is the mass density (kg/m^3) and T the temperature (K).
     Polynomial approximation."""      
  return 6.54e-151*X*XCNO*rho*T**20 

###################

### Plot power per unit of mass produced by nuclear reactions in different approximations.

T = np.linspace(5.e6, 25.e6, 50)
semilogy(T/1.e6, epsilon_PP(rho, T), "bs-", label = "PP")
semilogy(T/1.e6, epsilon_CNO(rho, T), "rd-", label = "CNO")
semilogy(T/1.e6, epsilon_PP(rho, T)+epsilon_CNO(rho, T), "ko-", label = "Total")
semilogy(T/1.e6, epsilon_PP_poly(rho, T), "b:")
semilogy(T/1.e6, epsilon_CNO_poly(rho, T), "r:")
semilogy(T/1.e6, epsilon_PP_poly(rho, T)+epsilon_CNO_poly(rho, T), "k:")
xlim(5., 25.)
ylim(1.e-5, 1.e1)
xlabel("$T$ ($\\times 10^6$ K)")
ylabel("$\\varepsilon_\mathrm{n}$ (W/kg)")
title("$X=0.7346$, $X_\mathrm{CNO}=0.0115$, $\\rho=150\\times 10^3$\,kg/m$^3$", fontsize = 16)
legend(loc = "upper left")
axvline(15., linestyle = "-.", color = "gray")

savefig("PPCNO.pdf")

show()
  
