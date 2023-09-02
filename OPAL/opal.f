C     Python interface to the Fortran OPAL interpolation code.
C     Compile with f2py3 -c -m opal opal.f --fcompiler=gfortran --f77flags="-cpp -O3"
C
C     This program is free software: you can redistribute it and/or modify it under the terms
C     of the GNU General Public License as published by the Free Software Foundation, either
C     version 3 of the License, or (at your option) any later version.
C     This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
C     without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
C     See the GNU General Public License for more details.
C     You should have received a copy of the GNU General Public License along with this program.
C     If not, see <https://www.gnu.org/licenses/>.
C     Author: Yann-Michel Niquet (contact@ymniquet.fr).
C     Version: 2023.09

C     Return OPAL Rosseland average opacity kappa (m^2/kg).
C     rho is the mass density (kg/m^3) and T the temperature (K).
C     X is the mass fraction of hydrogen, and Z the mass fraction of metals.

      real*8 function kappa(rho, T, X, Z)

      real*8 rho, T, X, Z
      real*8 T6, R

C     Legacy single precision data.
      real opact, dopact, dopacr, dopactd

      common /e/opact, dopact, dopacr, dopactd

      T6 = T/1.d6
      R = rho/1.d3/T6**3

C     Make sure that the input data are in the OPAL opacity tables.
C     The consistency of the solution can be checked with the check_kappa_bounds function below.

      T6 = min(T6, 10.d0**(8.70d0-6.d0))
      T6 = max(T6, 10.d0**(3.75d0-6.d0))

      R = min(R, 10.d0**( 1.d0))
      R = max(R, 10.d0**(-8.d0))

      call OPACGN93(real(Z), real(X), real(T6), real(R))

      kappa = (10.d0**opact)/10.d0

      return

      end

C     Check whether the inputs are in the OPAL opacity tables.
C     rho is the mass density (kg/m^3) and T the temperature (K).
C     X is the mass fraction of hydrogen, and Z the mass fraction of metals.

      logical function check_kappa_bounds(rho, T, X, Z)

      real*8 rho, T, X, Z
      real*8 inpT6, T6, inpR, R

      inpT6 = T/1.d6
      inpR = rho/1.d3/inpT6**3

C     Check whether the input data are in the OPAL tables.

      T6 = inpT6
      T6 = min(T6, 10.d0**(8.70d0-6.d0))
      T6 = max(T6, 10.d0**(3.75d0-6.d0))

      R = inpR
      R = min(R, 10.d0**( 1.d0))
      R = max(R, 10.d0**(-8.d0))

      check_kappa_bounds = ((T6.eq.inpT6).and.(R.eq.inpR))

      return

      end

#include "xztrin21.f"
