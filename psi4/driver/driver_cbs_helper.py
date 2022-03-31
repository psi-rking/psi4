#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2022 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

import math

import numpy as np

from psi4 import core
from psi4.driver.p4util.exceptions import *

_zeta_val2sym = {k + 2: v for k, v in enumerate('dtq5678')}


def xtpl_highest_1(functionname: str, zHI: int, valueHI: float, verbose: bool = True, **kwargs):
    r"""Scheme for total or correlation energies with a single basis or the highest
    zeta-level among an array of bases. Used by :py:func:`~psi4.cbs`.

    Parameters
    ----------
    functionname
        Name of the CBS component.
    zHI
        Zeta-level, only used for printing.
    valueHI
        Value of the CBS component.

    Returns
    -------
    float
        Returns :math:`E_{total}^{\infty}` which is equal to valueHI.

    Notes
    -----
    .. math:: E_{total}^X = E_{total}^{\infty}

    """
    if isinstance(valueHI, float):

        if verbose:
            # Output string with extrapolation parameters
            cbsscheme = ''
            cbsscheme += """\n   ==> %s <==\n\n""" % (functionname.upper())
            cbsscheme += """   HI-zeta (%s) Energy:               % 16.12f\n""" % (str(zHI), valueHI)

            core.print_out(cbsscheme)

        return valueHI

    elif isinstance(valueHI, (core.Matrix, core.Vector)):

        if verbose > 2:
            core.print_out("""   HI-zeta (%s) Total Energy:\n""" % (str(zHI)))
            valueHI.print_out()

        return valueHI


def scf_xtpl_helgaker_2(functionname: str, zLO: int, valueLO: float, zHI: int, valueHI: float, verbose: bool = True, alpha: float = None):
    r"""Extrapolation scheme using exponential form for reference energies with two adjacent
    zeta-level bases. Used by :py:func:`~psi4.cbs`.

    Parameters
    ----------
    functionname
        Name of the CBS component.
    zLO
        Lower zeta level.
    valueLO
        Lower value used for extrapolation.
    zHI
        Higher zeta level. Should be equal to zLO + 1.
    valueHI
        Higher value used for extrapolation.
    alpha
        Overrides the default :math:`\alpha = 1.63`

    Returns
    -------
    float
        Returns :math:`E_{total}^{\infty}`, see below.

    Notes
    -----
    The extrapolation is calculated according to [1]_:
    :math:`E_{total}^X = E_{total}^{\infty} + \beta e^{-\alpha X}, \alpha = 1.63`

    References
    ----------

    .. [1] Halkier, Helgaker, Jorgensen, Klopper, & Olsen, Chem. Phys. Lett. 302 (1999) 437-446,
       DOI: 10.1016/S0009-2614(99)00179-7

    """

    if type(valueLO) != type(valueHI):
        raise ValidationError(
            "scf_xtpl_helgaker_2: Inputs must be of the same datatype! (%s, %s)" % (type(valueLO), type(valueHI)))

    if alpha is None:
        alpha = 1.63

    beta_division = 1 / (math.exp(-1 * alpha * zLO) * (math.exp(-1 * alpha) - 1))
    beta_mult = math.exp(-1 * alpha * zHI)

    if isinstance(valueLO, float):
        beta = (valueHI - valueLO) * beta_division
        value = valueHI - beta * beta_mult

        if verbose:
            # Output string with extrapolation parameters
            cbsscheme = ''
            cbsscheme += """\n   ==> Helgaker 2-point exponential SCF extrapolation for method: %s <==\n\n""" % (
                functionname.upper())
            cbsscheme += """   LO-zeta (%s) Energy:               % 16.12f\n""" % (str(zLO), valueLO)
            cbsscheme += """   HI-zeta (%s) Energy:               % 16.12f\n""" % (str(zHI), valueHI)
            cbsscheme += """   Alpha (exponent) Value:           % 16.12f\n""" % (alpha)
            cbsscheme += """   Beta (coefficient) Value:         % 16.12f\n\n""" % (beta)

            name_str = "%s/(%s,%s)" % (functionname.upper(), _zeta_val2sym[zLO].upper(), _zeta_val2sym[zHI].upper())
            cbsscheme += """   @Extrapolated """
            cbsscheme += name_str + ':'
            cbsscheme += " " * (18 - len(name_str))
            cbsscheme += """% 16.12f\n\n""" % value
            core.print_out(cbsscheme)

        return value

    elif isinstance(valueLO, (core.Matrix, core.Vector)):
        beta = valueHI.clone()
        beta.name = 'Helgaker SCF (%s, %s) beta' % (zLO, zHI)
        beta.subtract(valueLO)
        beta.scale(beta_division)
        beta.scale(beta_mult)

        value = valueHI.clone()
        value.subtract(beta)
        value.name = 'Helgaker SCF (%s, %s) data' % (zLO, zHI)

        if verbose > 2:
            core.print_out("""\n   ==> Helgaker 2-point exponential SCF extrapolation for method: %s <==\n\n""" %
                           (functionname.upper()))
            core.print_out("""   LO-zeta (%s)""" % str(zLO))
            core.print_out("""   LO-zeta Data""")
            valueLO.print_out()
            core.print_out("""   HI-zeta (%s)""" % str(zHI))
            core.print_out("""   HI-zeta Data""")
            valueHI.print_out()
            core.print_out("""   Extrapolated Data:\n""")
            value.print_out()
            core.print_out("""   Alpha (exponent) Value:          %16.8f\n""" % (alpha))
            core.print_out("""   Beta Data:\n""")
            beta.print_out()

        return value

    else:
        raise ValidationError("scf_xtpl_helgaker_2: datatype is not recognized '%s'." % type(valueLO))


def scf_xtpl_truhlar_2(functionname: str, zLO: int, valueLO: float, zHI: int, valueHI: float, verbose: bool = True, alpha: float = None):
    r"""Extrapolation scheme using power form for reference energies with two adjacent
    zeta-level bases. Used by :py:func:`~psi4.cbs`.

    Parameters
    ----------
    functionname
        Name of the CBS component.
    zLO
        Lower zeta level.
    valueLO
        Lower value used for extrapolation.
    zHI
        Higher zeta level. Should be equal to zLO + 1.
    valueHI
        Higher value used for extrapolation.
    alpha
        Overrides the default :math:`\alpha = 3.4`

    Returns
    -------
    float
        Returns :math:`E_{total}^{\infty}`, see below.

    Notes
    -----
    The extrapolation is calculated according to [2]_:
    :math:`E_{total}^X = E_{total}^{\infty} + \beta X^{-\alpha}, \alpha = 3.4`

    References
    ----------

    .. [2] Truhlar, Chem. Phys. Lett. 294 (1998) 45-48,
       DOI: 10.1016/S0009-2614(98)00866-5

    """

    if type(valueLO) != type(valueHI):
        raise ValidationError(
            "scf_xtpl_truhlar_2: Inputs must be of the same datatype! (%s, %s)" % (type(valueLO), type(valueHI)))

    if alpha is None:
        alpha = 3.40

    beta_division = 1 / (zHI**(-1 * alpha) - zLO**(-1 * alpha))
    beta_mult = zHI**(-1 * alpha)

    if isinstance(valueLO, float):
        beta = (valueHI - valueLO) * beta_division
        value = valueHI - beta * beta_mult

        if verbose:
            # Output string with extrapolation parameters
            cbsscheme = ''
            cbsscheme += """\n   ==> Truhlar 2-point power form SCF extrapolation for method: %s <==\n\n""" % (
                functionname.upper())
            cbsscheme += """   LO-zeta (%s) Energy:               % 16.12f\n""" % (str(zLO), valueLO)
            cbsscheme += """   HI-zeta (%s) Energy:               % 16.12f\n""" % (str(zHI), valueHI)
            cbsscheme += """   Alpha (exponent) Value:           % 16.12f\n""" % (alpha)
            cbsscheme += """   Beta (coefficient) Value:         % 16.12f\n\n""" % (beta)

            name_str = "%s/(%s,%s)" % (functionname.upper(), _zeta_val2sym[zLO].upper(), _zeta_val2sym[zHI].upper())
            cbsscheme += """   @Extrapolated """
            cbsscheme += name_str + ':'
            cbsscheme += " " * (18 - len(name_str))
            cbsscheme += """% 16.12f\n\n""" % value
            core.print_out(cbsscheme)

        return value

    elif isinstance(valueLO, (core.Matrix, core.Vector)):
        beta = valueHI.clone()
        beta.name = 'Truhlar SCF (%s, %s) beta' % (zLO, zHI)
        beta.subtract(valueLO)
        beta.scale(beta_division)
        beta.scale(beta_mult)

        value = valueHI.clone()
        value.subtract(beta)
        value.name = 'Truhlar SCF (%s, %s) data' % (zLO, zHI)

        if verbose > 2:
            core.print_out("""\n   ==> Truhlar 2-point power from SCF extrapolation for method: %s <==\n\n""" %
                           (functionname.upper()))
            core.print_out("""   LO-zeta (%s)""" % str(zLO))
            core.print_out("""   LO-zeta Data""")
            valueLO.print_out()
            core.print_out("""   HI-zeta (%s)""" % str(zHI))
            core.print_out("""   HI-zeta Data""")
            valueHI.print_out()
            core.print_out("""   Extrapolated Data:\n""")
            value.print_out()
            core.print_out("""   Alpha (exponent) Value:          %16.8f\n""" % (alpha))
            core.print_out("""   Beta Data:\n""")
            beta.print_out()

        return value

    else:
        raise ValidationError("scf_xtpl_truhlar_2: datatype is not recognized '%s'." % type(valueLO))


def scf_xtpl_karton_2(functionname: str, zLO: int, valueLO: float, zHI: int, valueHI: float, verbose: bool = True, alpha: float = None):
    r"""Extrapolation scheme using root-power form for reference energies with two adjacent
    zeta-level bases. Used by :py:func:`~psi4.cbs`.

    Parameters
    ----------
    functionname
        Name of the CBS component.
    zLO
        Lower zeta level.
    valueLO
        Lower value used for extrapolation.
    zHI
        Higher zeta level. Should be equal to zLO + 1.
    valueHI
        Higher value used for extrapolation.
    alpha
        Overrides the default :math:`\alpha = 6.3`

    Returns
    -------
    float
        Returns :math:`E_{total}^{\infty}`, see below.

    Notes
    -----
    The extrapolation is calculated according to [3]_:
    :math:`E_{total}^X = E_{total}^{\infty} + \beta e^{-\alpha\sqrt{X}}, \alpha = 6.3`

    References
    ----------

    .. [3] Karton, Martin, Theor. Chem. Acc. 115 (2006) 330-333,
       DOI: 10.1007/s00214-005-0028-6

    """

    if type(valueLO) != type(valueHI):
        raise ValidationError(
            "scf_xtpl_karton_2: Inputs must be of the same datatype! (%s, %s)" % (type(valueLO), type(valueHI)))

    if alpha is None:
        alpha = 6.30

    beta_division = 1 / (math.exp(-1 * alpha) * (math.exp(math.sqrt(zHI)) - math.exp(math.sqrt(zLO))))
    beta_mult = math.exp(-1 * alpha * math.sqrt(zHI))

    if isinstance(valueLO, float):
        beta = (valueHI - valueLO) * beta_division
        value = valueHI - beta * beta_mult

        if verbose:
            # Output string with extrapolation parameters
            cbsscheme = ''
            cbsscheme += """\n   ==> Karton 2-point power form SCF extrapolation for method: %s <==\n\n""" % (
                functionname.upper())
            cbsscheme += """   LO-zeta (%s) Energy:               % 16.12f\n""" % (str(zLO), valueLO)
            cbsscheme += """   HI-zeta (%s) Energy:               % 16.12f\n""" % (str(zHI), valueHI)
            cbsscheme += """   Alpha (exponent) Value:           % 16.12f\n""" % (alpha)
            cbsscheme += """   Beta (coefficient) Value:         % 16.12f\n\n""" % (beta)

            name_str = "%s/(%s,%s)" % (functionname.upper(), _zeta_val2sym[zLO].upper(), _zeta_val2sym[zHI].upper())
            cbsscheme += """   @Extrapolated """
            cbsscheme += name_str + ':'
            cbsscheme += " " * (18 - len(name_str))
            cbsscheme += """% 16.12f\n\n""" % value
            core.print_out(cbsscheme)

        return value

    elif isinstance(valueLO, (core.Matrix, core.Vector)):
        beta = valueHI.clone()
        beta.name = 'Karton SCF (%s, %s) beta' % (zLO, zHI)
        beta.subtract(valueLO)
        beta.scale(beta_division)
        beta.scale(beta_mult)

        value = valueHI.clone()
        value.subtract(beta)
        value.name = 'Karton SCF (%s, %s) data' % (zLO, zHI)

        if verbose > 2:
            core.print_out("""\n   ==> Karton 2-point power from SCF extrapolation for method: %s <==\n\n""" %
                           (functionname.upper()))
            core.print_out("""   LO-zeta (%s)""" % str(zLO))
            core.print_out("""   LO-zeta Data""")
            valueLO.print_out()
            core.print_out("""   HI-zeta (%s)""" % str(zHI))
            core.print_out("""   HI-zeta Data""")
            valueHI.print_out()
            core.print_out("""   Extrapolated Data:\n""")
            value.print_out()
            core.print_out("""   Alpha (exponent) Value:          %16.8f\n""" % (alpha))
            core.print_out("""   Beta Data:\n""")
            beta.print_out()

        return value

    else:
        raise ValidationError("scf_xtpl_Karton_2: datatype is not recognized '%s'." % type(valueLO))


def scf_xtpl_helgaker_3(functionname: str, zLO: int, valueLO: float, zMD: int, valueMD: float, zHI: int, valueHI: float, verbose: bool = True, alpha: float = None):
    r"""Extrapolation scheme for reference energies with three adjacent zeta-level bases.
    Used by :py:func:`~psi4.cbs`.

    Parameters
    ----------
    functionname
        Name of the CBS component.
    zLO
        Lower zeta level.
    valueLO
        Lower value used for extrapolation.
    zMD
        Intermediate zeta level. Should be equal to zLO + 1.
    valueMD
        Intermediate value used for extrapolation.
    zHI
        Higher zeta level. Should be equal to zLO + 2.
    valueHI
        Higher value used for extrapolation.
    alpha
        Not used.

    Returns
    -------
    float
        Returns :math:`E_{total}^{\infty}`, see below.

    Notes
    -----
    The extrapolation is calculated according to [4]_:
    :math:`E_{total}^X = E_{total}^{\infty} + \beta e^{-\alpha X}, \alpha = 3.0`

    References
    ----------

    .. [4] Halkier, Helgaker, Jorgensen, Klopper, & Olsen, Chem. Phys. Lett. 302 (1999) 437-446,
       DOI: 10.1016/S0009-2614(99)00179-7

    """

    if (type(valueLO) != type(valueMD)) or (type(valueMD) != type(valueHI)):
        raise ValidationError("scf_xtpl_helgaker_3: Inputs must be of the same datatype! (%s, %s, %s)" %
                              (type(valueLO), type(valueMD), type(valueHI)))

    if isinstance(valueLO, float):

        ratio = (valueHI - valueMD) / (valueMD - valueLO)
        alpha = -1 * math.log(ratio)
        beta = (valueHI - valueMD) / (math.exp(-1 * alpha * zMD) * (ratio - 1))
        value = valueHI - beta * math.exp(-1 * alpha * zHI)

        if verbose:
            # Output string with extrapolation parameters
            cbsscheme = ''
            cbsscheme += """\n   ==> Helgaker 3-point SCF extrapolation for method: %s <==\n\n""" % (
                functionname.upper())
            cbsscheme += """   LO-zeta (%s) Energy:               % 16.12f\n""" % (str(zLO), valueLO)
            cbsscheme += """   MD-zeta (%s) Energy:               % 16.12f\n""" % (str(zMD), valueMD)
            cbsscheme += """   HI-zeta (%s) Energy:               % 16.12f\n""" % (str(zHI), valueHI)
            cbsscheme += """   Alpha (exponent) Value:           % 16.12f\n""" % (alpha)
            cbsscheme += """   Beta (coefficient) Value:         % 16.12f\n\n""" % (beta)

            name_str = "%s/(%s,%s,%s)" % (functionname.upper(), _zeta_val2sym[zLO].upper(), _zeta_val2sym[zMD].upper(),
                                          _zeta_val2sym[zHI].upper())
            cbsscheme += """   @Extrapolated """
            cbsscheme += name_str + ':'
            cbsscheme += " " * (18 - len(name_str))
            cbsscheme += """% 16.12f\n\n""" % value
            core.print_out(cbsscheme)

        return value

    elif isinstance(valueLO, (core.Matrix, core.Vector)):
        valueLO = np.array(valueLO)
        valueMD = np.array(valueMD)
        valueHI = np.array(valueHI)

        nonzero_mask = np.abs(valueHI) > 1.e-14
        top = (valueHI - valueMD)[nonzero_mask]
        bot = (valueMD - valueLO)[nonzero_mask]

        ratio = top / bot
        alpha = -1 * np.log(np.abs(ratio))
        beta = top / (np.exp(-1 * alpha * zMD) * (ratio - 1))
        np_value = valueHI.copy()
        np_value[nonzero_mask] -= beta * np.exp(-1 * alpha * zHI)
        np_value[~nonzero_mask] = 0.0

        # Build and set from numpy routines
        value = core.Matrix(*valueHI.shape)
        value_view = np.asarray(value)
        value_view[:] = np_value
        return value

    else:
        raise ValidationError("scf_xtpl_helgaker_3: datatype is not recognized '%s'." % type(valueLO))


#def corl_xtpl_helgaker_2(functionname, valueSCF, zLO, valueLO, zHI, valueHI, verbose=True):
def corl_xtpl_helgaker_2(functionname: str, zLO: int, valueLO: float, zHI: int, valueHI: float, verbose: bool = True, alpha: float = None):
    r"""Extrapolation scheme for correlation energies with two adjacent zeta-level bases.
    Used by :py:func:`~psi4.cbs`.

    Parameters
    ----------
    functionname
        Name of the CBS component.
    zLO
        Lower zeta level.
    valueLO
        Lower value used for extrapolation.
    zHI
        Higher zeta level. Should be equal to zLO + 1.
    valueHI
        Higher value used for extrapolation.
    alpha
        Overrides the default :math:`\alpha = 3.0`

    Returns
    -------
    float
        Returns :math:`E_{total}^{\infty}`, see below.

    Notes
    -----
    The extrapolation is calculated according to [5]_:
    :math:`E_{corl}^X = E_{corl}^{\infty} + \beta X^{-alpha}`

    References
    ----------

    .. [5] Halkier, Helgaker, Jorgensen, Klopper, Koch, Olsen, & Wilson,
       Chem. Phys. Lett. 286 (1998) 243-252,
       DOI: 10.1016/S0009-2614(99)00179-7

    """
    if type(valueLO) != type(valueHI):
        raise ValidationError(
            "corl_xtpl_helgaker_2: Inputs must be of the same datatype! (%s, %s)" % (type(valueLO), type(valueHI)))

    if alpha is None:
        alpha = 3.0

    if isinstance(valueLO, float):
        value = (valueHI * zHI**alpha - valueLO * zLO**alpha) / (zHI**alpha - zLO**alpha)
        beta = (valueHI - valueLO) / (zHI**(-alpha) - zLO**(-alpha))

        #        final = valueSCF + value
        final = value
        if verbose:
            # Output string with extrapolation parameters
            cbsscheme = """\n\n   ==> Helgaker 2-point correlated extrapolation for method: %s <==\n\n""" % (
                functionname.upper())
            #            cbsscheme += """   HI-zeta (%1s) SCF Energy:           % 16.12f\n""" % (str(zHI), valueSCF)
            cbsscheme += """   LO-zeta (%s) Energy:               % 16.12f\n""" % (str(zLO), valueLO)
            cbsscheme += """   HI-zeta (%s) Energy:               % 16.12f\n""" % (str(zHI), valueHI)
            cbsscheme += """   Alpha (exponent) Value:           % 16.12f\n""" % alpha
            cbsscheme += """   Extrapolated Energy:              % 16.12f\n\n""" % value
            #cbsscheme += """   LO-zeta (%s) Correlation Energy:   % 16.12f\n""" % (str(zLO), valueLO)
            #cbsscheme += """   HI-zeta (%s) Correlation Energy:   % 16.12f\n""" % (str(zHI), valueHI)
            #cbsscheme += """   Beta (coefficient) Value:         % 16.12f\n""" % beta
            #cbsscheme += """   Extrapolated Correlation Energy:  % 16.12f\n\n""" % value

            name_str = "%s/(%s,%s)" % (functionname.upper(), _zeta_val2sym[zLO].upper(), _zeta_val2sym[zHI].upper())
            cbsscheme += """   @Extrapolated """
            cbsscheme += name_str + ':'
            cbsscheme += " " * (19 - len(name_str))
            cbsscheme += """% 16.12f\n\n""" % final
            core.print_out(cbsscheme)

        return final

    elif isinstance(valueLO, (core.Matrix, core.Vector)):

        beta = valueHI.clone()
        beta.subtract(valueLO)
        beta.scale(1 / (zHI**(-alpha) - zLO**(-alpha)))
        beta.name = 'Helgaker Corl (%s, %s) beta' % (zLO, zHI)

        value = valueHI.clone()
        value.scale(zHI**alpha)

        tmp = valueLO.clone()
        tmp.scale(zLO**alpha)
        value.subtract(tmp)

        value.scale(1 / (zHI**alpha - zLO**alpha))
        value.name = 'Helgaker Corr (%s, %s) data' % (zLO, zHI)

        if verbose > 2:
            core.print_out("""\n   ==> Helgaker 2-point correlated extrapolation for """
                           """method: %s <==\n\n""" % (functionname.upper()))
            core.print_out("""   LO-zeta (%s) Data\n""" % (str(zLO)))
            valueLO.print_out()
            core.print_out("""   HI-zeta (%s) Data\n""" % (str(zHI)))
            valueHI.print_out()
            core.print_out("""   Extrapolated Data:\n""")
            value.print_out()
            core.print_out("""   Alpha (exponent) Value:          %16.8f\n""" % alpha)
            core.print_out("""   Beta Data:\n""")
            beta.print_out()


#        value.add(valueSCF)
        return value

    else:
        raise ValidationError("corl_xtpl_helgaker_2: datatype is not recognized '%s'." % type(valueLO))

