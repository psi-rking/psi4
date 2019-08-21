#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2019 The Psi4 Developers.
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
import shelve

from psi4 import core
from psi4.driver.p4util import *
from . import findif_response_utils
import roa

def run_roa(name, **kwargs):
    """
        Main driver for managing Raman Optical activity computations with
        CC response theory.

        Uses distributed finite differences approach -->
            1. Sets up a database to keep track of running/finished/waiting
                computations.
            2. Generates separate input files for displaced geometries.
            3. When all displacements are run, collects the necessary information
                from each displaced computation, and computes final result.

        Rewritten in 2019 to
         1. compute normal vibrational modes first and do 2*(3N-5/6) displacements
            (not 2*3N as before, but still ignoring molecule symmetry).
         2. analyze results with new python function
            (ccresponse/scatter.cc becomes obsolete)
         3. support computing Raman activity only of specified modes.
    """

    # Get list of omega values -> Make sure we only have one wavelength
    # Catch this now before any real work gets done
    print_lvl = core.get_global_option('PRINT')
    cnt = len(core.get_option('CCRESPONSE', 'OMEGA'))
    if cnt > 2:
        raise Exception('ROA scattering can only be performed for one wavelength.')
    elif cnt == 0:
        omega = 0.0
    elif cnt == 1:
        omega = roa.omega_in_au(core.get_option('CCRESPONSE', 'OMEGA'),'au')
    elif cnt == 2:
        omega = roa.omega_in_au(core.get_option('CCRESPONSE', 'OMEGA')[0],core.get_option('CCRESPONSE', 'OMEGA')[1])

    core.print_out( 'Running ROA computation. Subdirectories for each '
        'required displaced geometry are used.\n\n')

    dbno = 0
    db = shelve.open('database', writeback=True)
    # Check if final result is in here
    # ->if we have already computed roa, back up the dict
    # ->copy it setting this flag to false and continue
    if ('roa_computed' in db) and ( db['roa_computed'] ):
        db2 = shelve.open('.database.bak{}'.format(dbno), writeback=True)
        dbno += 1
        for key,value in db.items():
            db2[key]=value

        db2.close()
        db['roa_computed'] = False
    else:
        db['roa_computed'] = False

    print("roa_vib_modes length is %d" % len(core.get_option('CCRESPONSE', 'ROA_VIB_MODES')))

    use_atomic_cartesian_displacements = True
    if len(core.get_option('CCRESPONSE', 'ROA_VIB_MODES')):
        roa_vib_modes = core.get_option('CCRESPONSE', 'ROA_VIB_MODES')
        print('Doing displacements for normal modes:', roa_vib_modes)
        roa_vib_modes = [ m-1 for m in roa_vib_modes]  # internally -1
        use_atomic_cartesian_displacements = False
    else:
        print('Doing displacements for all atomic Cartesians')

    mol    = core.get_active_molecule()
    Natom  = mol.natom()
    geom   = mol.geometry().clone().np
    masses = np.array( [mol.mass(i) for i in range(Natom)] )
    hessian = roa.psi4_read_hessian(Natom)

    if use_atomic_cartesian_displacements:
        displacement_vectors = np.identity(3*Natom)
    else:
        # Do normal mode analysis and return the normal mode vectors (non-MW?) for
        # indices numbering from 0 (highest nu) downward.  vectors are columns
        (displacement_vectors, selected_freqs) = roa.modeVectors(geom,
          masses, hessian, roa_vib_modes, 2, core.print_out)

    # only needs to know how to label the displacements, e.g., 1_x_m or 5_p
    if 'inputs_generated' not in db:
        if use_atomic_cartesian_displacements:
            findif_response_utils.initialize_database(db,name,"roa", ["roa_tensor"])
        else:
            findif_response_utils.initialize_database(db,name,"roa", ["roa_tensor"],
              mode_indices=roa_vib_modes)

    # displaces + and - along provided vectors
    stepsize = core.get_global_option('RESPONSE_DISP_SIZE')
    vs_as_rows = displacement_vectors.T
    findif_response_utils.generate_inputs(db, name, vs_as_rows, stepsize)

    # Check job status
    if db['inputs_generated'] and not db['jobs_complete']:
        print('Checking status')
        findif_response_utils.stat(db)
        for job, status in db['job_status'].items():
            print("{} --> {}".format(job, status))

    # Compute ROA Scattering
    if db['jobs_complete']: # code does not and has not worked for multiple gauges
        # Gather data from displaced geometries
        fd_pol = findif_response_utils.collect_displaced_matrix_data(db,'Dipole Polarizability',3)
        fd_pol = np.array( fd_pol )
        if print_lvl > 2:
            core.print_out("Electric-Dipole/Dipole Polarizabilities")
            core.print_out(str(fd_pol))
        # custom function to read without db
        # fd_pol  = roa.psi4_read_polarizabilities(Natom, omega)

        fd_quad_list = findif_response_utils.collect_displaced_matrix_data(
            db, "Electric-Dipole/Quadrupole Polarizability", 9)
        fd_quad = []
        for row in fd_quad_list:
            fd_quad.append( np.array(row).reshape(9,3))
        if print_lvl > 2:
            core.print_out("Electric-Dipole/Quadrupole Polarizabilities")
            core.print_out(str(fd_quad)+'\n')
        # custom function to read without db
        # fd_quad = roa.psi4_read_dipole_quadrupole_polarizability(Natom, omega)

        mygauge = core.get_option('CCRESPONSE', 'GAUGE')
        consider_gauge = {
            'LENGTH': ['Length Gauge'],
            'VELOCITY': ['Modified Velocity Gauge'],
            'BOTH': ['Length Gauge', 'Modified Velocity Gauge']
        }

        # required for IR intensities; could be omitted if absent
        dipder  = roa.psi4_read_dipole_derivatives(Natom)
        if print_lvl > 2:
            core.print_out("Dipole Moment Derivatives")
            core.print_out(str(dipder)+'\n')

        for g in consider_gauge[mygauge]:
            core.print_out('Doing analysis (scatter function) for %s' % g)
            fd_rot = findif_response_utils.collect_displaced_matrix_data(db,
                    "Optical Rotation Tensor ({})".format(g), 3) # TODO check omega?
            fd_rot = np.array( fd_rot )
            if print_lvl > 2:
                core.print_out("Optical Rotation Tensor")
                core.print_out(str(fd_rot)+'\n')
            # custom function to read without db
            # fd_rot  = roa.psi4_read_optical_rotation_tensor(Natom, omega)
            core.print_out('\n\n----------------------------------------------------------------------\n')
            core.print_out('\t%%%%%%%%%% {} Results %%%%%%%%%%\n'.format(g))
            core.print_out('----------------------------------------------------------------------\n\n')

            bas = core.get_global_option('BASIS')
            NBF = core.BasisSet.build(mol, 'BASIS', bas).nbf()
            lbl = (name + '/' + bas).upper()

            if use_atomic_cartesian_displacements:
                roa.scatter(geom, masses, hessian, dipder, omega, stepsize,
                  fd_pol, fd_rot, fd_quad, print_lvl=2, pr=core.print_out, calc_type=lbl, nbf=NBF)
            else:
                roa.modeScatter(roa_vib_modes, displacement_vectors, selected_freqs,
                    geom, masses, dipder, omega, stepsize, fd_pol, fd_rot, fd_quad,
                    print_lvl=2, pr=core.print_out, calc_type=lbl, nbf=NBF)

            #print('roa.py:85 I am not being passed a molecule, grabbing from global :(')
            #core.scatter(core.get_active_molecule(), stepsize, dip_polar_list,
            # gauge, dip_quad_polar_list)

        db['roa_computed'] = True

    db.close()

# ################################
# ###                          ###
# ###    DATABASE STRUCTURE    ###
# ###                          ###
# ################################

# Dict of dicts
# inputs_generated (boolean)
# job_status: (ordered Dict)
#    key-> {atom}_{cord}_{p/m}
#       val-> (not_started,running,finished)
#    job_list: (string)
#        status (string)
# jobs_complete (boolean)
# roa_computed (boolean)
# prop (string) = roa
#

# ?
# data: dipole_polarizability
#    : optical_rotation
#    : dipole_quadrupole_polarizability
# ?
# results:
