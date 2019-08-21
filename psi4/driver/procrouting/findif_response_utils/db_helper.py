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

"""
Module of helper functions for ccresponse distributed property calculations.
Defines functions for interacting with the database created by the run_XXX
driver function.

Properties that are able to use this module should be added to
the registered_props dictionary.

"""
from __future__ import absolute_import
from __future__ import print_function
import os
import collections
import numpy as np

from psi4 import core
from psi4.driver import p4util

# RAK modeVectors are assumed 1D and normalized
def geom_displacements(mol, modeVectors, step):
    Natom = mol.natom()
    for v in modeVectors:
        if len(v) != 3*Natom:
            raise Exception('Mode displacement vector is wrong length.')

    disps_np = []
    for v in modeVectors:
        dispM = mol.geometry().clone().np
        dispP = mol.geometry().clone().np
        for atom in range(Natom):
           for xyz in range(3):
            dispM[atom][xyz] -= step * v[3*atom+xyz]
            dispP[atom][xyz] += step * v[3*atom+xyz]
        disps_np.append(dispM)
        disps_np.append(dispP)

    # need to get into psi4.core.Matrix form for set_geometry() function
    disps = []
    for d in disps_np:
        disps.append(core.Matrix.from_array(d))
    return disps

def generate_inputs(db,name,disp_vectors,stepsize):
    """
        Generates the input files in each sub-directory of the
        distributed finite differences property calculation.
        The default is to generate Cartesian atomics +/-, but alteratively
        user can pass in the desired displacement_geoms

    name: ( string ) method name passed to calling driver,
    db:   (database) The database object associated with this property
          calculation. On exit this db['inputs_generated'] has been set True

    Returns: nothing
    Throws: Exception if the number of atomic displacements is not correct.
    """
    molecule = core.get_active_molecule()
    natom = molecule.natom()

    print("Number of displacement vectors: %d" % len(disp_vectors))
    displacement_geoms = geom_displacements(molecule, disp_vectors, stepsize)
    print("Number of displaced geoms: %d" % len(displacement_geoms))
    if 2*len(disp_vectors) != len(displacement_geoms):
        raise Exception('The number of displacement vectors and geometries is inconsistent.')

    displacement_names = db['job_status'].keys()
    #print( str(displacement_names) )

    for n, entry in enumerate(displacement_names):
        if not os.path.exists(entry):
            os.makedirs(entry)

        # Setup up input file string
        inp_template = 'molecule {molname}_{disp}'
        inp_template += ' {{\n{molecule_info}\n}}\n{options}\n{jobspec}\n'
        molecule.set_geometry(displacement_geoms[n])
        molecule.fix_orientation(True)
        molecule.fix_com(True)
        inputfile = open('{0}/input.dat'.format(entry), 'w')
        inputfile.write("# This is a psi4 input file auto-generated for"
            "computing properties by finite differences.\n\n")
        inputfile.write(
            inp_template.format(
                molname=molecule.name(),
                disp=entry,
                molecule_info=molecule.create_psi4_string_from_molecule(),
                options=p4util.format_options_for_input(),
                jobspec=db['prop_cmd']))
        inputfile.close()
    db['inputs_generated'] = True

    # END generate_inputs


def initialize_database(database, name, prop, properties_array,
      additional_kwargs=None, mode_indices=None):
    """
        Initialize the database for computation of some property
        using distributed finite differences driver

    database: (database) the database object passed from the caller
    name:  (string) name as passed to calling driver
    prop: (string) the property being computed, used to add xxx_computed flag
        to database
    prop_array: (list of strings) properties to go in
        properties kwarg of the properties() cmd in each sub-dir
    additional_kwargs: (list of strings) *optional*
        any additional kwargs that should go in the call to the
        properties() driver method in each subdir
    modeIndices: (list of ints) *optional* If provided, serves as
        labels for displacements and indicates their number.  If absent,
        labels are atom indices and 3N displacements are expected.

    Returns: nothing
    Throws: nothing
    """
    database['inputs_generated'] = False
    database['jobs_complete'] = False

    # Construct the property input line a displacement
    prop_cmd ="properties('{0}',".format(name)
    prop_cmd += "properties=[ '{}' ".format(properties_array[0])
    if len(properties_array) > 1:
        for element in properties_array[1:]:
            prop_cmd += ",'{}'".format(element)
    prop_cmd += "]"
    if additional_kwargs is not None:
        for arg in additional_kwargs:
            prop_cmd += ", {}".format(arg)
    prop_cmd += ")"
    database['prop_cmd'] = prop_cmd

    database['job_status'] = collections.OrderedDict()
    # Populate the job_status dict
    molecule = core.get_active_molecule()
    natom = molecule.natom()
    xyz_coordinates = ['x', 'y', 'z']

    # Determine type of diplacements
    if mode_indices is not None:
        coordinate_lbls = [m+1 for m in mode_indices]  # will be directory name so +1
    else:
        coordinate_lbls = []
        # Number in lbl will refer to atom
        for atom in range(1, natom + 1):
            for xyz in xyz_coordinates:
                coordinate_lbls.append('{}_{}'.format(atom, xyz))
    print("Displacement labels to be used:")
    for l in coordinate_lbls:
        print(l)

    step_direction = ['m', 'p']
    for l in coordinate_lbls:
        for step in step_direction:
            job_name = '{}_{}'.format(l, step)
            database['job_status'].update({job_name: 'not_started'})

    database['{}_computed'.format(prop)] = False

    # END initialize_database()


def stat(db):
    """
        Checks displacement sub_directories for the status of each
        displacement computation

    db: (database) the database storing information for this distributed
        property calculation

    Returns: nothing
    Throws: nothing
    """
    n_finished = 0
    for job, status in db['job_status'].items():
        if status == 'finished':
            n_finished += 1
        elif status in ('not_started', 'running'):
            try:
                with open("{0}/output.dat".format(job),'r') as outfile:
                    for line in outfile:
                        if 'Psi4 exiting successfully' in line:
                            db['job_status'][job] = 'finished'
                            n_finished += 1
                            break
                        else:
                            db['job_status'][job] = 'running'
            except:
                pass
    # check all jobs done?
    if n_finished == len(db['job_status'].keys()):
        db['jobs_complete'] = True

    # END stat()
