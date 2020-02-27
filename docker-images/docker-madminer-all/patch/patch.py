#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import shutil
import argparse
from subprocess import Popen, PIPE


def call_command(cmd):
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = proc.communicate()
    exitcode = proc.returncode

    if exitcode != 0:
        raise RuntimeError(
            'Calling command {} returned exit code {}.\n\nStd output:\n\n{}Error output:\n\n{}'.format(
                cmd, exitcode, out, err
            )
        )


# Command-line arguments
parser = argparse.ArgumentParser(description='Patching reweighting issues in MG5-Pythia interface and Delphes')
parser.add_argument('madgraph', help='MadGraph directory')
parser.add_argument('--delphes', default=None, help='Delphes directory. Default is <MadGraph directory>/Delphes')
args = parser.parse_args()

print('')
print('Patching reweighting issues in MG5-Pythia interface and Delphes')
print('')

# Paths
new_files_path = './'
delphes_path = args.madgraph + '/Delphes' if args.delphes is None else args.delphes
pythia_path = args.madgraph + '/HEPTools/MG5aMC_PY8_interface'

print('Paths:')
print('  MadGraph:            {}'.format(args.madgraph))
print('  MG-Pythia interface: {}'.format(pythia_path))
print('  Delphes:             {}'.format(delphes_path))

# Update Pythia files
print('Updating {}'.format(pythia_path + '/MG5aMC_PY8_interface.cc'))

shutil.copyfile(pythia_path + '/MG5aMC_PY8_interface.cc', pythia_path + '/MG5aMC_PY8_interface_original.cc')
shutil.copyfile(new_files_path + '/MG5aMC_PY8_interface.cc', pythia_path + '/MG5aMC_PY8_interface.cc')

# Compile Pythia
print('Compiling Pythia')
call_command("cd " + pythia_path + "; ./compile.py ../pythia8/")

# Update Delphes files
print('Updating {}'.format(delphes_path + '/classes/DelphesHepMCReader_original.cc'))
print('Updating {}'.format(delphes_path + '/classes/classes/DelphesHepMCReader_original.cc'))
print('Updating {}'.format(delphes_path + '/readers/DelphesHepMC_original.cc'))

shutil.copyfile(delphes_path + '/classes/DelphesHepMCReader.cc',
                delphes_path + '/classes/DelphesHepMCReader_original.cc')
shutil.copyfile(delphes_path + '/classes/DelphesHepMCReader.h', delphes_path + '/classes/DelphesHepMCReader_original.h')
shutil.copyfile(delphes_path + '/readers/DelphesHepMC.cpp', delphes_path + '/readers/DelphesHepMC_original.cpp')

shutil.copyfile(new_files_path + '/DelphesHepMCReader.cc', delphes_path + '/classes/DelphesHepMCReader.cc')
shutil.copyfile(new_files_path + '/DelphesHepMCReader.h', delphes_path + '/classes/DelphesHepMCReader.h')
shutil.copyfile(new_files_path + '/DelphesHepMC.cpp', delphes_path + '/readers/DelphesHepMC.cpp')

# Compile Delphes
print('Compiling Delphes')

call_command("cd " + delphes_path + "; make")

print('')
print('All done -- have a nice day!')
print('')
