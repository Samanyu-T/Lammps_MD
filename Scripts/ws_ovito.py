# Boilerplate code generated by OVITO Pro 3.10.6
from ovito.io import *
from ovito.modifiers import *
from ovito.data import *
from ovito.pipeline import *
import sys, os 

# Data import:
file = sys.argv[1]
folder = os.path.basename(file).split('.')
split = folder[0].split('_')
hconc = int(split[0][2:])
heconc = int(split[1][2:])
dpa = int(folder[1]) * 0.0002

pipeline = import_file(file)

# Expression selection:
pipeline.modifiers.append(ExpressionSelectionModifier(expression = 'ParticleType!=1'))

# Delete selected:
pipeline.modifiers.append(DeleteSelectedModifier())

# Wigner-Seitz defect analysis:
mod = WignerSeitzAnalysisModifier()
mod.reference = FileSource()
mod.affine_mapping = ReferenceConfigurationModifier.AffineMapping.ToReference
pipeline.modifiers.append(mod)
mod.reference.load('%s/%s.0.dump.gz' % (os.path.dirname(file), folder[0]))

data = pipeline.compute()
occupancy = data.particles['Occupancy']

nvac = len(occupancy[occupancy == 0])

with open('vac_data.log', 'a') as file:
    file.write('%d %d %f %d' % (hconc, heconc, dpa, nvac))