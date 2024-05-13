# Boilerplate code generated by OVITO 3.1.3
from ovito.io import *
from ovito.modifiers import *
from ovito.pipeline import *

import numpy as np
import sys, os, glob
 
def main(rpath, fpath, sfolder, idx):

    # Data import:
    # rpath = sys.argv[1]
    # fpath = sys.argv[2]

    sfile = "int_%s" % idx

    spath = os.path.join(os.path.join(sfolder, sfile))

    print ("Processing %s, exporting interstitial cluster data into %s" % (fpath, spath))
    
    # if os.path.isfile(spath):
    #     print ("File %s already exists. Skipping." % spath)
    #     return 0
     
    # pipeline = import_file(fpath)

    # # Wigner-Seitz defect analysis:
    # mod = WignerSeitzAnalysisModifier()
    # mod.per_type_occupancies = True
    # mod.reference = FileSource()
    # mod.reference.load(rpath)
    # mod.affine_mapping = ReferenceConfigurationModifier.AffineMapping.ToReference
    # pipeline.modifiers.append(mod)
    
    # # Expression selection:
    # pipeline.modifiers.append(ExpressionSelectionModifier(expression = 'Occupancy <= 1'))

    # # Delete selected:
    # pipeline.modifiers.append(DeleteSelectedModifier())
 
    # # Cluster analysis of remaining interstitials
    # pipeline.modifiers.append(ClusterAnalysisModifier(cutoff = 3.3, sort_by_size=True))
    
    # for i,frame in enumerate(range(pipeline.source.num_frames)):
    #     data = pipeline.compute(frame)

    #     # get cluster size data
    #     if data.tables != None:

    #         cluster_size = np.array(data.tables['clusters'].y.T)

    #         print(spath)

    #         np.savetxt(spath, cluster_size)


    sfile = "vac_%s" % idx

    spath = os.path.join(os.path.join(sfolder, sfile))

    pipeline2 = import_file(fpath)

    # Wigner-Seitz defect analysis:
    mod = WignerSeitzAnalysisModifier()
    mod.per_type_occupancies = True
    mod.reference = FileSource()
    mod.reference.load(rpath)
    mod.affine_mapping = ReferenceConfigurationModifier.AffineMapping.ToReference
    pipeline2.modifiers.append(mod)
    
    # Expression selection:
    pipeline2.modifiers.append(ExpressionSelectionModifier(expression = 'Occupancy >= 1'))

    # Delete selected:
    pipeline2.modifiers.append(DeleteSelectedModifier())
 
    # Cluster analysis of remaining interstitials
    pipeline2.modifiers.append(ClusterAnalysisModifier(cutoff = 3.3, sort_by_size=True))
    
    for i,frame in enumerate(range(pipeline2.source.num_frames)):
        data = pipeline2.compute(frame)

        # get cluster size data
        if data.tables != None:

            cluster_size = np.array(data.tables['clusters'].y.T)

            print(spath)

            np.savetxt(spath, cluster_size)


if __name__=="__main__":
    
    rpath = '/home/ir-tiru1/rds/rds-ukaea-ap002-mOlK9qn0PlQ/CRAsimulations/Cascades/w_220_cascade/w_220_cascade.0.dump.gz'

    files = glob.glob('/home/ir-tiru1/rds/rds-ukaea-ap002-mOlK9qn0PlQ/CRAsimulations/Cascades/w_220_cascade/w_220_cascade.*.dump.gz')
    
    dump_idx = sorted([int(file.split('.')[1]) for file in files])

    chosen_idx = np.linspace(0, len(files) - 1, 100).astype(int)
    
    sfolder = '/home/ir-tiru1/w_220_cascade'

    if not os.path.exists(sfolder):
        os.mkdir(sfolder)
    
    
    for i in chosen_idx:

        idx = dump_idx[i]
        
        fpath = '/home/ir-tiru1/rds/rds-ukaea-ap002-mOlK9qn0PlQ/CRAsimulations/Cascades/w_220_cascade/w_220_cascade.%d.dump.gz' % idx

        main(rpath, fpath, sfolder, idx)
