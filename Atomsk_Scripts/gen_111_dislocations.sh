folder=Atomsk_Files/Dislocations
rm -rf $folder
mkdir $folder
b=2.721233684
atomsk --create bcc 3.1652 W orient 111 11-2 -110 $folder/W_unitcell_111.xsf
atomsk $folder/W_unitcell_111.xsf -duplicate 30 20 8 $folder/W_supercell_111.xsf     
atomsk $folder/W_supercell_111.xsf -dislocation "0.51*box" "0.51*box" edge_add Z Y $b 0.281965 $folder/W_edge_111.cfg
atomsk $folder/W_supercell_111.xsf -disloc loop "0.501*box" "0.501*box" "0.501*box" Y 15 $b 0 0 0.2819650067 $folder/W_loop_111.cfg
rm -rf $folder/W_unitcell_111.xsf
rm -rf $folder/W_supercell_111.xsf    
atomsk --create bcc 3.1652 W orient 11-2 1-10 111 $folder/W_unitcell_111.xsf
atomsk $folder/W_unitcell_111.xsf -duplicate 20 30 10 $folder/W_supercell_111.xsf     
atomsk $folder/W_supercell_111.xsf -dislocation "0.51*box" "0.51*box" screw Z Y $b $folder/W_screw_111.cfg
atomsk $folder/W_edge_111.cfg lammps
atomsk $folder/W_screw_111.cfg lammps
atomsk $folder/W_loop_111.cfg lammps
python git_folder/Scripts/edit_atomsk_datafiles.py $folder/W_edge_111.lmp
python git_folder/Scripts/edit_atomsk_datafiles.py $folder/W_loop_111.lmp
python git_folder/Scripts/edit_atomsk_datafiles.py $folder/W_screw_111.lmp
# mpiexec -n 4 python git_folder/Scripts/disc_loop_atomsk.py