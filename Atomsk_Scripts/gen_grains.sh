folder=Atomsk_Files/Grain_Boundaries
rm -rf $folder
mkdir $folder
echo 'box 50 50 50
node 0.5*box 0.25*box 0 0° 0° -10°
node 0.5*box 0.75*box 0 0° 0° 10°' >> $folder/init_file.txt

atomsk --create bcc 3.14221 W orient 100 010 001 $folder/W_unitcell.xsf
atomsk --polycrystal $folder/W_unitcell.xsf $folder/init_file.txt $folder/polycrystal.lmp -wrap
python git_folder/Scripts/edit_atomsk_datafiles.py $folder/polycrystal.lmp
