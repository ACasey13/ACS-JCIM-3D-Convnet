#s
# gaussian_output: parses Gaussian output file and returns band gap, dipole, and molecular volume
# calc_edat: Runs edat and returns heat of formation and density


import os
import math
import sys
import numpy as np
import random
from collections import OrderedDict
from rdkit import Chem
#START   ##############################################################

def sect_h(title, char='*', num=40):
   num = int(num)
   print("")
   print(char*num)
   n_chars = len(title)
   if n_chars % 2:
      title += ' '
      n_chars += 5
   else:
      n_chars += 4
   diff = int((num - n_chars)/2)
   if diff > 3:
      spaces = diff - 3
      print(char*3, end='')
      print(' '*(spaces+2), end='')
      print(title, end='')
      print(' '*(spaces+2), end='')
      print(char*3)
   else:
      print(' ',end='')
      print(title)
   print(char*num)


def gaussian_output(f_name=None):
   band_gap=0.
   dipole=0.
   ese=0.
   energy=0.
   dx=0.
   dy=0.
   dz=0.
   hf=0.
   status=False
   if f_name == None:
       with open("input_density.out") as file:
           gaussian_out = file.readlines()
   else: 
       with open(f_name) as file:
           gaussian_out = file.readlines()

   istart=0
   nlines=len(gaussian_out)
# find output for last structure on file
   located=False
   i=0
   while i<nlines:
       if("Structure from the checkpoint file:" in gaussian_out[i]):
           istart=i
           located=True
           break
       i+=1
   if(not located): #this should never happen
       status=False
       print("Did not locate optimized structure on input_density.out")
       return band_gap,dipole,dx,dy,dz,ese,hf,energy,status

# find orbital energy section
   located=False
   k=istart
   while k<nlines:
       if("Alpha  occ. eigenvalues" in gaussian_out[k] and "Alpha virt. eigenvalues" in gaussian_out[k+1]):
              homo_list=gaussian_out[k].split()
              lumo_list=gaussian_out[k+1].split()
              located=True
              break
       k+=1
   if(not located): 
       status=False
       print("Did not locate homo/lumo section on input_density.out")
       return band_gap,dipole,dx,dy,dz,ese,hf,energy,status

   homo=float(homo_list[-1])
   lumo=float(lumo_list[4])
   band_gap=27.2116*(lumo-homo) #27.2116 ev per au
   print("homo, lumo, band gap")
   print(homo,lumo,band_gap)


# find dipole moment section
   located=False
   k=istart
   while k<nlines:
       if("Dipole moment" in gaussian_out[k]):
              dipole_line=gaussian_out[k+1].split()
              dipole=float(dipole_line[-1])
              located=True
              print("dipole")
              print(dipole)
              break
       k+=1
   if(not located): 
       status=False
       print("Did not locate dipole section on input_density.out")
       return band_gap,dipole,dx,dy,dz,ese,hf,energy,status


# find dipole moment section
   located=False
   k=istart
   while k<nlines:
       if("Electric dipole moment (input orientation)" in gaussian_out[k]):
              dipole_x_line=gaussian_out[k+4].split()
              dipole_y_line=gaussian_out[k+5].split()
              dipole_z_line=gaussian_out[k+6].split()
              dipole_x=dipole_x_line[-2].replace('D', 'E')
              dipole_y=dipole_y_line[-2].replace('D', 'E')
              dipole_z=dipole_z_line[-2].replace('D', 'E')

              d_moment = np.asarray([dipole_x, dipole_y, dipole_z]).astype(np.float)
              dx = d_moment[0]; dy = d_moment[1]; dz = d_moment[2];
              located=True
              print("dipole moment: [{}, {}, {}] (Debye)".format(dx,dy,dz))
              break
       k+=1
   if(not located):
       status=False
       print("Did not locate dipole moment (input orientation)section on input_density.out")
       return band_gap,dipole,dx,dy,dz,ese,hf,energy,status

# find HF energy section
   located=False
   k=istart
   while k<nlines:
       if("HF=" in gaussian_out[k]):
              hf_line=gaussian_out[k].split("HF=")[1].split("\\")[0]
              try:
                  hf=float(hf_line)
              except Exception as e:
                  hf = np.nan
                  print('!'*50)
                  print('HF could not be properly formatted!')
                  print('Exception: {}'.format(e))
              located=True
              print("hf: {}".format(hf))
              break
       k+=1
   if(not located):
       #status=False
       hf = np.nan
       print("Did not locate HF energy section on input_density.out")
       #return band_gap,dipole,dx,dy,dz,ese,hf,energy,status

# electronic spatial extent. low budget molecular volume estimator (au)
   located=False
   k=istart
   while k<nlines:
       if("Electronic spatial extent" in gaussian_out[k]):
              ese_line=gaussian_out[k].split()
              ese=float(ese_line[-1])
              located=True
              print("ese")
              print(ese)
              break
       k+=1

   if(not located): 
       status=False
       print("Did not locate electronic spatial extent on input_density.out")
       return band_gap,dipole,dx,dy,dz,ese,hf,energy,status


   located = False
   #print('Parsing for energy...')
   k = istart
   optimized_energies=[]
   while k<nlines:
       if("SCF Done:" in gaussian_out[k]):
           energy_line = gaussian_out[k].split()
           energy = float(energy_line[4])
           #print('An energy found! : {}'.format(energy))
           optimized_energies.append(energy)
       k+=1
       
   if optimized_energies:       
       energy = optimized_energies[-1]
       located=True
       print('Energy: {}'.format(energy))

   if (not located):
       status=False
       print('Did not locate energy in input_density.out')
       return band_gap,dipole,dx,dy,dz,ese,hf,energy,status


   status=True
   return band_gap,dipole,dx,dy,dz,ese,hf,energy,status

#END #############################################









#START   ##############################################################
def calculate_edat(smile_string,elements,xyz,dir_name):
    sect_h('Entering calculate_edat function...')
    success=False
    density=0.0
    heat=0.0
    bgap=0.0
    dpole=0.0
    molv=0.0
    energy=0.0
    hf=0.0
    dx=0.
    dy=0.
    dz=0.
    #dname2=smile_string.replace('(','Q').split()
    #dir_name=dname2[0].replace(')','Z').split()
    parent_dir = os.path.abspath(os.getcwd())
    dir_name = os.path.abspath(os.path.join('../data', dir_name))
    print("using directory %s" % dir_name)
    os.mkdir(dir_name)
    os.chdir(dir_name)

# Paths aren't working in edat scripts to just copy all scripts to cwd
    edat_dir = os.path.join(parent_dir, 'edat_scripts')
    if edat_dir[-1] == '/':
        edat_dir=edat_dir[:-1]
    os.system("cp {}/* .".format(edat_dir))
#obgrep 
    sect_h('Writing smile string...')
    with open("smile","w") as file:
        file.write(smile_string)
    file.close()
#    os.system("./smile2heatstring > fragments")
#    os.system("cp ../fragments .")
    sect_h('Reading in molecule string...')
    m = Chem.MolFromSmiles(smile_string)
    m = Chem.AddHs(m)
    tkgroups = OrderedDict()
    tkgroups['csp3'] = len(m.GetSubstructMatches(Chem.MolFromSmarts('[#6^3]')))
    tkgroups['H'] = len(m.GetSubstructMatches(Chem.MolFromSmarts('[H]')))
    tkgroups['nsp3'] = len(m.GetSubstructMatches(Chem.MolFromSmarts('[N&!$([N]=*)&!$([N]#*)&!$([N-])]')))
    tkgroups['osp3'] = len(m.GetSubstructMatches(Chem.MolFromSmarts('[#8^2]')))
    tkgroups['cprime'] = len(m.GetSubstructMatches(Chem.MolFromSmarts('[#6&!$([#6^3])]')))
    tkgroups['nprime'] = len(m.GetSubstructMatches(Chem.MolFromSmarts('[n,$(N=*),$(N#*),NX2]')))
    tkgroups['oprime'] = len(m.GetSubstructMatches(Chem.MolFromSmarts('[#8&!$([#8^2])]')))
    tkgroups['cnitros'] = len(m.GetSubstructMatches(Chem.MolFromSmarts('[C](-[NX3](=[OX1])[OX1-])')))
    tkgroups['nnitros'] = len(m.GetSubstructMatches(Chem.MolFromSmarts('[N](-[NX3](=[OX1])[OX1-])')))
    tkgroups['onitros'] = len(m.GetSubstructMatches(Chem.MolFromSmarts('[O](-[NX3](=[OX1])[OX1-])')))
    tkgroups['azides'] = len(m.GetSubstructMatches(Chem.MolFromSmarts('[C](-[NX2-]-[NX2+]#[NX1])')))
    tkgroups['nitroso'] = len(m.GetSubstructMatches(Chem.MolFromSmarts('*(-[NX2](=[OX1]))')))
    tkstring = ' '.join(str(x) for x in tkgroups.values())
    sect_h(r'Writing ./fragments file...')
    with open('fragments', 'w+') as file:
       file.write(tkstring+'\n')

    exists=os.path.isfile('./fragments')
    if(exists):
       with open("fragments") as file:
            fragments = file.readlines()
       file.close()
    else:
        print("error with SMILES conversion for SMILES ",smile_string)
        os.chdir(parent_dir)
        return success,density,heat,bgap,dpole,molv,energy,dx,dy,dz,hf

#edat
    sect_h(r"Writing edat input 'input.dat'...")
    word="input.dat"
    with open(word,"w") as file:
        file.write(fragments[0])
        natoms=len(xyz)
        np.random.seed(100)
        j=0
        while j<natoms:
            x1=0.15*np.random.random(1) #add random displacements to break symmetry
            y1=0.15*np.random.random(1)
            z1=0.15*np.random.random(1)
            file.write(str(elements[j])+" "+str(float(xyz[j][0])+float(x1))+" "+str(float(xyz[j][1])+float(y1))+" "+str(float(xyz[j][2])+float(z1))+"\n")
            j+=1
    file.close
#run toolkit and get density/heat of formation
    sect_h(r"Calling ./master_densheat.s script...")
    os.system("./master_densheat.s "+word+" &>output")
    exists=os.path.isfile('./output')
   
    sect_h(r'Reading EDAT from ./output')

    if(not exists):
        print("EDAT output file missing for: ",smile_string)
        os.chdir(parent_dir)         
        return success,density,heat,bgap,dpole,molv,energy,dx,dy,dz,hf
    else:
        with open("output") as file:
             edat_out = file.readlines()
        nline=len(edat_out)
        i=0
        dens=False
        hof=False
        while i<nline:
            if("g/cc" in edat_out[i]):
                dens=True
                iline=i
            elif("_fg" in edat_out[i]):
                hof=True
                jline=i
            i+=1
        if( dens and hof):
            tmp=[]
            tmp=edat_out[iline].split()
            density=float(tmp[-1])
            tmp=[]
            tmp=edat_out[jline].replace(',','').split()
# Per Ed, last Hf number is one we need
            heat=float(tmp[-1])
#            success=True
        else:
            print("Error with EDAT run for: ",smile_string)
            os.chdir(parent_dir)
            return success,density,heat,bgap,dpole,molv,energy,dx,dy,dz,hf

    sect_h('Parsing gaussian output...')
    try:
        bgap,dpole,dx,dy,dz,molv,hf,energy,okay=gaussian_output()
    except Exception as e:
        print("Gaussian output parsing error for: ".format(smile_string))
        print("Exception: {}".format(e))
        os.chdir(parent_dir)
        return success,density,heat,bgap,dpole,molv,energy,dx,dy,dz,hf

    if(not okay):
        print("Gaussian output parsing error for: ",smile_string)
        os.chdir(parent_dir)
        return success,density,heat,bgap,dpole,molv,energy,dx,dy,dz,hf

    success=True
    os.chdir(parent_dir)
    sect_h('Success! Leaving calculate_edat function.')
    return success,density,heat,bgap,dpole,molv,energy,dx,dy,dz,hf
#END   ##############################################################


















