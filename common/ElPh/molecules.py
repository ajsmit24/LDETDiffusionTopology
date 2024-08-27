from pydoc import describe
import numpy as np
import json
from scipy import linalg
#from tqdm.auto import tqdm

class Molecules:
   def __init__(self, nmuc=None, coordmol=None, unitcell=None, 
   supercell=None, unique=None, uniqinter=None,
   javg=None, sigma=None, nrepeat=None,
   iseed=None, invtau=None, temp=None, 
   lattice_file=None, params_file=None, results={},
   static_disorder_params={"rel%":0}):
      """
      The molecules object represent a group of molecules. It has the relevant information regarding 
      the calculation of charge mobility such as the transfer integral of the pairs, the standard 
      deviation and structural information about the system.

      Args:
          nmuc (int, optional): Number of molcules in the unit cell. Defaults to None.
          coordmol (list, optional): Coordinate of the molecules in the unit cell. Defaults to None.
          unitcell (list, optional): The lattice parameters of the unitcell. Defaults to None.
          supercell (list, optional): Dimensions of the supercell. Defaults to None.
          unique (int, optional): Number of unique interactions. Defaults to None.
          uniqinter (list, optional): Description of each unique interaction. Defaults to None.
            This list contains six entries. 
            The first two entries describe the type of each molecule in the interaction.
            (Example: [1,1] represents the interaction between similar molecules.)
            The following three are the direction of interaction.
            (Example: [1,0,0] is an interaction in the x-direction)
            The last entry is the direction of the transfer integral.
            (Example: [1] uses the first value of the javg list.)
          javg (list, optional): Transfer integrals for all the pairs in eV. Defaults to None.
          sigma (list, optional): Standard deviation of the transfer integrals of 
            all the pairs in eV. Defaults to None.
          nrepeat (int, optional): Number of iterations with different realizations of 
            disorder. Defaults to None.
          iseed (int, optional): Index of random seed. Defaults to None.
          invtau (int, optional): Characteristic localization time. Defaults to None.
          temp (int, optional): Temperature in eV. Defaults to None.
          lattice_file (json, optional): The structural parameters json file. Defaults to None.
          params_file (json, optional): Parameters for the calculation of mobility. Defaults to None.
          results (dict, optional): Dictionary with resulting localization length 
            and mobilty. Defaults to {}.
      """

      if lattice_file:
         with open(lattice_file + '.json', 'r') as json_file:
            lat_dic = json.load(json_file)
         self.nmuc = lat_dic['nmuc']
         self.coordmol = np.array(lat_dic['coordmol'])
         self.unitcell = np.array(lat_dic['unitcell'])
         self.supercell = lat_dic['supercell']
         self.unique = lat_dic['unique']
         self.uniqinter = np.array(lat_dic['uniqinter'])
      else:
         self.nmuc = nmuc
         self.coordmol = np.array(coordmol)
         self.unitcell = np.array(unitcell)
         self.supercell = supercell
         self.unique = unique
         self.uniqinter = np.array(uniqinter)

      if params_file:
         with open(params_file + '.json', 'r') as json_file:
            par_dic = json.load(json_file)
         self.javg = np.array(par_dic['javg'])
         self.sigma = np.array(par_dic['sigma'])
         self.nrepeat = par_dic['nrepeat']
         self.iseed = par_dic['iseed']
         self.invtau = par_dic['invtau']
         self.temp = par_dic['temp']
      else:
         self.javg = np.array(javg )
         self.sigma = np.array(sigma)
         self.nrepeat = nrepeat
         self.iseed = iseed
         self.invtau = invtau
         self.temp = temp

      self.results = results

      # addind 0 for the case that molecules dont interact
      self.javg = np.insert(self.javg, 0, 0)
      self.sigma = np.insert(self.sigma, 0, 0)

      # making random numbers predictable
      np.random.seed(self.iseed)
      
      #print("*"*25+"\n",self.sigma,"\n"+"*"*25+"\n")
      self.sigma_scalar=self.sigma[1]
      self.static_disorder_params=static_disorder_params

   def get_interactions(self):
      """Calculate all the possible interactions in the supercell by translating the unit cell
      interactions and calculating the distance between molecules.

      Returns:
          tuple : (nmol, transinter_mm, distx_mm, disty_mm)
            (Number of molecules in super cell, 
            Translated interaction between a pair of molecules (Matrix size: [nmol, nmol]),
            Distances between all molecules THAT INTERACT with enforced PBC in x-direction,
            Distances between all molecules THAT INTERACT with enforced PBC in y-direction)
      """
      # nmol number of molecules in the supercell
      nmol = self.nmuc * self.supercell[0] * self.supercell[1] * \
         self.supercell[2]
      
      # t number of translations and k directions
      translations_tk = np.mgrid[0:self.supercell[0]:1, 
                                 0:self.supercell[1]:1, 
                                 0:self.supercell[2]:1].reshape(3,-1).T
      transvecs_tk = np.dot(translations_tk, self.unitcell)
      t = len(transvecs_tk)

      # nconnect number of connections
      nconnect = self.unique * t

      # mapping
      mapcell_k = [self.supercell[1] * self.supercell[2], 
                   self.supercell[2], 
                   1]
      map2same_t = self.nmuc * np.dot(translations_tk, mapcell_k)

      # u unique connections to other molecules and enforced PBC
      connec_tuk = (translations_tk[:, None] \
         + self.uniqinter[:, 2:5][None, :])
      connec_tuk[:, :, 0][connec_tuk[:, :, 0] == self.supercell[0]] = 0
      connec_tuk[:, :, 1][connec_tuk[:, :, 1] == self.supercell[1]] = 0
      connec_tuk[:, :, 2][connec_tuk[:, :, 2] == self.supercell[2]] = 0
      map2others_tu = self.nmuc * np.dot(connec_tuk, mapcell_k)
      
      # translated interactions between a pair of molecules
      transinter_mm = np.zeros([nmol, nmol], dtype='int')
      firstmol = (map2same_t[:, None] \
         + self.uniqinter[:, 0][None, :]).reshape(nconnect)
      secondmol = (map2others_tu[:, None] \
         + self.uniqinter[:, 1][None, :]).reshape(nconnect)
      typeinter = np.tile(self.uniqinter[:, 5], t)
      transinter_mm[firstmol - 1, secondmol - 1] = typeinter

      # enforce hermitian
      transinter_mm[secondmol - 1, firstmol - 1] = typeinter

      # translated coordinates for m molecules
      transcoords_mk = (transvecs_tk[:, None] + \
         self.coordmol[None, :]).reshape(nmol, 3)

      # distances between all molecules THAT INTERACT with enforced PBC
      distx_mm = np.zeros([nmol, nmol])
      disty_mm = np.zeros([nmol, nmol])
      for i, j in zip(firstmol, secondmol):
         distx_mm[i-1, j-1] = transcoords_mk[i-1, 0] - transcoords_mk[j-1, 0]
         disty_mm[i-1, j-1] = transcoords_mk[i-1, 1] - transcoords_mk[j-1, 1]

      superlengthx = self.unitcell[0, 0] * self.supercell[0]
      superlengthy = self.unitcell[1, 1] * self.supercell[1]

      distx_mm[distx_mm > superlengthx / 2] -= superlengthx
      distx_mm[distx_mm < -superlengthx / 2] += superlengthx
      disty_mm[disty_mm > superlengthy / 2] -= superlengthy
      disty_mm[disty_mm < -superlengthy / 2] += superlengthy

      return nmol, transinter_mm, distx_mm, disty_mm

   def get_hamiltonian(self, nmol, transinter_mm):
      """Calculating Hamiltonian

      Args:
          nmol (int): Number of molecules in supercell.
          transinter_mm (array): Translated interactions between a pair of molecules 
            (Matrix size: [nmol, nmol])

      Returns:
          array: Hamiltonian
      """
      rnd_mm = np.random.normal(0, 1, size=(nmol, nmol))
      rnd_mm = np.tril(rnd_mm) + np.tril(rnd_mm, -1).T

      hamiltonian_mm = self.javg[transinter_mm] + self.sigma[transinter_mm] * rnd_mm
      if(self.static_disorder_params["abs"]>0):
          #print(self.sigma[transinter_mm],self.sigma_scalar,self.javg[0],self.static_disorder_params["rel%"])
          hamiltonian_mm+= (self.sigma[transinter_mm]/self.sigma_scalar) * self.static_disorder_params["abs"]
    
      return hamiltonian_mm

   def get_energies(self, nmol, transinter_mm):
      """Calculate energies by solving Hamiltonian

      Args:
          nmol (int): Number of molecules in supercell.
          transinter_mm (array): Translated interactions between a pair of molecules 
            (Matrix size: [nmol, nmol])

      Returns:
          tuple: (energies_m.real, vectors_mm, hamiltonian_mm)
            (Real part of eigen values, Eigen vectors, Hamiltonian)
      """
      hamiltonian_mm = self.get_hamiltonian(nmol, transinter_mm)

      energies_m, vectors_mm = linalg.eigh(hamiltonian_mm)
      
      return energies_m.real, vectors_mm, hamiltonian_mm

   def get_squared_length(self):
      """Calculate squared transient localization length using
      distances, eigenvectors, eigenenergies and hamiltonian:

      L**2 = (1 / Z) ∑_nm exp(β E_n) |<n | j_x(y) | m>|**2 (2 / ((hbar**2 / tau) + (E_m + E_n)**2))

      <n | j_x(y) | m> = i <n | [H, x(y)] | m>

      Returns:
          tuple: (sqlx, sqly)
            (squared localization length in the x direction, and y direction)
      """
      nmol, transinter_mm, distx_mm, disty_mm = self.get_interactions()

      energies_m, vectors_mm, hamiltonian_mm = self.get_energies(nmol, transinter_mm)

      operatorx_mm = np.matmul(vectors_mm.T, np.matmul(distx_mm * hamiltonian_mm, vectors_mm))
      operatorx_mm -= np.matmul(vectors_mm.T, np.matmul(distx_mm * hamiltonian_mm, vectors_mm)).T

      operatory_mm = np.matmul(vectors_mm.T, np.matmul(disty_mm * hamiltonian_mm, vectors_mm))
      operatory_mm -= np.matmul(vectors_mm.T, np.matmul(disty_mm * hamiltonian_mm, vectors_mm)).T

      partfunc_m = np.exp(energies_m / self.temp)
      partfunc = sum(partfunc_m)

      sqlx = sum(sum(partfunc_m * (operatorx_mm**2 * 2 / (self.invtau**2 + \
         (energies_m[:, None] - energies_m[None, :])**2))))
      sqly = sum(sum(partfunc_m * (operatory_mm**2 * 2 / (self.invtau**2 + \
         (energies_m[:, None] - energies_m[None, :])**2))))

      sqlx /= partfunc
      sqly /= partfunc

      return sqlx, sqly

   def get_disorder_avg_sql(self):
      """Take average of squared localization length considering nrepeat iterations 
      with different disorder realizations

      Returns:
          tuple: (dsqlx, dsqly)
            (disorder-average squared localization length in the x direction, 
            and y direction)
      """
      dsqlx, dsqly = 0, 0
      self.results['squared_length_x'] = []
      self.results['squared_length_y'] = []
      self.results['squared_length'] = []
      self.results['avg_sqlx'] = []
      self.results['avg_sqly'] = []
      self.results['avg_sql'] = []      
      print('Calculating average of squared transient localization')
      for i in range(1, self.nrepeat + 1):
         sqlx, sqly = self.get_squared_length()
         self.results['squared_length_x'].append(sqlx)
         self.results['squared_length_y'].append(sqly)
         self.results['squared_length'].append((sqlx + sqly) / 2)

         #moving average
         dsqlx -= dsqlx / i
         dsqlx += sqlx / i

         dsqly -= dsqly / i
         dsqly += sqly / i

         self.results['avg_sqlx'].append(dsqlx)
         self.results['avg_sqly'].append(dsqly)
         self.results['avg_sql'].append((dsqlx + dsqly) / 2)

         # print(i, dsqlx, dsqly)

      return dsqlx, dsqly
   
   def get_mobility(self):
      """Calculate charge mobility in the framework of transient localization theory

      Returns:
          tuple: (mobx, moby)
            (mobility in the x direction, and y direction)
      """
      dsqlx, dsqly = self.get_disorder_avg_sql()

      # unit converter unit = ang**2 * e / hbar
      unit = 0.15192674605831966

      # in units of cm**2 / (V s)
      mobx = (1 / self.temp) * self.invtau * dsqlx / 2 * unit
      moby = (1 / self.temp) * self.invtau * dsqly / 2 * unit

      self.results['mobx'] = mobx
      self.results['moby'] = moby

      print('Calculating charge mobility')
      print('mu_x = ', mobx)
      print('mu_y = ', moby)
      print('mu_avg = ', (mobx + moby) / 2)

      return mobx, moby
