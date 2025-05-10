import numpy as np
from rdkit import Chem


def one_of_k_encoding(x, allowable_set):
  if x not in allowable_set:
    raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))


def get_intervals(l):
  intervals = len(l) * [0]
  intervals[0] = 1
  for k in range(1, len(l)):
    intervals[k] = (len(l[k]) + 1) * intervals[k - 1]

  return intervals


def safe_index(l, e):
  try:
    return l.index(e)
  except:
    return len(l)


possible_atom_list = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 'Fe', 'Ca', 'Cu',
    'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn'
]
possible_numH_list = [0, 1, 2, 3, 4]
possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
possible_hybridization_list = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
possible_number_radical_e_list = [0, 1, 2]
possible_chirality_list = ['R', 'S']

reference_lists = [
    possible_atom_list, possible_numH_list, possible_valence_list,
    possible_formal_charge_list, possible_number_radical_e_list,
    possible_hybridization_list, possible_chirality_list
]

intervals = get_intervals(reference_lists)


def get_feature_list(atom):
  features = 6 * [0]
  features[0] = safe_index(possible_atom_list, atom.GetSymbol())
  features[1] = safe_index(possible_numH_list, atom.GetTotalNumHs())
  features[2] = safe_index(possible_valence_list, atom.GetImplicitValence())
  features[3] = safe_index(possible_formal_charge_list, atom.GetFormalCharge())
  features[4] = safe_index(possible_number_radical_e_list,
                           atom.GetNumRadicalElectrons())
  features[5] = safe_index(possible_hybridization_list, atom.GetHybridization())
  return features


def features_to_id(features, intervals):
  id = 0
  for k in range(len(intervals)):
    id += features[k] * intervals[k]

  id = id + 1
  return id


def atom_to_id(atom):
  features = get_feature_list(atom)
  return features_to_id(features, intervals)


def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=False):
  if bool_id_feat:
    return np.array([atom_to_id(atom)])
  else:
    results = one_of_k_encoding_unk(atom.GetSymbol(),
                                     ['C',
                                      'N',
                                      'O',
                                      'S',
                                      'F',
                                      'Si',
                                      'P',
                                      'Cl',
                                      'Br',
                                      'Mg',
                                      'Na',
                                      'Ca',
                                      'Fe',
                                      'As',
                                      'Al',
                                      'I',
                                      'B',
                                      'V',
                                      'K',
                                      'Tl',
                                      'Yb',
                                      'Sb',
                                      'Sn',
                                      'Ag',
                                      'Pd',
                                      'Co',
                                      'Se',
                                      'Ti',
                                      'Zn',
                                      'H',
                                      'Li',
                                      'Ge',
                                      'Cu',
                                      'Au',
                                      'Ni',
                                      'Cd',
                                      'In',
                                      'Mn',
                                      'Zr',
                                      'Cr',
                                      'Pt',
                                      'Hg',
                                      'Pb',
                                      'Unknown']) + \
              one_of_k_encoding(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                                    Chem.rdchem.HybridizationType.SP3D2]) +\
              [atom.GetIsAromatic()]
    
    if not explicit_H:
      results = results + \
                one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
      try:
        results = results + \
                  one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) +\
                  [atom.HasProp('_ChiralityPossible')]
      except:
        results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]
    
    return np.array(results)
