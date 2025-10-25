import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdPartialCharges
from rdkit.Chem import MolSurf
from rdkit.Chem import Crippen
from rdkit.Chem import EState
from rdkit.Chem import Scaffolds
from rdkit import RDLogger
import torch
from typing import List
import torch_geometric.data as data
RDLogger.DisableLog('rdApp.*')

class StructureEncoderV2():
    def __init__(self, directed:bool=False):

        self.directed = directed

        self.ATOM_TYPE = [
            'H', 'C', 'N', 'O', 'F', 'P', 'S', 
            'Cl', 'Br', 'I', 'B', 'Si', 'Se', 
            'As', 'Al', 'Zn'
        ]

        self.HYBRIDIZATION = [
            rdchem.HybridizationType.SP,
            rdchem.HybridizationType.SP2,
            rdchem.HybridizationType.SP3,
            rdchem.HybridizationType.SP3D,
            rdchem.HybridizationType.SP3D2,
        ]
        
        self.TOTAL_NUM_Hs = list(range(0, 5))
        self.FORMAL_CHARGE = list(range(-2, 3))
        self.TOTAL_DEGREE = list(range(0, 6))
        self.NUM_RADICAL = list(range(0, 3))
        
        self.CHIRAL_TYPE = [
            rdchem.ChiralType.CHI_UNSPECIFIED,
            rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ]
        
        self.BOND_TYPE = [
            rdchem.BondType.SINGLE,
            rdchem.BondType.DOUBLE,
            rdchem.BondType.TRIPLE,
            rdchem.BondType.AROMATIC,
        ]

        self.BOND_TYPE = [
        rdchem.BondType.SINGLE,
        rdchem.BondType.DOUBLE,
        rdchem.BondType.TRIPLE,
        rdchem.BondType.AROMATIC,
        ]
    
        self.BOND_STEREO = [
            rdchem.BondStereo.STEREOANY,
            rdchem.BondStereo.STEREOATROPCCW,
            rdchem.BondStereo.STEREOATROPCW,
            rdchem.BondStereo.STEREOCIS,
            rdchem.BondStereo.STEREOTRANS,
            rdchem.BondStereo.STEREOE,
            rdchem.BondStereo.STEREOZ,
            rdchem.BondStereo.STEREONONE,
        ]

    def one_hot_encode(self, value, value_dict: List) -> List[float]:
        len_desc = len(value_dict)
        desc_one_hot = np.zeros(len_desc)
        if value in value_dict:
            desc_one_hot[value_dict.index(value)] = 1.0
        else:
            desc_one_hot[-1] = 1.0
        return desc_one_hot

    def _get_max_valence(self, atomic_num):
        max_valence = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1, 35: 1, 53: 1}
        return max_valence.get(atomic_num, 8)
    
    def _is_bond_donor(self, atom):
        return (atom.GetAtomicNum() in [7, 8, 16] and atom.GetTotalNumHs() > 0)
    
    def _is_bond_acceptor(self, atom):
        return (atom.GetAtomicNum() in [7, 8, 16] and 
                atom.GetTotalValence() - atom.GetTotalNumHs() < self._get_max_valence(atom.GetAtomicNum()))
    
    def _is_basic(self, atom):
        # Nitrogen with lone pair (sp3 or aromatic N)
        return (atom.GetAtomicNum() == 7 and 
                atom.GetTotalDegree() < 4 and
                atom.GetFormalCharge() == 0)
    
    def _is_acidic(self, atom, mol):
        # Carboxylic acid O, sulfonic acid O, phosphoric acid O
        if atom.GetAtomicNum() == 8:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() in ['C', 'S', 'P']:
                    double_bonded_O = sum(1 for n in neighbor.GetNeighbors() 
                                        if n.GetSymbol() == 'O' and 
                                        mol.GetBondBetweenAtoms(neighbor.GetIdx(), n.GetIdx()).GetBondType() == rdchem.BondType.DOUBLE)
                    if double_bonded_O >= 1:
                        return True
        return False
    
    def _is_halogen(self, atom):
        return atom.GetAtomicNum() in [9, 17, 35, 53]  # F, Cl, Br, I
    
    def one_hot_encode(self, value: str, value_dict: List[float]) -> List[float]:
        len_desc = len(value_dict)
        desc_one_hot = np.zeros(len_desc)
        if value in value_dict:
            desc_one_hot[value_dict.index(value)] = 1.0
        else:
            desc_one_hot[-1] = 1.0

        return desc_one_hot
    
    def _cal_bond_weights(self, bond):
        bond_wt = 0
        if bond.GetBondType() == rdchem.BondType.SINGLE:
            bond_wt += 0.25
        if bond.GetBondType() == rdchem.BondType.DOUBLE:
            bond_wt += 0.5
        if bond.GetBondType() == rdchem.BondType.TRIPLE:
            bond_wt += 0.75
        if bond.GetBondType() == rdchem.BondType.AROMATIC:
            bond_wt += 1.0

        if bond.GetBeginAtom().GetSymbol() != 'C' or bond.GetEndAtom().GetSymbol() != 'C':
            bond_wt += 0.5

        return bond_wt  

    def atom_encoder(self, atoms, mol, scaffold_atoms):
        atoms_encoder = []
        for atom in atoms:
            atom_encoded = []
            atom_encoded.extend(self.one_hot_encode(atom.GetSymbol(), self.ATOM_TYPE))
            atom_encoded.extend(self.one_hot_encode(atom.GetTotalDegree(), self.TOTAL_DEGREE))
            atom_encoded.extend(self.one_hot_encode(atom.GetFormalCharge(), self.FORMAL_CHARGE))
            atom_encoded.extend(self.one_hot_encode(atom.GetNumRadicalElectrons(), self.NUM_RADICAL))
            atom_encoded.extend(self.one_hot_encode(atom.GetHybridization(), self.HYBRIDIZATION))
            atom_encoded.append(float(atom.GetIsAromatic()))
            atom_encoded.extend(self.one_hot_encode(atom.GetTotalNumHs(), self.TOTAL_NUM_Hs))
            atom_encoded.extend(self.one_hot_encode(atom.GetChiralTag(), self.CHIRAL_TYPE))
            atom_encoded.append(float(self._is_bond_donor(atom)))
            atom_encoded.append(float(self._is_bond_acceptor(atom)))
            atom_encoded.append(float(self._is_basic(atom)))
            atom_encoded.append(float(self._is_acidic(atom, mol)))
            atom_encoded.append(float(self._is_halogen(atom)))
            atom_encoded.append(float(atom.GetIdx() in scaffold_atoms))
            atoms_encoder.append(atom_encoded)
        return np.asarray(atoms_encoder, dtype=np.float32)
    
    def bond_encoder(self, bond):
        bond_encoded = []
        bond_encoded.extend(self.one_hot_encode(bond.GetBondType(), self.BOND_TYPE)) # Get Bond Type
        bond_encoded.extend(self.one_hot_encode(bond.GetStereo(), self.BOND_STEREO)) # Get Bond Stereo
        bond_encoded.append(float(bond.GetIsConjugated())) # Is conjugated bond
        bond_encoded.append(float(bond.IsInRing()))
        bond_encoded.append(float(bond.GetIsAromatic()))
        
        return bond_encoded

    def encoding_structure(self, smi: str, label: float) -> data:

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError('Invalid SMILES!!!')
        atoms = list(mol.GetAtoms())
        if len(atoms) == 0:
            raise ValueError('Molecule has no atom')
        
        try:
            scaffold = Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_atoms = set(scaffold.GetSubstructMatch(scaffold))
        except:
            scaffold_atoms = set()

        bonds = list(mol.GetBonds())
        rings = list(mol.GetRingInfo().AtomRings())
            
        if len(bonds) == 0:
            embeddings = torch.tensor(self.atom_encoder(atom, mol), dtype=torch.float)
            edges_index = torch.zeros((2,0), dtype=torch.int64)
            edges_attr = torch.zero((0, 28), dtype=torch.float)
            edges_weights = torch.zeros((0,), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.float)
        
        edges_attr =  []
        edges_weights = []
        edges_index = [[], []]
        for bond in bonds:
            bond_features = self.bond_encoder(bond)
            bond_weight = self._cal_bond_weights(bond)

            if self.directed:
                edges_index[0].append(bond.GetBeginAtomIdx())
                edges_index[1].append(bond.GetEndAtomIdx())
                edges_attr.append(bond_features)
                edges_weights.append(bond_weight)
            else:
                edges_index[0].extend([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                edges_index[1].extend([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
                edges_attr.extend([bond_features, bond_features])
                edges_weights.extend([bond_weight, bond_weight])

        embeddings = torch.tensor(self.atom_encoder(atoms, mol, scaffold_atoms), dtype=torch.float)
        edges_index = torch.tensor(edges_index, dtype=torch.int64)
        edges_attr = torch.tensor(edges_attr, dtype=torch.float)
        edges_weights = torch.tensor(edges_weights, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.float)

        return data.Data(
            x=embeddings,
            edge_index=edges_index,
            edge_attr=edges_attr,
            edge_weight=edges_weights,
            y = y,
        )

class StructureEncoderV1():
    def __init__(self, directed:bool=False):

        self.directed = directed

        self.ATOM_TYPE = [
            'H',
            'C',
            'N',
            'O',
            'F',
            'P',
            'S',
            'Cl',
            'Br',
            'I',
            'B',
            'Si',
            'Se'
        ]

        self.HYBRIDIZATION = [
            rdchem.HybridizationType.SP,
            rdchem.HybridizationType.SP2,
            rdchem.HybridizationType.SP3,
            rdchem.HybridizationType.SP3D,
            rdchem.HybridizationType.SP3D2,]
        
        self.TOTAL_NUM_Hs = list(range(0,5))
        
        self.FORMAL_CHARGE = list(range(-2,3))
        
        self.TOTAL_DEGREE = list(range(0,6))
        
        self.RING_SIZE = list(range(3,9))

        self.AROMATIC = [True, False]

        self.CHIRAL_TYPE = [
            rdchem.ChiralType.CHI_UNSPECIFIED,
            rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ]
        
        self.BOND_TYPE = [
            rdchem.BondType.SINGLE,
            rdchem.BondType.DOUBLE,
            rdchem.BondType.TRIPLE,
            rdchem.BondType.AROMATIC,
            ]
        
        self.BOND_STEREO = [
            rdchem.BondStereo.STEREOANY,
            rdchem.BondStereo.STEREOATROPCCW,
            rdchem.BondStereo.STEREOATROPCW,
            rdchem.BondStereo.STEREOCIS,
            rdchem.BondStereo.STEREOTRANS,
            rdchem.BondStereo.STEREOE,
            rdchem.BondStereo.STEREOZ,
            rdchem.BondStereo.STEREONONE,
        ]
    
    def one_hot_encode(self, value: str, value_dict: List[float]) -> List[float]:
        len_desc = len(value_dict)
        desc_one_hot = np.zeros(len_desc)
        if value in value_dict:
            desc_one_hot[value_dict.index(value)] = 1.0
        else:
            desc_one_hot[-1] = 1.0

        return desc_one_hot
    
    def _cal_bond_weights(self, bond):
        bond_wt = 0
        if bond.GetBondType() == rdchem.BondType.SINGLE:
            bond_wt += 0.25
        if bond.GetBondType() == rdchem.BondType.DOUBLE:
            bond_wt += 0.5
        if bond.GetBondType() == rdchem.BondType.TRIPLE:
            bond_wt += 0.75
        if bond.GetBondType() == rdchem.BondType.AROMATIC:
            bond_wt += 1.0

        if bond.GetBeginAtom().GetSymbol() != 'C' or bond.GetEndAtom().GetSymbol() != 'C':
            bond_wt += 0.5

        return bond_wt  

    def atom_encoder(self, atoms, mol):
        atoms_encoder = []
        for atom in atoms:
            atom_encoded = []
            atom_type = atom.GetSymbol()
            atom_encoded.extend(self.one_hot_encode(atom_type, self.ATOM_TYPE)) # Atom symbol C, H, O, N, ...
            atom_encoded.extend(self.one_hot_encode(atom.GetTotalDegree(), self.TOTAL_DEGREE)) # Total degree
            atom_encoded.extend(self.one_hot_encode(atom.GetTotalNumHs(), self.TOTAL_NUM_Hs)) # Number of Hydrogens
            atom_encoded.extend(self.one_hot_encode(atom.GetHybridization(), self.HYBRIDIZATION)) # Atom hybridization type
            atom_encoded.append(float(atom.GetIsAromatic())) # Atom in aromatic ring
            atom_encoded.extend(self.one_hot_encode(atom.GetChiralTag(), self.CHIRAL_TYPE)) # Get Chiral Type for atom
            atoms_encoder.append(atom_encoded)

        return np.asarray(atoms_encoder, dtype=np.float32)
    
    def bond_encoder(self, bond):
        bond_encoded = []
        bond_encoded.extend(self.one_hot_encode(bond.GetBondType(), self.BOND_TYPE)) # Get Bond Type
        bond_encoded.extend(self.one_hot_encode(bond.GetStereo(), self.BOND_STEREO)) # Get Bond Stereo
        bond_encoded.append(float(bond.GetIsConjugated())) # Is conjugated bond
        bond_encoded.append(float(bond.IsInRing()))
        bond_encoded.append(float(bond.GetIsAromatic()))
        
        return bond_encoded

    def encoding_structure(self, smi: str, label: float) -> data:

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError('Invalid SMILES!!!')
        atoms = list(mol.GetAtoms())
        if len(atoms) == 0:
            raise ValueError('Molecule has no atom')
        
        bonds = list(mol.GetBonds())
        rings = list(mol.GetRingInfo().AtomRings())
            
        if len(bonds) == 0:
            embeddings = torch.tensor(self.atom_encoder(atom, mol), dtype=torch.float)
            edges_index = torch.zeros((2,0), dtype=torch.int64)
            edges_attr = torch.zero((0, 28), dtype=torch.float)
            edges_weights = torch.zeros((0,), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.float)
        
        edges_attr =  []
        edges_weights = []
        edges_index = [[], []]
        for bond in bonds:
            bond_features = self.bond_encoder(bond)
            bond_weight = self._cal_bond_weights(bond)

            if self.directed:
                edges_index[0].append(bond.GetBeginAtomIdx())
                edges_index[1].append(bond.GetEndAtomIdx())
                edges_attr.append(bond_features)
                edges_weights.append(bond_weight)
            else:
                edges_index[0].extend([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                edges_index[1].extend([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
                edges_attr.extend([bond_features, bond_features])
                edges_weights.extend([bond_weight, bond_weight])

        embeddings = torch.tensor(self.atom_encoder(atoms, mol), dtype=torch.float)
        edges_index = torch.tensor(edges_index, dtype=torch.int64)
        edges_attr = torch.tensor(edges_attr, dtype=torch.float)
        edges_weights = torch.tensor(edges_weights, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.float)

        return data.Data(
            x=embeddings,
            edge_index=edges_index,
            edge_attr=edges_attr,
            edge_weight=edges_weights,
            y = y,
        )

class StructureEncoderV3():
    def __init__(self, directed:bool=False):

        self.directed = directed

        # 10 basic atom types
        self.ATOM_TYPE = [
            'H', 'C', 'N', 'O', 'F', 
            'P', 'S', 'Cl', 'Br', 'I'
        ]
        
        self.FORMAL_CHARGE = list(range(-2, 3))
        self.TOTAL_NUM_Hs = list(range(0, 5))
        self.NUM_HEAVY_NEIGHBORS = list(range(0, 5))
        
        self.BOND_TYPE = [
            rdchem.BondType.SINGLE,
            rdchem.BondType.DOUBLE,
            rdchem.BondType.TRIPLE,
            rdchem.BondType.AROMATIC,
            ]
        
        self.BOND_STEREO = [
            rdchem.BondStereo.STEREOANY,
            rdchem.BondStereo.STEREOATROPCCW,
            rdchem.BondStereo.STEREOATROPCW,
            rdchem.BondStereo.STEREOCIS,
            rdchem.BondStereo.STEREOTRANS,
            rdchem.BondStereo.STEREOE,
            rdchem.BondStereo.STEREOZ,
            rdchem.BondStereo.STEREONONE,
        ]
    
    def one_hot_encode(self, value: str, value_dict: List[float]) -> List[float]:
        len_desc = len(value_dict)
        desc_one_hot = np.zeros(len_desc)
        if value in value_dict:
            desc_one_hot[value_dict.index(value)] = 1.0
        else:
            desc_one_hot[-1] = 1.0

        return desc_one_hot
    
    def _cal_bond_weights(self, bond):
        bond_wt = 0
        if bond.GetBondType() == rdchem.BondType.SINGLE:
            bond_wt += 0.25
        if bond.GetBondType() == rdchem.BondType.DOUBLE:
            bond_wt += 0.5
        if bond.GetBondType() == rdchem.BondType.TRIPLE:
            bond_wt += 0.75
        if bond.GetBondType() == rdchem.BondType.AROMATIC:
            bond_wt += 1.0

        if bond.GetBeginAtom().GetSymbol() != 'C' or bond.GetEndAtom().GetSymbol() != 'C':
            bond_wt += 0.5

        return bond_wt  

    def _get_num_heavy_neighbors(self, atom):
        """Count non-hydrogen neighbors"""
        return sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() != 'H')

    def atom_encoder(self, atoms, mol):
        atoms_encoder = []
        
        # Pre-compute molecular descriptors
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except:
            pass
        
        # Compute per-atom contributions
        crippen_contribs = Crippen._GetAtomContribs(mol)
        tpsa_contribs = MolSurf._pyTPSAContribs(mol)
        labute_contribs = MolSurf._LabuteHelper(mol)
        estate_indices = EState.EStateIndices(mol)
        
        for i, atom in enumerate(atoms):
            atom_encoded = []
            
            # Basic atom features
            atom_encoded.extend(self.one_hot_encode(atom.GetSymbol(), self.ATOM_TYPE))
            
            num_heavy = self._get_num_heavy_neighbors(atom)
            atom_encoded.extend(self.one_hot_encode(num_heavy, self.NUM_HEAVY_NEIGHBORS))
            
            atom_encoded.extend(self.one_hot_encode(atom.GetTotalNumHs(), self.TOTAL_NUM_Hs))
            atom_encoded.extend(self.one_hot_encode(atom.GetFormalCharge(), self.FORMAL_CHARGE))
            atom_encoded.append(float(atom.GetIsAromatic()))
            atom_encoded.append(float(atom.IsInRing()))
            
            # Crippen logP and MR contributions
            logp_contrib, mr_contrib = crippen_contribs[i]
            atom_encoded.append(logp_contrib)
            atom_encoded.append(mr_contrib)
            
            # TPSA contribution
            tpsa_contrib = tpsa_contribs[i] if i < len(tpsa_contribs) else 0.0
            atom_encoded.append(tpsa_contrib)
            
            # Labute ASA contribution
            asa_contrib = labute_contribs[i] if i < len(labute_contribs) else 0.0
            atom_encoded.append(asa_contrib)
            
            # EState index
            estate_val = estate_indices[i] if i < len(estate_indices) else 0.0
            atom_encoded.append(estate_val)
            
            # Gasteiger charges
            try:
                charge_no_h = atom.GetDoubleProp('_GasteigerCharge')
                charge_no_h = charge_no_h if not np.isnan(charge_no_h) and not np.isinf(charge_no_h) else 0.0
            except:
                charge_no_h = 0.0
            atom_encoded.append(charge_no_h)
            
            # Gasteiger charge with implicit hydrogens
            try:
                charge_with_h = atom.GetDoubleProp('_GasteigerHCharge')
                charge_with_h = charge_with_h if not np.isnan(charge_with_h) and not np.isinf(charge_with_h) else 0.0
            except:
                charge_with_h = charge_no_h
            atom_encoded.append(charge_with_h)
            
            atoms_encoder.append(atom_encoded)
        
        return np.asarray(atoms_encoder, dtype=np.float32)
    
    def bond_encoder(self, bond):
        bond_encoded = []
        bond_encoded.extend(self.one_hot_encode(bond.GetBondType(), self.BOND_TYPE)) # Get Bond Type
        bond_encoded.extend(self.one_hot_encode(bond.GetStereo(), self.BOND_STEREO)) # Get Bond Stereo
        bond_encoded.append(float(bond.GetIsConjugated())) # Is conjugated bond
        bond_encoded.append(float(bond.IsInRing()))
        bond_encoded.append(float(bond.GetIsAromatic()))
        
        return bond_encoded

    def encoding_structure(self, smi: str, label: float) -> data:

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError('Invalid SMILES!!!')
        atoms = list(mol.GetAtoms())
        if len(atoms) == 0:
            raise ValueError('Molecule has no atom')
        
        bonds = list(mol.GetBonds())
        rings = list(mol.GetRingInfo().AtomRings())
            
        if len(bonds) == 0:
            embeddings = torch.tensor(self.atom_encoder(atom, mol), dtype=torch.float)
            edges_index = torch.zeros((2,0), dtype=torch.int64)
            edges_attr = torch.zero((0, 28), dtype=torch.float)
            edges_weights = torch.zeros((0,), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.float)
        
        edges_attr =  []
        edges_weights = []
        edges_index = [[], []]
        for bond in bonds:
            bond_features = self.bond_encoder(bond)
            bond_weight = self._cal_bond_weights(bond)

            if self.directed:
                edges_index[0].append(bond.GetBeginAtomIdx())
                edges_index[1].append(bond.GetEndAtomIdx())
                edges_attr.append(bond_features)
                edges_weights.append(bond_weight)
            else:
                edges_index[0].extend([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                edges_index[1].extend([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
                edges_attr.extend([bond_features, bond_features])
                edges_weights.extend([bond_weight, bond_weight])

        embeddings = torch.tensor(self.atom_encoder(atoms, mol), dtype=torch.float)
        edges_index = torch.tensor(edges_index, dtype=torch.int64)
        edges_attr = torch.tensor(edges_attr, dtype=torch.float)
        edges_weights = torch.tensor(edges_weights, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.float)

        return data.Data(
            x=embeddings,
            edge_index=edges_index,
            edge_attr=edges_attr,
            edge_weight=edges_weights,
            y = y,
        )

class StructureEncoderV4():
    def __init__(self, directed:bool=False):

        self.directed = directed

        self.ATOM_TYPE = [
            'H',
            'C',
            'N',
            'O',
            'F',
            'P',
            'S',
            'Cl',
            'Br',
            'I',
        ]

        self.HYBRIDIZATION = [
            rdchem.HybridizationType.SP,
            rdchem.HybridizationType.SP2,
            rdchem.HybridizationType.SP3,
            rdchem.HybridizationType.SP3D,
            rdchem.HybridizationType.SP3D2,]
        
        self.TOTAL_NUM_Hs = list(range(0,5))
        
        self.FORMAL_CHARGE = list(range(-2,3))
        
        self.TOTAL_DEGREE = list(range(0,6))
        
        self.RING_SIZE = list(range(3,9))

        self.AROMATIC = [True, False]

        self.CHIRAL_TYPE = [
            rdchem.ChiralType.CHI_UNSPECIFIED,
            rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            'UNKNOWN'
        ]
        
        self.BOND_TYPE = [
            rdchem.BondType.SINGLE,
            rdchem.BondType.DOUBLE,
            rdchem.BondType.TRIPLE,
            rdchem.BondType.AROMATIC,
            'UNKNOWN']
        
        self.BOND_STEREO = [
            rdchem.BondStereo.STEREOANY,
            rdchem.BondStereo.STEREOATROPCCW,
            rdchem.BondStereo.STEREOATROPCW,
            rdchem.BondStereo.STEREOCIS,
            rdchem.BondStereo.STEREOTRANS,
            rdchem.BondStereo.STEREOE,
            rdchem.BondStereo.STEREOZ,
            rdchem.BondStereo.STEREONONE,
            'UNKNOWN'
        ]
        
        self.BOND_NEIGHBOR = [
            ['C', 'C'],
            ['C', 'O'],
            ['C', 'N'],
            ['C', 'S'],
            ['C', 'F'],
            ['C', 'Cl'],
            ['C', 'P'],
            ['S', 'O'],
            ['P', 'O'],
            ['UNKNOWN', 'UNKNOWN']
        ]

    
    def one_hot_encode(self, value: str, value_dict: List[float]) -> List[float]:
        len_desc = len(value_dict)
        desc_one_hot = np.zeros(len_desc)
        if value in value_dict:
            desc_one_hot[value_dict.index(value)] = 1.0
        elif 'UNKNOWN' in value_dict:
            desc_one_hot[value_dict.index('UNKNOWN')] = 1.0
        else:
            desc_one_hot[-1] = 1.0

        return desc_one_hot
    
    def _get_ring_substitution_position(self, atom, mol) -> List[int]:

        """
        Returns [is_substituted_ring_atom, ortho, meta, para]:
        - For ring atoms WITH substituents: [1, 0, 0, 0]
        - For ring atoms WITHOUT substituents: [0, 0, 0, 0]
        - For substituent atoms (non-ring): [0, ortho, meta, para] 
        
        For fused rings: atoms from OTHER rings are treated as substituents
        when calculating ortho/meta/para relationships.
        """
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        
        if atom.IsInRing():

            for ring in atom_rings:
                if atom.GetIdx() in ring:
                    ring_atoms = list(ring)
                    

                    has_substituent = any(
                        neighbor.GetSymbol() != 'H' and neighbor.GetIdx() not in ring_atoms
                        for neighbor in atom.GetNeighbors()
                    )
                    
                    if has_substituent:
                        return [1.0, 0.0, 0.0, 0.0]
                    else:
                        return [0.0, 0.0, 0.0, 0.0]
        
        else:

            parent_ring_atom = None
            parent_rings = []
            
            for neighbor in atom.GetNeighbors():
                if neighbor.IsInRing():
                    parent_ring_atom = neighbor

                    for ring in atom_rings:
                        if neighbor.GetIdx() in ring:
                            parent_rings.append(list(ring))
                    break
            
            if not parent_ring_atom or not parent_rings:
                return [0.0, 0.0, 0.0, 0.0]
            
            has_ortho = False
            has_meta = False  
            has_para = False
            
            for parent_ring in parent_rings:
                ring_size = len(parent_ring)
                parent_pos = parent_ring.index(parent_ring_atom.GetIdx())
                
                substituted_positions = []
                for i, ring_atom_idx in enumerate(parent_ring):
                    if ring_atom_idx == parent_ring_atom.GetIdx():
                        continue
                        
                    ring_atom = mol.GetAtomWithIdx(ring_atom_idx)

                    has_other_substituent = any(
                        n.GetSymbol() != 'H' and n.GetIdx() not in parent_ring
                        for n in ring_atom.GetNeighbors()
                    )
                    if has_other_substituent:
                        substituted_positions.append(i)

                for sub_pos in substituted_positions:
                    clockwise_dist = (sub_pos - parent_pos) % ring_size
                    counter_clockwise_dist = (parent_pos - sub_pos) % ring_size
                    min_dist = min(clockwise_dist, counter_clockwise_dist)
                    
                    if min_dist == 1:
                        has_ortho = True
                    elif min_dist == 2:
                        has_meta = True
                    elif ring_size == 6 and min_dist == 3:
                        has_para = True
            
            return [0.0, float(has_ortho), float(has_meta), float(has_para)]
        
        return [0.0, 0.0, 0.0, 0.0]

    
    def _get_max_valence(self, atomic_num):
        max_valence = {
            1: 1,
            6: 4, 
            7: 3,
            8: 2,
            9: 1,
            15: 5,
            16: 6,
            17: 1,
            35: 1,
            53: 1
        }
        return max_valence.get(atomic_num, 8)
    
    def _is_bond_donor(self, atom):
        return (atom.GetAtomicNum() in [7, 8, 16] and atom.GetTotalNumHs() > 0)
    
    def _is_bond_acceptor(self, atom):
        return (atom.GetAtomicNum() in [7, 8, 16] and atom.GetTotalValence() - atom.GetTotalNumHs() < self._get_max_valence(atom.GetAtomicNum()))

    def _check_bond_in_same_ring(self, atom_1_idx, atom_2_idx, rings: List[Chem.Mol]) -> float:
        for ring in rings:
            if atom_1_idx in ring and atom_2_idx in ring:
                return True
        return False

    def _cal_bond_weights(self, bond):
        bond_wt = 0
        if bond.GetBondType() == rdchem.BondType.SINGLE:
            bond_wt += 0.25
        if bond.GetBondType() == rdchem.BondType.DOUBLE:
            bond_wt += 0.5
        if bond.GetBondType() == rdchem.BondType.TRIPLE:
            bond_wt += 0.75
        if bond.GetBondType() == rdchem.BondType.AROMATIC:
            bond_wt += 1.0

        if bond.GetBeginAtom().GetSymbol() != 'C' or bond.GetEndAtom().GetSymbol() != 'C':
            bond_wt += 0.5

        return bond_wt  

    def atom_encoder(self, atoms, mol):
        atoms_encoder = []
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol) # Gasteiger charges compute
        except:
            pass
        for atom in atoms:
            atom_encoded = []
            atom_type = atom.GetSymbol()
            atom_encoded.extend(self.one_hot_encode(atom_type, self.ATOM_TYPE)) # Atom symbol C, H, O, N, ...
            atom_encoded.extend(self.one_hot_encode(atom.GetHybridization(), self.HYBRIDIZATION)) # Atom hybridization type
            atom_encoded.extend(self.one_hot_encode(atom.GetTotalNumHs(), self.TOTAL_NUM_Hs)) # Number of Hydrogens
            atom_encoded.extend(self.one_hot_encode(atom.GetFormalCharge(), self.FORMAL_CHARGE))# Total charge
            atom_encoded.extend(self.one_hot_encode(atom.GetTotalDegree(), self.TOTAL_DEGREE)) # Total degree
            atom_encoded.append(atom.GetTotalValence() / 8.0) # Total valence
            atom_encoded.append(float(atom.GetIsAromatic())) # Atom in aromatic ring
            atom_encoded.append(float(atom.IsInRing())) # Atom in ring
            atom_encoded.extend(self.one_hot_encode(atom.GetChiralTag(), self.CHIRAL_TYPE)) # Get Chiral Type for atom
            # Calculate Gasteiger charges for atom
            try:
                atom_charge = atom.GetDoubleProp('_GasteigerCharge')
                atom_charge = atom_charge if not np.isnan(atom_charge) and not np.isinf(atom_charge) else 0.0
            except:
                atom_charge = 0.0
            atom_encoded.append(atom_charge)
            # carbon chain > [is in chain, if yes is terminal or middle]
            if atom.GetSymbol() == 'C':
                carbon_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol()=='C' and not n.IsInRing()]
                is_in_chain = 1 <= len(carbon_neighbors) <= 2
                atom_encoded.append(float(is_in_chain))
                is_terminal = len(carbon_neighbors) == 1
                atom_encoded.append(float(is_terminal))

            else:
                atom_encoded.extend([0.0, 0.0])

            # substitution on aromatic ring [ortho, meta, para]
            atom_encoded.extend(self._get_ring_substitution_position(atom, mol))
            atom_encoded.append(float(self._is_bond_donor(atom))) # Is bond donor
            atom_encoded.append(float(self._is_bond_acceptor(atom))) # Is bond acceptor

            atoms_encoder.append(atom_encoded)

        return np.asarray(atoms_encoder, dtype=np.float32)
    
    def bond_encoder(self, bond, rings):
        bond_encoded = []
        bond_encoded.extend(self.one_hot_encode(bond.GetBondType(), self.BOND_TYPE)) # Get Bond Type
        bond_encoded.extend(self.one_hot_encode(bond.GetStereo(), self.BOND_STEREO)) # Get Bond Stereo
        bond_encoded.append(float(bond.GetIsConjugated())) # Is conjugated bond
        bond_encoded.append(float(bond.IsInRing()))
        bond_encoded.append(float(bond.GetIsAromatic()))
        # Encode bond between carbon or heteroatom
        neighbor = [bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]
        neighbor.sort()
        if neighbor not in [pair for pair in self.BOND_NEIGHBOR if pair != ['UNKNOWN', 'UNKNOWN']]:
            neighbor = ['UNKNOWN', 'UNKNOWN']
        bond_encoded.extend(self.one_hot_encode(neighbor, self.BOND_NEIGHBOR))
        # Encode if two atom is in the same ring
        bond_encoded.append(self._check_bond_in_same_ring(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), rings))

        return bond_encoded

    def encoding_structure(self, smi: str, label: float) -> data:


        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError('Invalid SMILES!!!')
        atoms = list(mol.GetAtoms())
        if len(atoms) == 0:
            raise ValueError('Molecule has no atom')
        
        bonds = list(mol.GetBonds())
        rings = list(mol.GetRingInfo().AtomRings())
            
        if len(bonds) == 0:
            embeddings = torch.tensor(self.atom_encoder(atom, mol), dtype=torch.float)
            edges_index = torch.zeros((2,0), dtype=torch.int64)
            edges_attr = torch.zero((0, 28), dtype=torch.float)
            edges_weights = torch.zeros((0,), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.float)
        
        edges_attr =  []
        edges_weights = []
        edges_index = [[], []]
        for bond in bonds:
            bond_features = self.bond_encoder(bond, rings)
            bond_weight = self._cal_bond_weights(bond)

            if self.directed:
                edges_index[0].append(bond.GetBeginAtomIdx())
                edges_index[1].append(bond.GetEndAtomIdx())
                edges_attr.append(bond_features)
                edges_weights.append(bond_weight)
            else:
                edges_index[0].extend([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                edges_index[1].extend([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
                edges_attr.extend([bond_features, bond_features])
                edges_weights.extend([bond_weight, bond_weight])

        embeddings = torch.tensor(self.atom_encoder(atoms, mol), dtype=torch.float)
        edges_index = torch.tensor(edges_index, dtype=torch.int64)
        edges_attr = torch.tensor(edges_attr, dtype=torch.float)
        edges_weights = torch.tensor(edges_weights, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.float)

        return data.Data(
            x=embeddings,
            edge_index=edges_index,
            edge_attr=edges_attr,
            edge_weight=edges_weights,
            y = y,
        )


class StructureEncoderV5():
    def __init__(self, directed:bool=False):

        self.directed = directed

        self.ATOM_TYPE = [
            'H',
            'C',
            'N',
            'O',
            'F',
            'P',
            'S',
            'Cl',
            'Br',
            'I',
        ]

        self.HYBRIDIZATION = [
            rdchem.HybridizationType.SP,
            rdchem.HybridizationType.SP2,
            rdchem.HybridizationType.SP3,
            rdchem.HybridizationType.SP3D,
            rdchem.HybridizationType.SP3D2,]
        
        self.TOTAL_NUM_Hs = list(range(0,5))
        
        self.FORMAL_CHARGE = list(range(-2,3))
        
        self.TOTAL_DEGREE = list(range(0,6))
        
        self.RING_SIZE = list(range(3,9))

        self.AROMATIC = [True, False]

        self.CHIRAL_TYPE = [
            rdchem.ChiralType.CHI_UNSPECIFIED,
            rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            'UNKNOWN'
        ]
        
        self.BOND_TYPE = [
            rdchem.BondType.SINGLE,
            rdchem.BondType.DOUBLE,
            rdchem.BondType.TRIPLE,
            rdchem.BondType.AROMATIC,
            'UNKNOWN']
        
        self.BOND_STEREO = [
            rdchem.BondStereo.STEREOANY,
            rdchem.BondStereo.STEREOATROPCCW,
            rdchem.BondStereo.STEREOATROPCW,
            rdchem.BondStereo.STEREOCIS,
            rdchem.BondStereo.STEREOTRANS,
            rdchem.BondStereo.STEREOE,
            rdchem.BondStereo.STEREOZ,
            rdchem.BondStereo.STEREONONE,
            'UNKNOWN'
        ]
        
        self.BOND_NEIGHBOR = [
            ['C', 'C'],
            ['C', 'O'],
            ['C', 'N'],
            ['C', 'S'],
            ['C', 'F'],
            ['C', 'Cl'],
            ['C', 'P'],
            ['S', 'O'],
            ['P', 'O'],
            ['UNKNOWN', 'UNKNOWN']
        ]

    
    def one_hot_encode(self, value: str, value_dict: List[float]) -> List[float]:
        len_desc = len(value_dict)
        desc_one_hot = np.zeros(len_desc)
        if value in value_dict:
            desc_one_hot[value_dict.index(value)] = 1.0
        elif 'UNKNOWN' in value_dict:
            desc_one_hot[value_dict.index('UNKNOWN')] = 1.0
        else:
            desc_one_hot[-1] = 1.0

        return desc_one_hot
    
    def _get_ring_substitution_position(self, atom, mol) -> List[int]:

        """
        Returns [is_substituted_ring_atom, ortho, meta, para]:
        - For ring atoms WITH substituents: [1, 0, 0, 0]
        - For ring atoms WITHOUT substituents: [0, 0, 0, 0]
        - For substituent atoms (non-ring): [0, ortho, meta, para] 
        
        For fused rings: atoms from OTHER rings are treated as substituents
        when calculating ortho/meta/para relationships.
        """
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        
        if atom.IsInRing():

            for ring in atom_rings:
                if atom.GetIdx() in ring:
                    ring_atoms = list(ring)
                    

                    has_substituent = any(
                        neighbor.GetSymbol() != 'H' and neighbor.GetIdx() not in ring_atoms
                        for neighbor in atom.GetNeighbors()
                    )
                    
                    if has_substituent:
                        return [1.0, 0.0, 0.0, 0.0]
                    else:
                        return [0.0, 0.0, 0.0, 0.0]
        
        else:

            parent_ring_atom = None
            parent_rings = []
            
            for neighbor in atom.GetNeighbors():
                if neighbor.IsInRing():
                    parent_ring_atom = neighbor

                    for ring in atom_rings:
                        if neighbor.GetIdx() in ring:
                            parent_rings.append(list(ring))
                    break
            
            if not parent_ring_atom or not parent_rings:
                return [0.0, 0.0, 0.0, 0.0]
            
            has_ortho = False
            has_meta = False  
            has_para = False
            
            for parent_ring in parent_rings:
                ring_size = len(parent_ring)
                parent_pos = parent_ring.index(parent_ring_atom.GetIdx())
                
                substituted_positions = []
                for i, ring_atom_idx in enumerate(parent_ring):
                    if ring_atom_idx == parent_ring_atom.GetIdx():
                        continue
                        
                    ring_atom = mol.GetAtomWithIdx(ring_atom_idx)

                    has_other_substituent = any(
                        n.GetSymbol() != 'H' and n.GetIdx() not in parent_ring
                        for n in ring_atom.GetNeighbors()
                    )
                    if has_other_substituent:
                        substituted_positions.append(i)

                for sub_pos in substituted_positions:
                    clockwise_dist = (sub_pos - parent_pos) % ring_size
                    counter_clockwise_dist = (parent_pos - sub_pos) % ring_size
                    min_dist = min(clockwise_dist, counter_clockwise_dist)
                    
                    if min_dist == 1:
                        has_ortho = True
                    elif min_dist == 2:
                        has_meta = True
                    elif ring_size == 6 and min_dist == 3:
                        has_para = True
            
            return [0.0, float(has_ortho), float(has_meta), float(has_para)]
        
        return [0.0, 0.0, 0.0, 0.0]

    
    def _get_max_valence(self, atomic_num):
        max_valence = {
            1: 1,
            6: 4, 
            7: 3,
            8: 2,
            9: 1,
            15: 5,
            16: 6,
            17: 1,
            35: 1,
            53: 1
        }
        return max_valence.get(atomic_num, 8)
    
    def _is_bond_donor(self, atom):
        return (atom.GetAtomicNum() in [7, 8, 16] and atom.GetTotalNumHs() > 0)
    
    def _is_bond_acceptor(self, atom):
        return (atom.GetAtomicNum() in [7, 8, 16] and atom.GetTotalValence() - atom.GetTotalNumHs() < self._get_max_valence(atom.GetAtomicNum()))

    def _check_bond_in_same_ring(self, atom_1_idx, atom_2_idx, rings: List[Chem.Mol]) -> float:
        for ring in rings:
            if atom_1_idx in ring and atom_2_idx in ring:
                return True
        return False

    def _cal_bond_weights(self, bond):
        bond_wt = 0
        if bond.GetBondType() == rdchem.BondType.SINGLE:
            bond_wt += 0.25
        if bond.GetBondType() == rdchem.BondType.DOUBLE:
            bond_wt += 0.5
        if bond.GetBondType() == rdchem.BondType.TRIPLE:
            bond_wt += 0.75
        if bond.GetBondType() == rdchem.BondType.AROMATIC:
            bond_wt += 1.0

        if bond.GetBeginAtom().GetSymbol() != 'C' or bond.GetEndAtom().GetSymbol() != 'C':
            bond_wt += 0.5

        return bond_wt  

    def atom_encoder(self, atoms, mol):
        atoms_encoder = []
        
        # Pre-compute molecular descriptors (from V3)
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except:
            pass
        
        # Compute per-atom contributions (from V3)
        crippen_contribs = Crippen._GetAtomContribs(mol)
        tpsa_contribs = MolSurf._pyTPSAContribs(mol)
        labute_contribs = MolSurf._LabuteHelper(mol)
        estate_indices = EState.EStateIndices(mol)
        
        for i, atom in enumerate(atoms):
            atom_encoded = []
            atom_type = atom.GetSymbol()
            
            # V4 features
            atom_encoded.extend(self.one_hot_encode(atom_type, self.ATOM_TYPE))
            atom_encoded.extend(self.one_hot_encode(atom.GetHybridization(), self.HYBRIDIZATION))
            atom_encoded.extend(self.one_hot_encode(atom.GetTotalNumHs(), self.TOTAL_NUM_Hs))
            atom_encoded.extend(self.one_hot_encode(atom.GetFormalCharge(), self.FORMAL_CHARGE))
            atom_encoded.extend(self.one_hot_encode(atom.GetTotalDegree(), self.TOTAL_DEGREE))
            atom_encoded.append(atom.GetTotalValence() / 8.0)
            atom_encoded.append(float(atom.GetIsAromatic()))
            atom_encoded.append(float(atom.IsInRing()))
            atom_encoded.extend(self.one_hot_encode(atom.GetChiralTag(), self.CHIRAL_TYPE))
            
            # Gasteiger charge (V4 version)
            try:
                atom_charge = atom.GetDoubleProp('_GasteigerCharge')
                atom_charge = atom_charge if not np.isnan(atom_charge) and not np.isinf(atom_charge) else 0.0
            except:
                atom_charge = 0.0
            atom_encoded.append(atom_charge)
            
            # Carbon chain features (from V4)
            if atom.GetSymbol() == 'C':
                carbon_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol()=='C' and not n.IsInRing()]
                is_in_chain = 1 <= len(carbon_neighbors) <= 2
                atom_encoded.append(float(is_in_chain))
                is_terminal = len(carbon_neighbors) == 1
                atom_encoded.append(float(is_terminal))
            else:
                atom_encoded.extend([0.0, 0.0])

            # Ring substitution features (from V4)
            atom_encoded.extend(self._get_ring_substitution_position(atom, mol))
            atom_encoded.append(float(self._is_bond_donor(atom)))
            atom_encoded.append(float(self._is_bond_acceptor(atom)))
            
            # V3 molecular descriptor contributions
            logp_contrib, mr_contrib = crippen_contribs[i]
            atom_encoded.append(logp_contrib)
            atom_encoded.append(mr_contrib)
            
            tpsa_contrib = tpsa_contribs[i] if i < len(tpsa_contribs) else 0.0
            atom_encoded.append(tpsa_contrib)
            
            asa_contrib = labute_contribs[i] if i < len(labute_contribs) else 0.0
            atom_encoded.append(asa_contrib)
            
            estate_val = estate_indices[i] if i < len(estate_indices) else 0.0
            atom_encoded.append(estate_val)
            
            # Gasteiger charge with implicit hydrogens (from V3)
            try:
                charge_with_h = atom.GetDoubleProp('_GasteigerHCharge')
                charge_with_h = charge_with_h if not np.isnan(charge_with_h) and not np.isinf(charge_with_h) else 0.0
            except:
                charge_with_h = atom_charge
            atom_encoded.append(charge_with_h)

            atoms_encoder.append(atom_encoded)

        return np.asarray(atoms_encoder, dtype=np.float32)
    
    def bond_encoder(self, bond, rings):
        bond_encoded = []
        bond_encoded.extend(self.one_hot_encode(bond.GetBondType(), self.BOND_TYPE)) # Get Bond Type
        bond_encoded.extend(self.one_hot_encode(bond.GetStereo(), self.BOND_STEREO)) # Get Bond Stereo
        bond_encoded.append(float(bond.GetIsConjugated())) # Is conjugated bond
        bond_encoded.append(float(bond.IsInRing()))
        bond_encoded.append(float(bond.GetIsAromatic()))
        # Encode bond between carbon or heteroatom
        neighbor = [bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]
        neighbor.sort()
        if neighbor not in [pair for pair in self.BOND_NEIGHBOR if pair != ['UNKNOWN', 'UNKNOWN']]:
            neighbor = ['UNKNOWN', 'UNKNOWN']
        bond_encoded.extend(self.one_hot_encode(neighbor, self.BOND_NEIGHBOR))
        # Encode if two atom is in the same ring
        bond_encoded.append(self._check_bond_in_same_ring(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), rings))

        return bond_encoded

    def encoding_structure(self, smi: str, label: float) -> data:


        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError('Invalid SMILES!!!')
        atoms = list(mol.GetAtoms())
        if len(atoms) == 0:
            raise ValueError('Molecule has no atom')
        
        bonds = list(mol.GetBonds())
        rings = list(mol.GetRingInfo().AtomRings())
            
        if len(bonds) == 0:
            embeddings = torch.tensor(self.atom_encoder(atom, mol), dtype=torch.float)
            edges_index = torch.zeros((2,0), dtype=torch.int64)
            edges_attr = torch.zero((0, 28), dtype=torch.float)
            edges_weights = torch.zeros((0,), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.float)
        
        edges_attr =  []
        edges_weights = []
        edges_index = [[], []]
        for bond in bonds:
            bond_features = self.bond_encoder(bond, rings)
            bond_weight = self._cal_bond_weights(bond)

            if self.directed:
                edges_index[0].append(bond.GetBeginAtomIdx())
                edges_index[1].append(bond.GetEndAtomIdx())
                edges_attr.append(bond_features)
                edges_weights.append(bond_weight)
            else:
                edges_index[0].extend([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                edges_index[1].extend([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
                edges_attr.extend([bond_features, bond_features])
                edges_weights.extend([bond_weight, bond_weight])

        embeddings = torch.tensor(self.atom_encoder(atoms, mol), dtype=torch.float)
        edges_index = torch.tensor(edges_index, dtype=torch.int64)
        edges_attr = torch.tensor(edges_attr, dtype=torch.float)
        edges_weights = torch.tensor(edges_weights, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.float)

        return data.Data(
            x=embeddings,
            edge_index=edges_index,
            edge_attr=edges_attr,
            edge_weight=edges_weights,
            y = y,
        )

class StructureEncoderV5():
    def __init__(self, directed:bool=False):

        self.directed = directed

        self.ATOM_TYPE = [
            'H',
            'C',
            'N',
            'O',
            'F',
            'P',
            'S',
            'Cl',
            'Br',
            'I',
        ]

        self.HYBRIDIZATION = [
            rdchem.HybridizationType.SP,
            rdchem.HybridizationType.SP2,
            rdchem.HybridizationType.SP3,
            rdchem.HybridizationType.SP3D,
            rdchem.HybridizationType.SP3D2,]
        
        self.TOTAL_NUM_Hs = list(range(0,5))
        
        self.FORMAL_CHARGE = list(range(-2,3))
        
        self.TOTAL_DEGREE = list(range(0,6))
        
        self.RING_SIZE = list(range(3,9))

        self.AROMATIC = [True, False]

        self.CHIRAL_TYPE = [
            rdchem.ChiralType.CHI_UNSPECIFIED,
            rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            'UNKNOWN'
        ]
        
        self.BOND_TYPE = [
            rdchem.BondType.SINGLE,
            rdchem.BondType.DOUBLE,
            rdchem.BondType.TRIPLE,
            rdchem.BondType.AROMATIC,
            'UNKNOWN']
        
        self.BOND_STEREO = [
            rdchem.BondStereo.STEREOANY,
            rdchem.BondStereo.STEREOATROPCCW,
            rdchem.BondStereo.STEREOATROPCW,
            rdchem.BondStereo.STEREOCIS,
            rdchem.BondStereo.STEREOTRANS,
            rdchem.BondStereo.STEREOE,
            rdchem.BondStereo.STEREOZ,
            rdchem.BondStereo.STEREONONE,
            'UNKNOWN'
        ]
        
        self.BOND_NEIGHBOR = [
            ['C', 'C'],
            ['C', 'O'],
            ['C', 'N'],
            ['C', 'S'],
            ['C', 'F'],
            ['C', 'Cl'],
            ['C', 'P'],
            ['S', 'O'],
            ['P', 'O'],
            ['UNKNOWN', 'UNKNOWN']
        ]

    
    def one_hot_encode(self, value: str, value_dict: List[float]) -> List[float]:
        len_desc = len(value_dict)
        desc_one_hot = np.zeros(len_desc)
        if value in value_dict:
            desc_one_hot[value_dict.index(value)] = 1.0
        elif 'UNKNOWN' in value_dict:
            desc_one_hot[value_dict.index('UNKNOWN')] = 1.0
        else:
            desc_one_hot[-1] = 1.0

        return desc_one_hot
    
    def _get_ring_substitution_position(self, atom, mol) -> List[int]:

        """
        Returns [is_substituted_ring_atom, ortho, meta, para]:
        - For ring atoms WITH substituents: [1, 0, 0, 0]
        - For ring atoms WITHOUT substituents: [0, 0, 0, 0]
        - For substituent atoms (non-ring): [0, ortho, meta, para] 
        
        For fused rings: atoms from OTHER rings are treated as substituents
        when calculating ortho/meta/para relationships.
        """
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        
        if atom.IsInRing():

            for ring in atom_rings:
                if atom.GetIdx() in ring:
                    ring_atoms = list(ring)
                    

                    has_substituent = any(
                        neighbor.GetSymbol() != 'H' and neighbor.GetIdx() not in ring_atoms
                        for neighbor in atom.GetNeighbors()
                    )
                    
                    if has_substituent:
                        return [1.0, 0.0, 0.0, 0.0]
                    else:
                        return [0.0, 0.0, 0.0, 0.0]
        
        else:

            parent_ring_atom = None
            parent_rings = []
            
            for neighbor in atom.GetNeighbors():
                if neighbor.IsInRing():
                    parent_ring_atom = neighbor

                    for ring in atom_rings:
                        if neighbor.GetIdx() in ring:
                            parent_rings.append(list(ring))
                    break
            
            if not parent_ring_atom or not parent_rings:
                return [0.0, 0.0, 0.0, 0.0]
            
            has_ortho = False
            has_meta = False  
            has_para = False
            
            for parent_ring in parent_rings:
                ring_size = len(parent_ring)
                parent_pos = parent_ring.index(parent_ring_atom.GetIdx())
                
                substituted_positions = []
                for i, ring_atom_idx in enumerate(parent_ring):
                    if ring_atom_idx == parent_ring_atom.GetIdx():
                        continue
                        
                    ring_atom = mol.GetAtomWithIdx(ring_atom_idx)

                    has_other_substituent = any(
                        n.GetSymbol() != 'H' and n.GetIdx() not in parent_ring
                        for n in ring_atom.GetNeighbors()
                    )
                    if has_other_substituent:
                        substituted_positions.append(i)

                for sub_pos in substituted_positions:
                    clockwise_dist = (sub_pos - parent_pos) % ring_size
                    counter_clockwise_dist = (parent_pos - sub_pos) % ring_size
                    min_dist = min(clockwise_dist, counter_clockwise_dist)
                    
                    if min_dist == 1:
                        has_ortho = True
                    elif min_dist == 2:
                        has_meta = True
                    elif ring_size == 6 and min_dist == 3:
                        has_para = True
            
            return [1.0, float(has_ortho), float(has_meta), float(has_para)]
        
        return [0.0, 0.0, 0.0, 0.0]

    
    def _get_max_valence(self, atomic_num):
        max_valence = {
            1: 1,
            6: 4, 
            7: 3,
            8: 2,
            9: 1,
            15: 5,
            16: 6,
            17: 1,
            35: 1,
            53: 1
        }
        return max_valence.get(atomic_num, 8)
    
    def _is_bond_donor(self, atom):
        return (atom.GetAtomicNum() in [7, 8, 16] and atom.GetTotalNumHs() > 0)
    
    def _is_bond_acceptor(self, atom):
        return (atom.GetAtomicNum() in [7, 8, 16] and atom.GetTotalValence() - atom.GetTotalNumHs() < self._get_max_valence(atom.GetAtomicNum()))

    def _check_bond_in_same_ring(self, atom_1_idx, atom_2_idx, rings: List[Chem.Mol]) -> float:
        for ring in rings:
            if atom_1_idx in ring and atom_2_idx in ring:
                return True
        return False

    def _cal_bond_weights(self, bond):
        bond_wt = 0
        if bond.GetBondType() == rdchem.BondType.SINGLE:
            bond_wt += 0.25
        if bond.GetBondType() == rdchem.BondType.DOUBLE:
            bond_wt += 0.5
        if bond.GetBondType() == rdchem.BondType.TRIPLE:
            bond_wt += 0.75
        if bond.GetBondType() == rdchem.BondType.AROMATIC:
            bond_wt += 1.0

        if bond.GetBeginAtom().GetSymbol() != 'C' or bond.GetEndAtom().GetSymbol() != 'C':
            bond_wt += 0.5

        return bond_wt  

    def atom_encoder(self, atoms, mol):
        atoms_encoder = []
        
        # Pre-compute molecular descriptors (from V3)
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except:
            pass
        
        # Compute per-atom contributions (from V3)
        crippen_contribs = Crippen._GetAtomContribs(mol)
        tpsa_contribs = MolSurf._pyTPSAContribs(mol)
        labute_contribs = MolSurf._LabuteHelper(mol)
        estate_indices = EState.EStateIndices(mol)
        
        for i, atom in enumerate(atoms):
            atom_encoded = []
            atom_type = atom.GetSymbol()
            
            # V4 features
            atom_encoded.extend(self.one_hot_encode(atom_type, self.ATOM_TYPE))
            atom_encoded.extend(self.one_hot_encode(atom.GetHybridization(), self.HYBRIDIZATION))
            atom_encoded.extend(self.one_hot_encode(atom.GetTotalNumHs(), self.TOTAL_NUM_Hs))
            atom_encoded.extend(self.one_hot_encode(atom.GetFormalCharge(), self.FORMAL_CHARGE))
            atom_encoded.extend(self.one_hot_encode(atom.GetTotalDegree(), self.TOTAL_DEGREE))
            atom_encoded.append(atom.GetTotalValence() / 8.0)
            atom_encoded.append(float(atom.GetIsAromatic()))
            atom_encoded.append(float(atom.IsInRing()))
            atom_encoded.extend(self.one_hot_encode(atom.GetChiralTag(), self.CHIRAL_TYPE))
            
            # Gasteiger charge (V4 version)
            try:
                atom_charge = atom.GetDoubleProp('_GasteigerCharge')
                atom_charge = atom_charge if not np.isnan(atom_charge) and not np.isinf(atom_charge) else 0.0
            except:
                atom_charge = 0.0
            atom_encoded.append(atom_charge)
            
            # Carbon chain features (from V4)
            if atom.GetSymbol() == 'C':
                carbon_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol()=='C' and not n.IsInRing()]
                is_in_chain = 1 <= len(carbon_neighbors) <= 2
                atom_encoded.append(float(is_in_chain))
                is_terminal = len(carbon_neighbors) == 1
                atom_encoded.append(float(is_terminal))
            else:
                atom_encoded.extend([0.0, 0.0])

            # Ring substitution features (from V4)
            atom_encoded.extend(self._get_ring_substitution_position(atom, mol))
            atom_encoded.append(float(self._is_bond_donor(atom)))
            atom_encoded.append(float(self._is_bond_acceptor(atom)))
            
            # V3 molecular descriptor contributions
            logp_contrib, mr_contrib = crippen_contribs[i]
            atom_encoded.append(logp_contrib)
            atom_encoded.append(mr_contrib)
            
            tpsa_contrib = tpsa_contribs[i] if i < len(tpsa_contribs) else 0.0
            atom_encoded.append(tpsa_contrib)
            
            asa_contrib = labute_contribs[i] if i < len(labute_contribs) else 0.0
            atom_encoded.append(asa_contrib)
            
            estate_val = estate_indices[i] if i < len(estate_indices) else 0.0
            atom_encoded.append(estate_val)
            
            # Gasteiger charge with implicit hydrogens (from V3)
            try:
                charge_with_h = atom.GetDoubleProp('_GasteigerHCharge')
                charge_with_h = charge_with_h if not np.isnan(charge_with_h) and not np.isinf(charge_with_h) else 0.0
            except:
                charge_with_h = atom_charge
            atom_encoded.append(charge_with_h)

            atoms_encoder.append(atom_encoded)

        return np.asarray(atoms_encoder, dtype=np.float32)
    
    def bond_encoder(self, bond, rings):
        bond_encoded = []
        bond_encoded.extend(self.one_hot_encode(bond.GetBondType(), self.BOND_TYPE))
        bond_encoded.extend(self.one_hot_encode(bond.GetStereo(), self.BOND_STEREO))
        bond_encoded.append(float(bond.GetIsConjugated()))
        bond_encoded.append(float(bond.IsInRing()))
        bond_encoded.append(float(bond.GetIsAromatic()))
        
        neighbor = [bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]
        neighbor.sort()
        if neighbor not in [pair for pair in self.BOND_NEIGHBOR if pair != ['UNKNOWN', 'UNKNOWN']]:
            neighbor = ['UNKNOWN', 'UNKNOWN']
        bond_encoded.extend(self.one_hot_encode(neighbor, self.BOND_NEIGHBOR))
        
        bond_encoded.append(self._check_bond_in_same_ring(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), rings))

        return bond_encoded

    def encoding_structure(self, smi: str, label: float) -> data:

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError('Invalid SMILES!!!')
        atoms = list(mol.GetAtoms())
        if len(atoms) == 0:
            raise ValueError('Molecule has no atom')
        
        bonds = list(mol.GetBonds())
        rings = list(mol.GetRingInfo().AtomRings())
            
        if len(bonds) == 0:
            embeddings = torch.tensor(self.atom_encoder(atoms, mol), dtype=torch.float)
            edges_index = torch.zeros((2,0), dtype=torch.int64)
            edges_attr = torch.zeros((0, 25), dtype=torch.float)
            edges_weights = torch.zeros((0,), dtype=torch.float)
            y = torch.tensor(label, dtype=torch.float)
        
        edges_attr =  []
        edges_weights = []
        edges_index = [[], []]
        for bond in bonds:
            bond_features = self.bond_encoder(bond, rings)
            bond_weight = self._cal_bond_weights(bond)

            if self.directed:
                edges_index[0].append(bond.GetBeginAtomIdx())
                edges_index[1].append(bond.GetEndAtomIdx())
                edges_attr.append(bond_features)
                edges_weights.append(bond_weight)
            else:
                edges_index[0].extend([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                edges_index[1].extend([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
                edges_attr.extend([bond_features, bond_features])
                edges_weights.extend([bond_weight, bond_weight])

        embeddings = torch.tensor(self.atom_encoder(atoms, mol), dtype=torch.float)
        edges_index = torch.tensor(edges_index, dtype=torch.int64)
        edges_attr = torch.tensor(edges_attr, dtype=torch.float)
        edges_weights = torch.tensor(edges_weights, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.float)

        return data.Data(
            x=embeddings,
            edge_index=edges_index,
            edge_attr=edges_attr,
            edge_weight=edges_weights,
            y = y,
        )