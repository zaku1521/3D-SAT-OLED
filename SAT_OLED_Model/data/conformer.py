# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# This file has been modified  based on the original version.
#  Streamlined dependencies by removing multiprocessing and progress visualization modules (Pool, tqdm)
# - Decoupled external logger and configuration dependencies for enhanced modularity
# - Simplified weight management by omitting weight download and directory handling logic
# - Refactored conformer generation flow by replacing multiprocessing-based transform with a more concise approach
# - Consolidated coordinate generation by removing inner_smi2coords function to centralize processing
# - Renamed coords2unimol to coords2molrep to better represent updated functionality
# - Adjusted dictionary loading mechanism from dynamic weight directory to a fixed relative path for consistency
# - Optimized feature initialization by removing data_type-dependent dictionary selection logic
from __future__ import absolute_import, division, print_function

import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import warnings
from scipy.spatial import distance_matrix

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action='ignore')
from .dictionary import Dictionary


class ConformerGen(object):
    '''
    This class designed to generate conformers for molecules represented as SMILES strings using provided parameters and configurations. The `transform` method uses multiprocessing to speed up the conformer generation process.
    '''

    def __init__(self, **params):
        """
        Initializes the neural network model based on the provided model name and parameters.

        :param model_name: (str) The name of the model to initialize.
        :param params: Additional parameters for model configuration.

        :return: An instance of the specified neural network model.
        :raises ValueError: If the model name is not recognized.
        """
        self._init_features(**params)

    def _init_features(self, **params):
        """
        Initializes the features of the ConformerGen object based on provided parameters.

        :param params: Arbitrary keyword arguments for feature configuration.
                       These can include the random seed, maximum number of atoms, data type,
                       generation method, generation mode, and whether to remove hydrogens.
        """
        self.seed = params.get('seed', 42)
        self.max_atoms = params.get('max_atoms', 256)
        self.data_type = params.get('data_type', 'molecule')
        self.method = params.get('method', 'rdkit_random')
        self.mode = params.get('mode', 'fast')
        self.remove_hs = params.get('remove_hs', False)
        self.dictionary = Dictionary.load("SAT_OLED_Model/weights/mol.dict.txt")
        self.dictionary.add_symbol("[MASK]", is_special=True)

    def transform_raw(self, atoms_list, coordinates_list):
        inputs = []
        for atoms, coordinates in zip(atoms_list, coordinates_list):
            inputs.append(coords2molrep(atoms, coordinates, self.dictionary, self.max_atoms, remove_hs=self.remove_hs))
        return inputs


def inner_coords(atoms, coordinates, remove_hs=True):
    """
    Processes a list of atoms and their corresponding coordinates to remove hydrogen atoms if specified.
    This function takes a list of atom symbols and their corresponding coordinates and optionally removes hydrogen atoms from the output. It includes assertions to ensure the integrity of the data and uses numpy for efficient processing of the coordinates. 

    :param atoms: (list) A list of atom symbols (e.g., ['C', 'H', 'O']).
    :param coordinates: (list of tuples or list of lists) Coordinates corresponding to each atom in the `atoms` list.
    :param remove_hs: (bool, optional) A flag to indicate whether hydrogen atoms should be removed from the output.
                      Defaults to True.
    
    :return: A tuple containing two elements; the filtered list of atom symbols and their corresponding coordinates.
             If `remove_hs` is False, the original lists are returned.
    
    :raises AssertionError: If the length of `atoms` list does not match the length of `coordinates` list.
    """
    assert len(atoms) == len(coordinates), "coordinates shape is not align atoms"
    coordinates = np.array(coordinates).astype(np.float32)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with atoms"
        return atoms_no_h, coordinates_no_h
    else:
        return atoms, coordinates


def coords2molrep(atoms, coordinates, dictionary, max_atoms=256, remove_hs=True, **params):
    """
    Converts atom symbols and coordinates into a unified molecular representation.

    :param atoms: (list) List of atom symbols.
    :param coordinates: (ndarray) Array of atomic coordinates.
    :param dictionary: (Dictionary) An object that maps atom symbols to unique integers.
    :param max_atoms: (int) The maximum number of atoms to consider for the molecule.
    :param remove_hs: (bool) Whether to remove hydrogen atoms from the representation.
    :param params: Additional parameters.

    :return: A dictionary containing the molecular representation with tokens, distances, coordinates, and edge types.
    """
    atoms, coordinates = inner_coords(atoms, coordinates, remove_hs=remove_hs)
    atoms = np.array(atoms)
    coordinates = np.array(coordinates).astype(np.float32)
    # cropping atoms and coordinates
    if len(atoms) > max_atoms:
        idx = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = atoms[idx]
        coordinates = coordinates[idx]
    # tokens padding
    src_tokens = np.array([dictionary.bos()] + [dictionary.index(atom) for atom in atoms] + [dictionary.eos()])
    src_distance = np.zeros((len(src_tokens), len(src_tokens)))
    # coordinates normalize & padding
    src_coord = coordinates - coordinates.mean(axis=0)
    src_coord = np.concatenate([np.zeros((1, 3)), src_coord, np.zeros((1, 3))], axis=0)
    # distance matrix
    src_distance = distance_matrix(src_coord, src_coord)
    # edge type
    src_edge_type = src_tokens.reshape(-1, 1) * len(dictionary) + src_tokens.reshape(1, -1)

    return {
        'src_tokens': src_tokens.astype(int),
        'src_distance': src_distance.astype(np.float32),
        'src_coord': src_coord.astype(np.float32),
        'src_edge_type': src_edge_type.astype(int),
    }
