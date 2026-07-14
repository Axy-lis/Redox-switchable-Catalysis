from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Atom:
    element: str
    coord: np.ndarray


@dataclass
class Molecule:
    atoms: List[Atom]


@dataclass
class Catalyst:
    name: str
    metal: str
    ligands: List[str]
    oxidation_state: Optional[int] = None


@dataclass
class ReactionDescriptor:
    catalyst: Catalyst
    yield_percent: float
    temperature_c: float
    descriptors: dict
