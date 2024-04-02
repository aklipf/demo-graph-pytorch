import dataclasses
from dataclasses import dataclass
from typing import TypeVar

Self = TypeVar("Self")

import torch
import torch.nn.functional as F

from graph_torch.qm7 import get_qm7


class ToDeviceMixin:

    def to(self, device: torch.device) -> Self:
        kwargs = {}
        for field in dataclasses.fields(self):
            kwargs[field.name] = getattr(self, field.name).to(device)

        return self.__class__(**kwargs)


@dataclass
class Molecules(ToDeviceMixin):
    num_atoms: torch.LongTensor
    x: torch.FloatTensor
    z: torch.LongTensor


@dataclass
class Graph(ToDeviceMixin):
    molecules_index: torch.LongTensor
    edges_index: torch.LongTensor


def get_qm7_molecules(path: str = "data") -> Molecules:
    qm7 = get_qm7(path)

    num_atoms = torch.tensor([z.shape[0] for _, z in qm7], dtype=torch.long)
    x = torch.cat([x for x, _ in qm7], dim=0)
    z = torch.cat([z for _, z in qm7], dim=0)

    return Molecules(num_atoms=num_atoms, x=x, z=z)


def get_graph_from_molecules(molecules: Molecules) -> Graph:
    device = molecules.num_atoms.device
    num_edges = molecules.num_atoms.pow(2)

    ptr_nodes = F.pad(molecules.num_atoms.cumsum(dim=0), (1, 0))
    ptr_edges = F.pad(num_edges.cumsum(dim=0), (1, 0))

    batch_edge = torch.arange(num_edges.shape[0], device=device).repeat_interleave(
        num_edges
    )
    edge_idx = torch.arange(ptr_edges[-1], device=device) - ptr_edges[batch_edge]

    source_idx = edge_idx // molecules.num_atoms[batch_edge] + ptr_nodes[batch_edge]
    target_idx = edge_idx % molecules.num_atoms[batch_edge] + ptr_nodes[batch_edge]

    edges_index = torch.stack((source_idx, target_idx), dim=0)
    edges_index = edges_index[:, source_idx != target_idx]

    molecules_index = torch.arange(molecules.num_atoms.shape[0], device=device)
    molecules_index = molecules_index.repeat_interleave(molecules.num_atoms)

    return Graph(molecules_index=molecules_index, edges_index=edges_index)


def calculate_potential(molecules: Molecules, graph: Graph):
    device = molecules.num_atoms.device
    n_molecules = molecules.num_atoms.shape[0]

    i, j = graph.edges_index
    r_ij = (molecules.x[i] - molecules.x[j]).norm(dim=1) * 1e-10
    Z_ij = molecules.z[i] * molecules.z[j]

    inv_epsilon_0 = 1 / (4 * torch.pi * 55.26349406e6)

    E_ij = inv_epsilon_0 * Z_ij / r_ij

    E = 0.5 * torch.scatter_add(
        torch.zeros(n_molecules, device=device), 0, graph.molecules_index[i], E_ij
    )

    return E
