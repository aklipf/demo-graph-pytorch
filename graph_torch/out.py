import torch
import torch.nn.functional as F

from graph_torch.potential import Molecules

import json


def save_json(molecules: Molecules, energies: torch.FloatTensor, filename: str):
    data = []

    molecules = molecules.to("cpu")
    energies = energies.cpu()

    ptr = F.pad(molecules.num_atoms.cumsum(0), (1, 0))
    for idx, energy in enumerate(energies):
        data.append(
            {
                "idx": idx,
                "z": molecules.z[ptr[idx] : ptr[idx + 1]].tolist(),
                "energy": energy.item(),
            }
        )

    with open(filename, "w") as fp:
        json.dump(data, fp, indent=4)
