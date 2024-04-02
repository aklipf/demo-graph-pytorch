import requests
import os
import zipfile
from typing import List, Tuple
import io

import torch
from ase.io import read


def download_qm7(url, path: str = None):
    local_filename = url.split("/")[-1]

    if path is not None:
        local_filename = os.path.join(path, local_filename)

    if os.path.exists(local_filename):
        return

    print("download qm7")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def process_qm7(zip_file: str, xyz_file: str = "dsgdb7ae.xyz"):
    out_dir = os.path.split(zip_file)[0]

    if os.path.exists(os.path.join(out_dir, xyz_file)):
        return

    print("extract qm7")

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extract(xyz_file, path=out_dir)


def str_to_torch(xyz: str) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    atoms = read(io.StringIO(xyz), format="xyz")

    x = torch.from_numpy(atoms.get_positions()).float()
    z = torch.from_numpy(atoms.get_atomic_numbers()).long()
    return x, z


def load_xyz(xyz_file: str) -> List[Tuple[torch.FloatTensor, torch.LongTensor]]:
    print("load qm7")

    with open(xyz_file, "r") as fp:
        xyz = fp.read()
        xyz_structres = xyz.split("\n\n")

    return list(map(str_to_torch, xyz_structres))


def get_qm7(
    path: str = "data", url: str = "https://qmml.org/Datasets/gdb7-12.zip"
) -> List[tuple[torch.FloatTensor, torch.LongTensor]]:
    os.makedirs(path, exist_ok=True)

    download_qm7(url, path)

    process_qm7(os.path.join(path, "gdb7-12.zip"))

    return load_xyz(os.path.join(path, "dsgdb7ae.xyz"))
