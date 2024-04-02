import argparse


def main():

    parser = argparse.ArgumentParser(
        "Calculate the nuclear-nuclear coulombic potential of QM7 molecules."
    )
    parser.add_argument("-d", "--data", default="data", help="Path to QM7")
    parser.add_argument(
        "-o",
        "--output",
        default="energies.json",
        help="Output file with potential energies",
    )
    parser.add_argument("--device", default="cpu", help="device (cpu or cuda)")

    args = parser.parse_args()

    from graph_torch.potential import (
        get_qm7_molecules,
        get_graph_from_molecules,
        calculate_potential,
    )
    from graph_torch.out import save_json
    import time

    molecules = get_qm7_molecules(args.data)
    print(f"{molecules.num_atoms.shape[0]} molecules loaded")

    molecules = molecules.to(args.device)

    print("calculate potentials")
    t_start = time.time()
    graphs = get_graph_from_molecules(molecules)
    potential = calculate_potential(molecules, graphs)
    t_end = time.time()

    print(f"done in {(t_end-t_start)*1000:.1f} ms")

    print(f"save to {args.output}")
    save_json(molecules, potential, args.output)


if __name__ == "__main__":
    main()
