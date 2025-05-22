import argparse

from nequip_eqx.model import load_model
from nequip_eqx.data import DataLoader, Dataset
from nequip_eqx.train import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    model, config = load_model(args.model)
    test_dataset = Dataset(
        file_path=args.file,
        atomic_numbers=config["atomic_numbers"],
        cutoff=config["cutoff"],
    )
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])
    _, val_energy_rmse, val_force_rmse = evaluate(model, test_loader)
    print(f"file: {args.file}")
    print(f"energy RMSE: {val_energy_rmse * 1000:.2f} meV")
    print(f"force RMSE: {val_force_rmse * 1000:.2f} meV/Å")


if __name__ == "__main__":
    main()
