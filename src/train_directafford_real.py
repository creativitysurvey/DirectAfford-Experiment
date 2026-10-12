"""Compatibility entry point; all options are parsed by the shared trainer."""
from train import main
from real_data import LASORealDataset

if __name__ == "__main__":
    main()
