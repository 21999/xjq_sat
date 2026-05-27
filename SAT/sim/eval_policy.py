if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import hydra
from omegaconf import OmegaConf
from train import SATTrainWorkspace

@hydra.main(
    version_base=None,
    config_path="sat/config",
    config_name="sat"
)
def main(cfg):
    workspace = SATTrainWorkspace(cfg)
    workspace.eval()

if __name__ == "__main__":
    main()
