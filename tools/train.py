import hydra
import sys
from hmr4d.train import train

@hydra.main(version_base=None, config_path="../hmr4d/configs", config_name="train")
def main(cfg) -> None:
    train(cfg)
    
if __name__ == "__main__":
    main()