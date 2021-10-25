from .nmt_estimator import NMTEstimator
from .download_utils import download_mbart_qe

import os
import yaml


def load_mbart_qe(checkpoint: str):
    if not os.path.exists(checkpoint):
        raise Exception(f"{checkpoint} file not found!")

    hparam_yaml_file = "/".join(checkpoint.split("/")[:-1] + ["hparams.yaml"])

    if os.path.exists(hparam_yaml_file):
        with open(hparam_yaml_file) as yaml_file:
            hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        model = NMTEstimator.load_from_checkpoint(
            checkpoint, hparams=hparams
        )
    else:
        raise Exception("hparams file not found.")

    model.eval()
    model.freeze()
    return model