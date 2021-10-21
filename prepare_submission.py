"""
Run this script with the following command:

```bash
python prepare_submission.py --checkpoint {path/to/checkpoint}.ckpt --method {method_name}
```

This will save the model predictions for task1-multilingual in the model folder along 
with the .zip for the submission.
"""
import argparse
import os
import shutil
from zipfile import ZipFile

import pandas as pd
import yaml

from model.nmt_estimator import NMTEstimator
from pytorch_lightning import seed_everything



def load_checkpoint(checkpoint: str):
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

LANGUAGE_PAIRS = [
    "en-cs",
    "en-de",
    "en-ja",
    "en-zh",
    "et-en",
    "km-en",
    "ne-en",
    "ps-en",
    "ro-en",
    "ru-en",
    "si-en"
]

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Model checkpoint to be tested.")
    parser.add_argument("--method", required=True, help="Method name for the submission.")
    parser.add_argument("--lp", default="all", help="Method name for the submission.")
    parser.add_argument("--mc_dropout", default=False, help="Model checkpoint to be tested.")
    parser.add_argument(
        "--seed_everything",
        help="Prediction seed.",
        type=int,
        default=12,
    )
    args = parser.parse_args()
    seed_everything(args.seed_everything)

    model = load_checkpoint(args.checkpoint)
    if args.mc_dropout:
        model.set_mc_dropout(int(args.mc_dropout))

    model_parameters = sum(p.numel() for p in model.parameters())
    disk_footprint = os.path.getsize(args.checkpoint)

    output = str(disk_footprint)+"\n"+str(model_parameters)+"\n"
    data = pd.read_csv("data/test21/glass-box.test21.csv")

    lps = LANGUAGE_PAIRS if args.lp == "all" else [args.lp]
    for lp in lps:
        if lp not in data.lp.unique():
            print(f"Glass box features missing for {lp}! Skipping...")
            continue
        
        df = data[data.lp == lp]
        lp_data = df.to_dict("records")
        import pdb; pdb.set_trace()
        _, y_hat = model.predict(lp_data, show_progress=True, cuda=True, batch_size=16)
        if isinstance(y_hat[0], list):
            y_hat = [s[0] for s in y_hat]

        for i, score in enumerate(y_hat):
            output += lp + "\t" + args.method + "\t" + str(i) + "\t" + str(score) + "\n"
    
    model_path = args.checkpoint.split("/")[:-1]
    

    with open("predictions.txt", "w") as text_file:
        text_file.write(output)
        
    zipObj = ZipFile("predictions.txt.zip", 'w')
    zipObj.write("predictions.txt")
    zipObj.close()

    if os.path.isfile("/".join(model_path)+"/predictions.txt"):
        os.remove("/".join(model_path)+"/predictions.txt")
    if os.path.isfile("/".join(model_path)+"/predictions.txt.zip"):
        os.remove("/".join(model_path)+"/predictions.txt.zip")
    shutil.move("predictions.txt", "/".join(model_path))
    shutil.move("predictions.txt.zip", "/".join(model_path))
