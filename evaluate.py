import pandas as pd
from model.nmt_estimator import NMTEstimator
from scipy.stats import pearsonr, spearmanr
import numpy as np
import argparse

import os
import yaml

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

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Model checkpoint to be tested.")
    parser.add_argument("--mc_dropout", default=False, help="Model checkpoint to be tested.")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    data = pd.read_csv("data/glass-box.2020-da.qe.test20.csv")
    
    data["TP"] = data["TP"].astype(float)
    data["Soft-Ent"] = data["Soft-Ent"].astype(float)
    data["Sent-Var"] = data["Sent-Var"].astype(float)
    data["D-TP"] = data["D-TP"].astype(float)
    data["D-Var"] = data["D-Var"].astype(float)
    data["D-Combo"] = data["D-Combo"].astype(float)
    data["D-Lex_Sim"] = data["D-Lex_Sim"].astype(float)
    data["src"] = data["src"].astype(str)
    data["mt"] = data["mt"].astype(str)
    data["score"] = data["score"].astype(float)
    
    data = data.to_dict("records")

    if args.mc_dropout:
        model.set_mc_dropout(int(args.mc_dropout))
        
    all_scores = []
    for lp in ['en-de', 'en-zh', 'ro-en', 'et-en', 'ne-en', 'si-en', "ru-en"]:
        lp_data = [sample for sample in data if sample["lp"] == lp]
        y = [d["score"] for d in lp_data]

        _, y_hat = model.predict(lp_data, show_progress=True, cuda=True, batch_size=16)
        if isinstance(y_hat[0], list):
            y_hat = [s[0] for s in y_hat]
            
        pearson = pearsonr(y, y_hat)[0]
        spearman = spearmanr(y, y_hat)[0]
        diff = np.array(y) - np.array(y_hat)
        mae = np.abs(diff).mean()
        rmse = (diff ** 2).mean() ** 0.5
        print (f"Results for {lp}")
        print(f"pearson: {pearson}")
        print(f"spearman: {spearman}")
        
        print(f"mae: {mae}")
        print(f"rmse: {rmse}")
        all_scores.append([pearson, spearman, mae, rmse])        

    print("\t averaging...")
    average_scores = np.mean(np.array(all_scores), axis=0)
    print("done.")
    print("pearson: {}".format(average_scores[0]))
    print("spearman: {}".format(average_scores[1]))
    print("mae: {}".format(average_scores[2]))
    print("rmse: {}".format(average_scores[3]))
