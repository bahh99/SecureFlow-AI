# PaySim Dataset

SecureFlow AI uses the **PaySim** synthetic mobile money transaction dataset
for federated fraud detection training and evaluation.

## Why PaySim?

PaySim is the most widely-used public dataset for mobile money fraud research.
It was created from real African mobile money transaction logs by Lopez-Rojas
et al. (2016), with all sensitive information removed. Its fraud patterns
mirror the actual attack types SecureFlow AI is designed to detect:

- **TRANSFER → CASH_OUT** chains (the dominant real-world fraud pattern)
- Account drainage attacks
- Mule account flows

## Get the data

The repo runs with **synthetic fallback data** out of the box, so you can
demo SecureFlow AI without downloading anything. To use the real PaySim:

1. Download from Kaggle:
   https://www.kaggle.com/datasets/ealaxi/paysim1

2. Unzip and place the CSV here:
   ```
   data/paysim.csv
   ```
   (Or keep the original Kaggle filename `PS_20174392719_1491204439457_log.csv`
   in this folder — both names are detected automatically.)

3. Re-run training:
   ```bash
   flwr run . local-simulation --stream
   ```

## Without the download

If `data/paysim.csv` is missing, SecureFlow AI auto-generates a
50,000-transaction synthetic PaySim-shaped dataset with realistic
fraud patterns. This is sufficient for demos and CI testing.

## Data does not get committed

The `.gitignore` excludes `data/*.csv`, so PaySim never gets pushed to GitHub.
This keeps the repo light and respects the dataset's terms of use.

## Citation

If you use PaySim in publications:

> Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016).
> *PaySim: A financial mobile money simulator for fraud detection.*
> 28th European Modeling and Simulation Symposium (EMSS).
