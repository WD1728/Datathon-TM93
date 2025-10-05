# Datathon-TM93
# TM93 Chang Wenda; Ishita Singhal; Aadya Gupta

Datathon_finalized.ipynb is the completed notebook, which includes all our working progress, offering all information and thinking progress. 

Run finalised model instructions:
Download the rf file
pip install numpy pandas scikit-learn joblib torch seaborn matplotlib ucimlrepo

From the rf file project root directory:
## Prepare dataset (download UCI CTG and generate train/test)
python3 prepare_data.py
✅ Saved data/train.csv and data/test.csv

## Train final model (Random Forest + Feature Engineering)
python3 train.py
[Train] Balanced Acc: ... | Macro-F1: ...
✅ Saved: models/rf_fe.pt and models/rf_fe.joblib

## Run inference/testing
python3 infer.py
✅ Saved predictions to predictions.csv

After you run the code correctly, the output should be like:
we include it here in case your local environment/ libraries are different, to prevent any segmentation fault

![IMAGE 2025-10-05 15:42:26](https://github.com/user-attachments/assets/631b8b49-eef5-4dc1-b557-bee62e70734a)
