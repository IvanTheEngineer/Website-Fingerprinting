# Website-Fingerprinting
Elliot Hong (kyk4ge), Ivan Kisselev (wnc8zw), Kyle Vitayanuvatti (acd6hn)

## Background/Purpose:
Website Fingerprinting describes a special attack technique used to deanonymize users and identify the websites these users browse. It threatens the anonymity of the users’ privacy since these users often use tools like Tor or FireFox to conceal their footprint. In a previous homework, we performed traffic correlation to link source-to-entry nodes with exit-to-server nodes, but this project serves as an alternate approach through website fingerprinting: capturing packets of user traffic, extracting features from these packets, and training models on these features. In this way, the attack can be performed using traffic from a single point in the communication stream. Our goal for this project was to investigate the whole process of training a website fingerprinting ML model including packet captures, feature extraction, model creation and training, model evaluation, and overall analysis.

### How to run:
(Make sure all dependencies in requirements.txt are installed)

Training (from scratch)
1. First run collect.py to populate 'testing' and 'training' folders
2. Then run feature_extractor.py
3. Make sure no '.keras' saved models exist already
4. Run each {modelname}model.py file to train the models on the extracted data

Evaluation
1. Run eval_acc_comparison.py to show all accuracies and losses

### (Consult Report PDF for detailed implemetation details and results analysis)
