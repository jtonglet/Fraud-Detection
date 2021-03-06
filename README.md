# Car insurance fraud project   πΈππ΅οΈββοΈ

Small fraud prediction project for the course [Advanced Analytics](https://onderwijsaanbod.kuleuven.be/2020/syllabi/e/D0S06AE.htm#activetab=doelstellingen_idm6206064) @ KULeuven. Logistic Regression is used to predict whether a car insurance claim is fraudulent or not. *preprocessing.py* creates features from the raw data. *lr_classification.py* returns the classification performance on a stratified test set constituing 20% of the original dataset.

## Results 

| Metric | Score |
| --- | --- |
| Precision |  3.55 |
| Recall |  72.58 |
| F1 |   6.77  |
| AUC   |   80.75  |

**ROC Curve**

Positive = fraudulent claim

  <img src="output/roc_curve.png" width="350" title="ROC Curve">

