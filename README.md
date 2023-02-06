# ModelPred
Repository for paper "[ModelPred: A Framework for Predicting Trained Model from Training Data](http://128.84.21.203/abs/2111.12545)" 2023 SaTML submission.

# Learning to Refit for Convex Learning Problems

# Requirements
You can first install the environment specified in the requirements.txt

## optLearn DNN model training
Logistic Regression as the base model:
Iris_LR.py, spam_LR.py, HIGGS_LR.py, MNIST_LR.py
Support Vector Machine as the base model:
Iris_SVM.py, spam_SVM.py, HIGGS_SVM.py

To train the model, run this command:
Iris_LR.py --sampling perm --datapath --modelpath
The same for other scripts


## Dataset deletion and addition
Logistic Regression as the base model:
Deletion_Eva.py
Addition_Eva.py

To run the experiment, run this command:
Deletion_Eva.py --sampling Perm --rawdatapth  --modelpath  --savepath
Addition_Eva.py --sampling Perm --rawdatapth  --modelpath  --savepath


## Shapley value
LR_Shapley.py

To run the experiment, run this command:
LR_Shapley.py --sampling Perm --maxiter 50 --rawdatapth  --modelpath  --savepath

Please make sure the data_path (where you save the raw data),
save_path (where you save the training samples for OptLearn),
result_path (where you save all the results), and model_path (where you save the trained models) are all correctly configured.
