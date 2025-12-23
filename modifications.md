In this fork the following changes were made:
- Metrics were added i.e. HD95, HD miccai conform, accuracy, sensitivity, specificity, lesionrecall and lesionprecision
  - See **nnunetvs/evaluation/evaluate_predictions.py** -> the metrics will be added to the summary.json file when running **nnUNetv2_evaluate_folder**
- custom Trainer
- custom ExperimentPlanner
- future: Custom Predictors