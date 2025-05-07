# Wicket Prediction using Machine Learning

This project implements a machine learning pipeline to predict the probability of a wicket falling during T20 cricket matches. It includes scripts for training, evaluation, inference, ablation studies, and threshold analysis.

## 📂 Repository Structure

```
├── train.py
├── eval.py
├── infer.py
├── ablation_study.py
├── threshold_analyzer.py
```

## 📁 Dataset and Model Files

> 📌 **Note:** The dataset files and trained model checkpoints can be accessed at the following Google Drive link:  
👉 [**Google Drive - Dataset and Model**](https://drive.google.com/drive/folders/1ecFgYHz5cJlta4M_lLT5Edxic_-74lYo?usp=sharing)

---

## 🔧 Code Overview

### `train.py`

This script handles the training of the wicket prediction model.

- Preprocesses input features and labels.
- Trains the model using a deep learning architecture.
- Saves the model to a checkpoint file for future inference.

Run with:

```bash
python train.py
```

---

### `eval.py`

Evaluates the trained model on a validation/test dataset.

- Loads the saved model.
- Computes standard evaluation metrics (precision, recall, F1, etc.).

Run with:

```bash
python eval.py
```

---

### `infer.py`

Runs inference using a trained model on unseen match data.

- Outputs predicted wicket probabilities for each ball.
- Saves a CSV with probabilities and actual labels (if provided).
- **🔧 Note:** The threshold used to classify a ball as a wicket can be **manually modified by the user** within the script (`THRESHOLD` variable).

Run with:

```bash
python infer.py --model wicket_probability_model.pkl --match_id 733989 --over 17 --ball 2 --inning 2
```


---

### `ablation_study.py`

Performs ablation experiments by removing or modifying input features to assess their importance.

- Useful for analyzing model robustness and feature contributions.
- Saves logs and metric comparisons for different feature sets.

Run with:

```bash
python ablation_study.py
```

---

### `threshold_analyzer.py`

Analyzes how different classification thresholds affect model performance.

- Reads a CSV file containing `wicket_probability` and `actual_wicket` columns.
- Computes precision, recall, F1, AUC metrics for multiple thresholds.
- Saves per-threshold predictions and summary results.
- Generates precision-recall curve data.

Run with:

```bash
python threshold_analyzer.py --input predictions.csv --thresholds 0.5 0.55 0.6 0.65 --output_dir results/
```

---

## 📈 Output Files

- `predictions.csv`: Inference results with probabilities and labels.
- `threshold_analysis.csv`: Summary of metrics across thresholds.
- `summary.json`: Best threshold and its performance metrics.
- `precision_recall_curve.csv`: PR curve data.

---

