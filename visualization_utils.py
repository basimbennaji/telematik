import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

def evaluate_and_plot(model, X_test, y_test, model_name, label_mapping):
    print(f"Evaluating {model_name}...")
    
    # 1. Get Predictions and Probabilities
    if hasattr(model, 'predict_proba'):
        # For Scikit-Learn Models
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    else:
        # For PyTorch Wrapper (handled internally) or models without probability
        y_pred = model.predict(X_test)
        y_prob = getattr(model, 'last_probs', None) # Custom attribute if needed

    # 2. Text Report
    print(f"\n--- {model_name} Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))

    # 3. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Attack'], 
                yticklabels=['Benign', 'Attack'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"cm_{model_name.replace(' ', '_').lower()}.png")
    print(f"Saved cm_{model_name.replace(' ', '_').lower()}.png")
    plt.show()

    # 4. ROC Curve Plot (Only if we have probabilities)
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"roc_{model_name.replace(' ', '_').lower()}.png")
        print(f"Saved roc_{model_name.replace(' ', '_').lower()}.png")
        plt.show()