import joblib
import time
from sklearn.linear_model import LogisticRegression
from visualization_utils import evaluate_and_plot

def run():
    print("Loading data...")
    X_train = joblib.load('processed_data/X_train.pkl')
    X_test = joblib.load('processed_data/X_test.pkl')
    y_train = joblib.load('processed_data/y_train.pkl')
    y_test = joblib.load('processed_data/y_test.pkl')
    label_mapping = joblib.load('processed_data/label_mapping.pkl')

    print("Training Logistic Regression...")
    # solver='saga': Faster for large datasets
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', solver='saga')
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    print(f"Training Time: {time.time() - start_time:.2f} seconds")

    evaluate_and_plot(clf, X_test, y_test, "Logistic Regression", label_mapping)

if __name__ == "__main__":
    run()