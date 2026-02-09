import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from visualization_utils import evaluate_and_plot

def run():
    print("Loading data...")
    X_train = joblib.load('processed_data/X_train.pkl')
    X_test = joblib.load('processed_data/X_test.pkl')
    y_train = joblib.load('processed_data/y_train.pkl')
    y_test = joblib.load('processed_data/y_test.pkl')
    label_mapping = joblib.load('processed_data/label_mapping.pkl')

    print("Training Random Forest...")
    # n_estimators=100: Create 100 trees
    # n_jobs=-1: Use all CPU cores
    # class_weight='balanced': Handle the imbalance
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced')
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    print(f"Training Time: {time.time() - start_time:.2f} seconds")

    evaluate_and_plot(clf, X_test, y_test, "Random Forest", label_mapping)

if __name__ == "__main__":
    run()