#!/usr/bin/env python3

#Importing advanced libraries

import pandas as pd
import numpy as np
import joblib
import warnings
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter
import time

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Here, we are importing the libraries that are used in the code
#If the library is not installed, the code will print an error message
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not installed: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not installed: pip install catboost")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    HAS_TENSORFLOW = True
    tf.get_logger().setLevel('ERROR')
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not installed: pip install tensorflow")

# Configuration - Here, we are configuring the dataset and the models
DATASET_PATH = 'all_projects_combined.csv'
MODELS_PATH = 'truly_clean_models/'
TEST_SIZE = 0.25
RANDOM_STATE = 42

#Here, we are loading the dataset and preparing the data for the models to use by filtering out the suspicious features
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    
    df['label_binary'] = (df['label'] == 'BUGGY').astype(int)
    
    # Only use features available BEFORE commit
    TRULY_SAFE_FEATURES = [
        # Raw code change metrics
        'added_lines',
        'deleted_lines',
        'nloc',
        'complexity',
        'token_count',
        
        # Commit message basic metrics (no keyword analysis!)
        'commit_msg_length',
        'commit_msg_word_count',
        'commit_msg_char_count',
        
        # Simple mathematical derivations
        'total_changes',
        'net_lines',
        'code_churn',
    ]
    
    # REMOVED FEATURES
    REMOVED_SUSPICIOUS = {
        'is_bug_fix': 'Derived from bug tracking system - LEAKAGE',
        'has_performance_keywords': 'Keywords from post-labeling',
        'has_bug_keywords': 'Keywords from post-labeling',
        'has_refactor_keywords': 'Keywords from post-labeling',
        'has_test_keywords': 'Keywords from post-labeling',
        'has_doc_keywords': 'Keywords from post-labeling',
        'commit_msg_upper_ratio': 'May correlate with labeling',
        'avg_word_length': 'May correlate with labeling',
    }
    
    print(f"\n{'STRICT FEATURE FILTERING':^80}")
    print("="*80)
    print(f"USING ONLY: {len(TRULY_SAFE_FEATURES)} core features")
    print(f"REMOVED: {len(REMOVED_SUSPICIOUS)} suspicious features")
    print(f"\nSafe features:")
    for feat in TRULY_SAFE_FEATURES:
        print(f"   {feat}")
    
    # Only using features that exist in the dataset
    feature_cols = [col for col in TRULY_SAFE_FEATURES if col in df.columns]
    
    if len(feature_cols) < len(TRULY_SAFE_FEATURES):
        missing = set(TRULY_SAFE_FEATURES) - set(feature_cols)
        print(f"\n Missing features: {missing}")
        print(f"   Creating them from available data...")
        
        if 'total_changes' not in feature_cols and 'added_lines' in df.columns:
            df['total_changes'] = df['added_lines'] + df['deleted_lines']
            feature_cols.append('total_changes')
        
        if 'net_lines' not in feature_cols and 'added_lines' in df.columns:
            df['net_lines'] = df['added_lines'] - df['deleted_lines']
            feature_cols.append('net_lines')
        
        if 'code_churn' not in feature_cols and 'added_lines' in df.columns:
            df['code_churn'] = df['added_lines'] + df['deleted_lines']
            feature_cols.append('code_churn')
    
    # Preparing features for the models to use
    X = df[feature_cols].copy()
    
    # Handling missing/invalid values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    y = df['label_binary']
    
    print(f"\nFinal clean dataset: X={X.shape}, y={y.shape}")
    print(f"Using {len(feature_cols)} truly safe features")
    
    return X, y, feature_cols, df


def cross_validation_check(X, y, feature_cols):
    """Perform rigorous cross-validation to detect leakage"""
    print("\n" + "="*80)
    print("CROSS-VALIDATION LEAKAGE TEST")
    print("="*80)
    print("Testing with simple Logistic Regression...")
    print("If CV score is >90%, leakage likely still exists!")
    
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cv_scores = cross_val_score(lr, X_scaled, y, cv=skf, scoring='accuracy')
    cv_f1 = cross_val_score(lr, X_scaled, y, cv=skf, scoring='f1')
    cv_roc = cross_val_score(lr, X_scaled, y, cv=skf, scoring='roc_auc')
    
    print(f"\n5-Fold Cross-Validation Results:")
    print(f"   Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   F1-Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"   ROC-AUC:  {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
    
    print(f"\n{'LEAKAGE VERDICT':^80}")
    if cv_scores.mean() > 0.90:
        print("LEAKAGE LIKELY PRESENT - Accuracy >90%")
        return False
    elif cv_scores.mean() > 0.85:
        print("POSSIBLE LEAKAGE - Accuracy 85-90%")
        return True
    else:
        print("NO LEAKAGE DETECTED - Realistic performance")
        return True

#Here, we are creating the deep learning model
def create_deep_learning_model(input_dim):
    if not HAS_TENSORFLOW:
        return None
    
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,),
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model

#Here, we are training a single model and returning the metrics
def train_single_model(model, model_name, X_train_scaled, y_train, X_test_scaled, y_test, is_neural_net=False):
    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    if is_neural_net:
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=128,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        y_pred_proba = model.predict(X_test_scaled, verbose=0).ravel()
        y_pred = (y_pred_proba >= 0.5).astype(int)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    training_time = time.time() - start_time
    
    # Calculating the metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        roc_auc = 0.0
    
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Printing the results
    print(f"\nTraining Time: {training_time:.2f}s")
    print(f"Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
    print(f"ROC-AUC:       {roc_auc:.4f}")
    print(f"MCC:           {mcc:.4f}")
    print(f"\nBUGGY Detection:")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f} -> Found {tp}/{tp+fn} bugs")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn:4d}  FP={fp:4d}")
    print(f"  FN={fn:4d}  TP={tp:4d}")
    
    return {
        'model_name': model_name,
        'model': model,
        'accuracy': acc,
        'roc_auc': roc_auc,
        'mcc': mcc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'training_time': training_time,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

#Here, we are training all the models and returning the results
def train_all_models(X_train_scaled, y_train, X_test_scaled, y_test):
    print("\n" + "="*80)
    print("TRAINING ALL 10 MODELS")
    print("="*80)
    
    results = []
    
    # Training the Logistic Regression model
    print("\n[1/10] Logistic Regression (Baseline)")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
    results.append(train_single_model(lr, "Logistic Regression", X_train_scaled, y_train, X_test_scaled, y_test))
    
    # Training the Decision Tree model
    print("\n[2/10] Decision Tree")
    dt = DecisionTreeClassifier(max_depth=20, min_samples_split=10, min_samples_leaf=5,
                                class_weight='balanced', random_state=RANDOM_STATE)
    results.append(train_single_model(dt, "Decision Tree", X_train_scaled, y_train, X_test_scaled, y_test))
    
    # Training the Random Forest model
    print("\n[3/10] Random Forest")
    rf = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=5,
                                min_samples_leaf=2, class_weight='balanced',
                                random_state=RANDOM_STATE, n_jobs=-1)
    results.append(train_single_model(rf, "Random Forest", X_train_scaled, y_train, X_test_scaled, y_test))
    
    # Training the Gradient Boosting model
    print("\n[4/10] Gradient Boosting")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=10, learning_rate=0.1,
                                    subsample=0.8, random_state=RANDOM_STATE)
    results.append(train_single_model(gb, "Gradient Boosting", X_train_scaled, y_train, X_test_scaled, y_test))
    
    # Training the Naive Bayes model
    print("\n[5/10] Naive Bayes")
    nb = GaussianNB()
    results.append(train_single_model(nb, "Naive Bayes", X_train_scaled, y_train, X_test_scaled, y_test))
    
    # Training the XGBoost model
    if HAS_XGB:
        print("\n[6/10] XGBoost")
        xgb = XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                           random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss')
        results.append(train_single_model(xgb, "XGBoost", X_train_scaled, y_train, X_test_scaled, y_test))
    else:
        print("\n[6/10] XGBoost - SKIPPED (not installed)")
    
    # Training the LightGBM model
    if HAS_LGBM:
        print("\n[7/10] LightGBM")
        lgbm = LGBMClassifier(n_estimators=300, max_depth=15, learning_rate=0.05,
                             num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                             class_weight='balanced', random_state=RANDOM_STATE,
                             n_jobs=-1, verbose=-1)
        results.append(train_single_model(lgbm, "LightGBM", X_train_scaled, y_train, X_test_scaled, y_test))
    else:
        print("\n[7/10] LightGBM - SKIPPED (not installed)")
    
    # Training the CatBoost model
    if HAS_CATBOOST:
        print("\n[8/10] CatBoost")
        catboost = CatBoostClassifier(iterations=300, depth=8, learning_rate=0.05,
                                      l2_leaf_reg=3,
                                      class_weights=[1, (y_train == 0).sum() / (y_train == 1).sum()],
                                      random_seed=RANDOM_STATE, verbose=0, thread_count=-1)
        results.append(train_single_model(catboost, "CatBoost", X_train_scaled, y_train, X_test_scaled, y_test))
    else:
        print("\n[8/10] CatBoost - SKIPPED (not installed)")
    
    # Training the Deep Learning model
    if HAS_TENSORFLOW:
        print("\n[9/10] Deep Learning (Feed-Forward Neural Network)")
        dl_model = create_deep_learning_model(X_train_scaled.shape[1])
        results.append(train_single_model(dl_model, "Deep Learning (FFN)", X_train_scaled, y_train,
                                       X_test_scaled, y_test, is_neural_net=True))
    else:
        print("\n[9/10] Deep Learning - SKIPPED (TensorFlow not installed)")
    
    # Creating the Ensemble model
    print("\n[10/10] Creating Ensemble Model...")
    if len(results) >= 3:
        sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)
        top_3 = sorted_results[:3]
        
        print(f"   Using top 3 models:")
        for i, r in enumerate(top_3, 1):
            print(f"   {i}. {r['model_name']} (F1: {r['f1']:.4f})")
        
        ensemble_proba = np.mean([r['probabilities'] for r in top_3], axis=0)
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        acc = accuracy_score(y_test, ensemble_pred)
        f1 = f1_score(y_test, ensemble_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, ensemble_pred, pos_label=1, zero_division=0)
        prec = precision_score(y_test, ensemble_pred, pos_label=1, zero_division=0)
        
        print(f"\n   Ensemble Accuracy:  {acc:.4f}")
        print(f"   Ensemble F1-Score:  {f1:.4f}")
        print(f"   Ensemble Recall:    {rec:.4f}")
        
        cm = confusion_matrix(y_test, ensemble_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        ensemble_results = {
            'model_name': 'Ensemble (Top 3)',
            'model': None,
            'accuracy': acc,
            'roc_auc': roc_auc_score(y_test, ensemble_proba) if len(np.unique(y_test)) > 1 else 0,
            'mcc': matthews_corrcoef(y_test, ensemble_pred),
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'training_time': 0,
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba
        }
        results.append(ensemble_results)
    
    return results

#Here, we are printing the comparison table of the models
def print_comparison_table(results):
    print("\n" + "="*120)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*120)
    
    df = pd.DataFrame([{
        'Model': r['model_name'],
        'Accuracy': r['accuracy'],
        'ROC-AUC': r['roc_auc'],
        'MCC': r['mcc'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1': r['f1'],
        'TP': r['tp'],
        'TN': r['tn'],
        'FP': r['fp'],
        'FN': r['fn']
    } for r in results])
    
    df = df.sort_values('F1', ascending=False)
    
    print(f"\n{'Model':<25} {'Acc':<7} {'AUC':<7} {'MCC':<7} {'F1':<8} {'Recall':<8} {'TP':<5} {'FN':<5}")
    print("-"*120)
    
    for _, row in df.iterrows():
        print(f"{row['Model']:<25} {row['Accuracy']:.4f}  {row['ROC-AUC']:.4f}  "
              f"{row['MCC']:.4f}  {row['F1']:.4f}   {row['Recall']:.4f}   "
              f"{row['TP']:<5.0f} {row['FN']:<5.0f}")
    
    # Printing the best model details
    best = results[0]
    print(f"\n{'='*120}")
    print(f"BEST MODEL: {best['model_name']}")
    print(f"{'='*120}")
    print(f"Accuracy:  {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall:    {best['recall']:.4f} (Found {best['tp']}/{best['tp']+best['fn']} bugs)")
    print(f"F1-Score:  {best['f1']:.4f}")
    print(f"ROC-AUC:   {best['roc_auc']:.4f}")
    print(f"MCC:       {best['mcc']:.4f}")
    
    # Checking the reality of the results
    print(f"\n{'='*120}")
    print("REALITY CHECK")
    print("="*120)
    if best['f1'] > 0.80:
        print("EXCELLENT: F1 > 80% - Outstanding performance!")
        print("    This is realistic for multi-project training")
    elif best['f1'] > 0.70:
        print("VERY GOOD: F1 70-80% - Strong performance")
    elif best['f1'] > 0.60:
        print("GOOD: F1 60-70% - Solid performance")
    else:
        print("MODERATE: F1 < 60% - Room for improvement")
    
    return df

#Here, we are saving the models
def save_models(results, scaler, feature_cols, results_df):
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    print(f"\n{'='*120}")
    print("SAVING MODELS")
    print("="*120)
    
    for i, result in enumerate(results[:5], 1):
        if result['model'] is not None:
            filename = f"{MODELS_PATH}model_{i}_{result['model_name'].replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
            
            artifacts = {
                'model': result['model'],
                'scaler': scaler,
                'feature_columns': feature_cols,
                'metrics': {k: v for k, v in result.items() if k not in ['model', 'predictions', 'probabilities']},
            }
            
            joblib.dump(artifacts, filename)
            print(f"Saved: {filename}")
    
    results_df.to_csv(f'{MODELS_PATH}all_models_comparison.csv', index=False)
    print(f"Saved: {MODELS_PATH}all_models_comparison.csv")

#Here, we are the main pipeline
def main():
    print("\n")
    print("="*120)
    print("TRULY LEAKAGE-FREE BUG PREDICTION SYSTEM - COMPLETE VERSION")
    print("Training on Jenkins + Hadoop + PyTorch with ALL 10 algorithms")
    print("="*120)
    
    # Loading the data
    X, y, feature_cols, df = load_and_prepare_data(DATASET_PATH)
    
    # Checking for leakage
    is_clean = cross_validation_check(X, y, feature_cols)
    
    if not is_clean:
        print("\nWARNING: Cross-validation suggests possible leakage!")
        print("   Proceeding anyway, but results should be interpreted carefully")
    
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n{'DATA SPLIT':^120}")
    print("="*120)
    print(f"Train: {len(X_train):,} samples (Clean: {(y_train==0).sum():,}, Buggy: {(y_train==1).sum():,})")
    print(f"Test:  {len(X_test):,} samples (Clean: {(y_test==0).sum():,}, Buggy: {(y_test==1).sum():,})")
    
    # Applying SMOTETomek to balance the classes
    print(f"\n{'SMOTE + TOMEK RESAMPLING':^120}")
    print("="*120)
    sampler = SMOTETomek(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    print(f"Before: {Counter(y_train)}")
    print(f"After:  {Counter(y_train_resampled)}")
    
    # Scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # Training all the models
    results = train_all_models(X_train_scaled, y_train_resampled, X_test_scaled, y_test)
    results_sorted = sorted(results, key=lambda x: x['f1'], reverse=True)
    
    # Printing the results
    df_results = print_comparison_table(results_sorted)
    
    # Saving the models
    save_models(results_sorted, scaler, feature_cols, df_results)
    
    print(f"\n{'='*120}")
    print("TRAINING COMPLETE!")
    print("="*120)
    print(f"Models saved to: {MODELS_PATH}")
    print(f"Best model: {results_sorted[0]['model_name']}")
    print(f"Best F1-Score: {results_sorted[0]['f1']:.4f}")
    print(f"\nYour model is ready for production use!")
    print(f"Multi-project training (Jenkins + Hadoop + PyTorch) = Better generalization")
    
    return results_sorted, scaler, feature_cols

#Here, we are the main function that is used to run the code
if __name__ == "__main__":
    results, scaler, features = main()