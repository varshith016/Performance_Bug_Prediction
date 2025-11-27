"""
Training Service Module
Wraps the ML training logic for API use.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import threading
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import Sampleeight functions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Import training functions from Sampleeight.py
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
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, chi2,
    RFE, RFECV, SelectFromModel
)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Try to import optional libraries
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    HAS_TENSORFLOW = True
    tf.get_logger().setLevel('ERROR')
except ImportError:
    HAS_TENSORFLOW = False

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.30



class EnsembleModel:
    """Ensemble model that combines predictions from top 3 models."""
    
    def __init__(self, top_3_models):
        """
        Initialize ensemble with top 3 models.
        top_3_models: List of model dictionaries with 'model' key
        """
        self.top_3_models = top_3_models
        self.model_names = [m['model_name'] for m in top_3_models]
    
    def predict_proba(self, X):
        """Average predictions from top 3 models."""
        probas = []
        for model_info in self.top_3_models:
            model = model_info['model']
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[:, 1]
            else:
                proba = model.predict(X).astype(float)
            probas.append(proba)
        avg_proba = np.mean(probas, axis=0)
        # Return in sklearn format: [prob_class_0, prob_class_1]
        return np.column_stack([1 - avg_proba, avg_proba])
    
    def predict(self, X):
        """Predict using ensemble."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


class TrainingJob:
    """Represents a training job with status tracking."""
    
    def __init__(self, job_id: str, dataset_path: str):
        self.job_id = job_id
        self.dataset_path = dataset_path
        self.status = "pending"  # pending, running, completed, failed, cancelled
        self.progress = 0.0
        self.message = "Job created"
        self.start_time = None
        self.end_time = None
        self.results = None
        self.error = None
        self.thread = None
        self.repo_path = None  # Path to cloned repository
        self.repo_url = None  # Original repository URL
        self.commits = None  # List of commits
        self.cancelled = False  # Flag to indicate if job was cancelled
    
    def to_dict(self) -> Dict:
        """Convert job to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error,
            "results": self.results,
            "repo_path": self.repo_path,
            "repo_url": self.repo_url,
            "commits": self.commits
        }


class TrainingService:
    """Service for training ML models."""
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(BASE_DIR, "truly_clean_models")
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Store active training jobs
        self.jobs: Dict[str, TrainingJob] = {}
        self.jobs_lock = threading.Lock()
    
    def load_and_prepare_data(self, filepath: Optional[str] = None) -> Tuple:
        """Load and prepare data (from Sampleeight.py)."""
        # Use default dataset if filepath is None
        if filepath is None:
            default_path = os.path.join(BASE_DIR, "all_projects_combined.csv")
            if os.path.exists(default_path):
                filepath = default_path
                logger.info(f"Using default dataset: {filepath}")
            else:
                raise FileNotFoundError(
                    f"Default dataset not found at {default_path}. "
                    "Please provide a dataset path or ensure all_projects_combined.csv exists."
                )
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        logger.info(f"Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)
        
        # Log dataset info
        logger.info(f"Dataset loaded: {len(df)} samples, {len(df.columns)} columns")
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            logger.info(f"Label distribution: {label_counts.to_dict()}")
        
        # Ensure label column exists and is correct
        if 'label' not in df.columns:
            raise ValueError("Dataset must contain a 'label' column with 'BUGGY' or 'CLEAN' values")
        
        df['label_binary'] = (df['label'] == 'BUGGY').astype(int)
        
        # Log binary label distribution
        binary_counts = df['label_binary'].value_counts()
        logger.info(f"Binary label distribution: CLEAN={binary_counts.get(0, 0)}, BUGGY={binary_counts.get(1, 0)}")
        
        # Validate minimum dataset size
        if len(df) < 20:
            logger.warning(f"Dataset is very small ({len(df)} samples). Training may not be reliable.")
        elif len(df) < 100:
            logger.warning(f"Dataset is small ({len(df)} samples). Consider using more data for better results.")
        
        # Only use features available BEFORE commit (including performance-aware metrics)
        TRULY_SAFE_FEATURES = [
            'added_lines', 'deleted_lines', 'nloc', 'complexity', 'token_count',
            'commit_msg_length', 'commit_msg_word_count', 'commit_msg_char_count',
            'total_changes', 'net_lines', 'code_churn',
            # Performance-aware metrics
            'sync_constructs_count', 'max_loop_nesting_depth', 'nested_loops_count',
        ]
        
        # Only using features that exist in the dataset
        feature_cols = [col for col in TRULY_SAFE_FEATURES if col in df.columns]
        
        # Create missing features
        if 'total_changes' not in feature_cols and 'added_lines' in df.columns:
            df['total_changes'] = df['added_lines'] + df['deleted_lines']
            feature_cols.append('total_changes')
        
        if 'net_lines' not in feature_cols and 'added_lines' in df.columns:
            df['net_lines'] = df['added_lines'] - df['deleted_lines']
            feature_cols.append('net_lines')
        
        if 'code_churn' not in feature_cols and 'added_lines' in df.columns:
            df['code_churn'] = df['added_lines'] + df['deleted_lines']
            feature_cols.append('code_churn')
        
        # Preparing features
        X = df[feature_cols].copy()
        
        # Handling missing/invalid values
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        y = df['label_binary']
        
        return X, y, feature_cols, df
    
    def select_features_automated(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method: str = 'selectkbest',
        n_features: Optional[int] = None,
        job: Optional[TrainingJob] = None
    ) -> Tuple[pd.DataFrame, List[str], object]:
        """
        Automatically select features using various methods.
        
        Args:
            X_train: Training features DataFrame
            y_train: Training labels
            method: Selection method ('selectkbest', 'rfe', 'importance', 'mutual_info', 'chi2')
            n_features: Number of features to select (None = auto-select)
            job: Optional training job for progress updates
        
        Returns:
            Tuple of (selected_features_df, selected_feature_names, selector_object)
        """
        if job:
            job.message = f"Selecting features using {method}..."
        
        n_features_auto = min(10, max(5, len(X_train.columns) // 2)) if n_features is None else n_features
        n_features_auto = min(n_features_auto, len(X_train.columns))
        
        selected_features = None
        selector = None
        feature_names = list(X_train.columns)
        
        try:
            if method == 'selectkbest':
                # Univariate feature selection using f_classif
                selector = SelectKBest(score_func=f_classif, k=n_features_auto)
                selected_features = selector.fit_transform(X_train, y_train)
                selected_feature_names = [feature_names[i] for i in selector.get_support(indices=True)]
                logger.info(f"SelectKBest selected {len(selected_feature_names)} features: {selected_feature_names}")
            
            elif method == 'mutual_info':
                # Mutual information-based selection
                selector = SelectKBest(score_func=mutual_info_classif, k=n_features_auto)
                selected_features = selector.fit_transform(X_train, y_train)
                selected_feature_names = [feature_names[i] for i in selector.get_support(indices=True)]
                logger.info(f"Mutual Info selected {len(selected_feature_names)} features: {selected_feature_names}")
            
            elif method == 'chi2':
                # Chi-squared test (for categorical features, but works with numeric too)
                # Ensure all values are non-negative
                X_train_positive = X_train - X_train.min() + 1
                selector = SelectKBest(score_func=chi2, k=n_features_auto)
                selected_features = selector.fit_transform(X_train_positive, y_train)
                selected_feature_names = [feature_names[i] for i in selector.get_support(indices=True)]
                logger.info(f"Chi2 selected {len(selected_feature_names)} features: {selected_feature_names}")
            
            elif method == 'rfe':
                # Recursive Feature Elimination
                estimator = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
                selector = RFE(estimator=estimator, n_features_to_select=n_features_auto, step=1)
                selected_features = selector.fit_transform(X_train, y_train)
                selected_feature_names = [feature_names[i] for i in selector.get_support(indices=True)]
                logger.info(f"RFE selected {len(selected_feature_names)} features: {selected_feature_names}")
            
            elif method == 'importance':
                # Feature importance from Random Forest
                rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
                rf.fit(X_train, y_train)
                
                # Get feature importances
                importances = rf.feature_importances_
                feature_importance_pairs = list(zip(feature_names, importances))
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Select top n_features
                top_features = [name for name, _ in feature_importance_pairs[:n_features_auto]]
                selected_feature_names = top_features
                selected_features = X_train[top_features].values
                selector = SelectFromModel(rf, prefit=True, max_features=n_features_auto)
                logger.info(f"Feature Importance selected {len(selected_feature_names)} features: {selected_feature_names}")
            
            else:
                # Default: use all features
                logger.warning(f"Unknown feature selection method: {method}. Using all features.")
                selected_features = X_train.values
                selected_feature_names = feature_names
                selector = None
            
            # Convert to DataFrame
            selected_df = pd.DataFrame(selected_features, columns=selected_feature_names, index=X_train.index)
            
            return selected_df, selected_feature_names, selector
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}", exc_info=True)
            # Fallback: use all features
            logger.warning("Falling back to using all features")
            return X_train, feature_names, None
    
    def select_best_feature_method(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        methods: List[str] = ['selectkbest', 'rfe', 'importance'],
        job: Optional[TrainingJob] = None
    ) -> Tuple[pd.DataFrame, List[str], str, object]:
        """
        Try multiple feature selection methods and return the best one based on a simple test.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            methods: List of methods to try
            job: Optional training job
        
        Returns:
            Tuple of (best_features_df, best_feature_names, best_method_name, selector)
        """
        if job:
            job.message = "Evaluating feature selection methods..."
        
        best_score = -1
        best_features = None
        best_feature_names = None
        best_method = None
        best_selector = None
        
        for method in methods:
            try:
                # Select features
                X_train_selected, feature_names, selector = self.select_features_automated(
                    X_train, y_train, method=method, job=job
                )
                
                # Quick test with a simple model
                test_model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
                test_model.fit(X_train_selected, y_train)
                
                # Get test features (same selection)
                if selector is not None and hasattr(selector, 'transform'):
                    X_test_selected = selector.transform(X_test)
                else:
                    X_test_selected = X_test[feature_names].values
                
                # Evaluate
                score = test_model.score(X_test_selected, y_test)
                
                logger.info(f"Feature selection method '{method}' score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_features = X_train_selected
                    best_feature_names = feature_names
                    best_method = method
                    best_selector = selector
                    
            except Exception as e:
                logger.warning(f"Error evaluating method '{method}': {e}")
                continue
        
        if best_features is None:
            # Fallback: use all features
            logger.warning("All feature selection methods failed. Using all features.")
            return X_train, list(X_train.columns), 'none', None
        
        logger.info(f"Best feature selection method: {best_method} (score: {best_score:.4f})")
        return best_features, best_feature_names, best_method, best_selector
    
    def train_single_model(self, model, model_name, X_train_scaled, y_train, 
                          X_test_scaled, y_test, is_neural_net=False, 
                          job: Optional[TrainingJob] = None) -> Dict:
        """Train a single model (adapted from Sampleeight.py)."""
        if job:
            if job.cancelled:
                job.status = "cancelled"
                job.message = "Training cancelled by user"
                job.end_time = datetime.now()
                return
            job.message = f"Training {model_name}..."
        
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
        
        # Calculate metrics
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
            'probabilities': y_pred_proba,
        }
    
    def create_deep_learning_model(self, input_dim):
        """Create deep learning model (from Sampleeight.py)."""
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
    
    def train_all_models(self, X_train_scaled, y_train, X_test_scaled, y_test, job: Optional[TrainingJob] = None) -> List[Dict]:
        """Train all models (from Sampleeight.py)."""
        results = []
        total_models = 9  # 9 base models + 1 ensemble
        
        # Helper to check cancellation
        def check_cancelled():
            if job and job.cancelled:
                raise Exception("Training cancelled by user")
        
        # Model 1: Logistic Regression
        check_cancelled()
        if job:
            job.progress = 45.0
            job.message = "Training Logistic Regression..."
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
        results.append(self.train_single_model(lr, "Logistic Regression", X_train_scaled, y_train, X_test_scaled, y_test, job=job))
        
        # Model 2: Decision Tree
        check_cancelled()
        if job:
            job.progress = 50.0
            job.message = "Training Decision Tree..."
        dt = DecisionTreeClassifier(max_depth=20, min_samples_split=10, min_samples_leaf=5,
                                    class_weight='balanced', random_state=RANDOM_STATE)
        results.append(self.train_single_model(dt, "Decision Tree", X_train_scaled, y_train, X_test_scaled, y_test, job=job))
        
        # Model 3: Random Forest
        check_cancelled()
        if job:
            job.progress = 55.0
            job.message = "Training Random Forest..."
        rf = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=5,
                                    min_samples_leaf=2, class_weight='balanced',
                                    random_state=RANDOM_STATE, n_jobs=-1)
        results.append(self.train_single_model(rf, "Random Forest", X_train_scaled, y_train, X_test_scaled, y_test, job=job))
        
        # Model 4: Gradient Boosting
        check_cancelled()
        if job:
            job.progress = 60.0
            job.message = "Training Gradient Boosting..."
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=10, learning_rate=0.1,
                                        subsample=0.8, random_state=RANDOM_STATE)
        results.append(self.train_single_model(gb, "Gradient Boosting", X_train_scaled, y_train, X_test_scaled, y_test, job=job))
        
        # Model 5: Naive Bayes
        check_cancelled()
        if job:
            job.progress = 65.0
            job.message = "Training Naive Bayes..."
        nb = GaussianNB()
        results.append(self.train_single_model(nb, "Naive Bayes", X_train_scaled, y_train, X_test_scaled, y_test, job=job))
        
        # Model 6: XGBoost
        check_cancelled()
        if HAS_XGB:
            if job:
                job.progress = 70.0
                job.message = "Training XGBoost..."
            xgb = XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                               random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss')
            results.append(self.train_single_model(xgb, "XGBoost", X_train_scaled, y_train, X_test_scaled, y_test, job=job))
        
        # Model 7: LightGBM
        check_cancelled()
        if HAS_LGBM:
            if job:
                job.progress = 75.0
                job.message = "Training LightGBM..."
            lgbm = LGBMClassifier(n_estimators=300, max_depth=15, learning_rate=0.05,
                                 num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                                 class_weight='balanced', random_state=RANDOM_STATE,
                                 n_jobs=-1, verbose=-1)
            results.append(self.train_single_model(lgbm, "LightGBM", X_train_scaled, y_train, X_test_scaled, y_test, job=job))
        
        # Model 8: CatBoost
        check_cancelled()
        if HAS_CATBOOST:
            if job:
                job.progress = 80.0
                job.message = "Training CatBoost..."
            catboost = CatBoostClassifier(iterations=300, depth=8, learning_rate=0.05,
                                         l2_leaf_reg=3,
                                         class_weights=[1, (y_train == 0).sum() / (y_train == 1).sum()],
                                         random_seed=RANDOM_STATE, verbose=0, thread_count=-1)
            results.append(self.train_single_model(catboost, "CatBoost", X_train_scaled, y_train, X_test_scaled, y_test, job=job))
        
        # Model 9: Deep Learning
        check_cancelled()
        if HAS_TENSORFLOW:
            if job:
                job.progress = 85.0
                job.message = "Training Deep Learning Model..."
            dl_model = self.create_deep_learning_model(X_train_scaled.shape[1])
            results.append(self.train_single_model(dl_model, "Deep Learning (FFN)", X_train_scaled, y_train,
                                                   X_test_scaled, y_test, is_neural_net=True, job=job))
        
        # Model 10: Ensemble
        check_cancelled()
        if job:
            job.progress = 90.0
            job.message = "Creating Ensemble Model..."
        if len(results) >= 3:
            sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)
            top_3 = sorted_results[:3]
            
            # Create ensemble by averaging probabilities from top 3 models
            ensemble_proba = np.mean([r['probabilities'] for r in top_3], axis=0)
            ensemble_pred = (ensemble_proba >= 0.5).astype(int)
            
            # Calculate ensemble metrics
            acc = accuracy_score(y_test, ensemble_pred)
            f1 = f1_score(y_test, ensemble_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_test, ensemble_pred, pos_label=1, zero_division=0)
            prec = precision_score(y_test, ensemble_pred, pos_label=1, zero_division=0)
            
            cm = confusion_matrix(y_test, ensemble_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # Create ensemble model wrapper that uses top 3 models
            ensemble_model = EnsembleModel(top_3)
            
            ensemble_results = {
                'model_name': 'Ensemble (Top 3)',
                'model': ensemble_model,
                'accuracy': acc,
                'roc_auc': roc_auc_score(y_test, ensemble_proba) if len(np.unique(y_test)) > 1 else 0,
                'mcc': matthews_corrcoef(y_test, ensemble_pred),
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'training_time': 0,
                'predictions': ensemble_pred,
                'probabilities': ensemble_proba,
            }
            results.append(ensemble_results)
        
        if job:
            job.progress = 95.0
            job.message = "Training complete!"
        
        return results
    
    def save_models(self, results, scaler, feature_cols, job: Optional[TrainingJob] = None):
        """Save trained models - save all 10 models."""
        if job:
            job.progress = 96.0
            job.message = "Saving models..."
        
        # Save all models (not just top 5)
        saved_count = 0
        for i, result in enumerate(results, 1):
            if result['model'] is not None:
                # Clean model name for filename
                model_name_clean = result['model_name'].replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
                filename = os.path.join(self.models_dir, f"model_{i}_{model_name_clean}.pkl")
                
                artifacts = {
                    'model': result['model'],
                    'scaler': scaler,
                    'feature_columns': feature_cols,
                    'metrics': {k: v for k, v in result.items() if k not in ['model', 'predictions', 'probabilities']},
                }
                
                try:
                    joblib.dump(artifacts, filename)
                    saved_count += 1
                    logger.info(f"Saved: {filename}")
                except Exception as e:
                    logger.warning(f"Failed to save {filename}: {e}")
        
        # Also save comparison CSV
        try:
            import pandas as pd
            results_df = pd.DataFrame([{
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
            results_df = results_df.sort_values('F1', ascending=False)
            csv_path = os.path.join(self.models_dir, 'all_models_comparison.csv')
            results_df.to_csv(csv_path, index=False)
            logger.info(f"Saved comparison CSV: {csv_path}")
        except Exception as e:
            logger.warning(f"Failed to save comparison CSV: {e}")
        
        if job:
            job.progress = 100.0
            job.message = f"Models saved successfully! ({saved_count} models saved)"
    
    def train_models(self, dataset_path: str, job_id: str, repo_url: Optional[str] = None, 
                     max_commits: Optional[int] = None, since: Optional[datetime] = None, 
                     until: Optional[datetime] = None, commit_hash: Optional[str] = None,
                     function_level: bool = False, jira_url: Optional[str] = None,
                     jira_username: Optional[str] = None, jira_api_token: Optional[str] = None) -> TrainingJob:
        """Main training function that runs in background thread."""
        job = TrainingJob(job_id, dataset_path)
        job.repo_url = repo_url
        
        with self.jobs_lock:
            self.jobs[job_id] = job
        
        def training_thread():
            # Capture outer scope variables to avoid UnboundLocalError
            # These are parameters from train_models method
            outer_function_level = function_level
            outer_max_commits = max_commits
            outer_since = since
            outer_until = until
            outer_commit_hash = commit_hash
            outer_jira_url = jira_url
            outer_jira_username = jira_username
            outer_jira_api_token = jira_api_token
            outer_repo_url = repo_url
            
            try:
                job.status = "running"
                job.start_time = datetime.now()
                
                # Check if cancelled before starting
                if job.cancelled:
                    job.status = "cancelled"
                    job.message = "Training cancelled by user"
                    job.end_time = datetime.now()
                    return
                
                # Initialize dataset_path (will be set by SZZ or use default)
                dataset_path = None
                
                # If repo_url provided, run SZZ analysis first
                if outer_repo_url:
                    # Store advanced config in job for commit fetching
                    job._max_commits = outer_max_commits
                    job._since = outer_since
                    job._until = outer_until
                    job._commit_hash = outer_commit_hash
                    job._function_level = outer_function_level
                    job._jira_url = outer_jira_url
                    job._jira_username = outer_jira_username
                    job._jira_api_token = outer_jira_api_token
                    job.message = "Running SZZ analysis..."
                    job.progress = 2.0
                    
                    try:
                        # Try absolute imports first, then relative
                        try:
                            from git_service import GitService
                            from szz_service import SZZService
                            from jira_service import JiraService
                        except ImportError:
                            try:
                                from .git_service import GitService
                                from .szz_service import SZZService
                                from .jira_service import JiraService
                            except ImportError:
                                import sys
                                webapp_dir = os.path.dirname(os.path.abspath(__file__))
                                sys.path.insert(0, webapp_dir)
                                from git_service import GitService
                                from szz_service import SZZService
                                from jira_service import JiraService
                        
                        logger.info("Initializing Git and SZZ services...")
                        git_service = GitService()
                        
                        # Initialize Jira service if credentials provided
                        jira_service = None
                        jira_enabled = False
                        if hasattr(job, '_jira_url') and job._jira_url and job._jira_username and job._jira_api_token:
                            try:
                                jira_service = JiraService(
                                    jira_url=job._jira_url,
                                    username=job._jira_username,
                                    api_token=job._jira_api_token
                                )
                                jira_enabled = True
                                logger.info("Jira service initialized for enhanced bug detection")
                                job.message = "Jira integration enabled - Enhanced bug detection active"
                            except Exception as e:
                                logger.warning(f"Failed to initialize Jira service: {e}")
                                job.message = f"Jira initialization failed: {str(e)}. Continuing without Jira."
                        else:
                            if hasattr(job, '_jira_url') and job._jira_url:
                                logger.info("Jira URL provided but credentials incomplete. Jira integration disabled.")
                                job.message = "Jira credentials incomplete. Continuing without Jira integration."
                        
                        szz_service = SZZService(git_service, jira_service=jira_service)
                        
                        # Store Jira status for later reporting
                        job._jira_enabled = jira_enabled
                        job._jira_issues_found = 0
                        job._jira_issues_used = 0
                        
                        # Validate repository URL
                        if not git_service.is_valid_repo_url(outer_repo_url):
                            raise ValueError(
                                f"Invalid repository URL format: {outer_repo_url}. "
                                "Please use a valid Git repository URL (e.g., https://github.com/user/repo.git)"
                            )
                        
                        # Clone repository
                        job.message = "Cloning repository..."
                        job.progress = 5.0
                        logger.info(f"Cloning repository: {outer_repo_url}")
                        try:
                            repo_path = git_service.clone_repository(outer_repo_url)
                            logger.info(f"Repository cloned to: {repo_path}")
                            job.repo_path = repo_path
                            
                            # Fetch commits after cloning
                            job.message = "Fetching commit history..."
                            job.progress = 6.0
                            logger.info("Fetching commits from repository...")
                            
                            # Use advanced configuration if provided, otherwise default to 10
                            # Check if max_commits was explicitly set (could be None if not provided)
                            if hasattr(job, '_max_commits') and job._max_commits is not None:
                                commit_count = int(job._max_commits)
                            else:
                                commit_count = 10  # Default to 10 commits
                            
                            logger.info(f"Fetching {commit_count} commits from repository")
                            since_date = job._since if hasattr(job, '_since') else None
                            until_date = job._until if hasattr(job, '_until') else None
                            specific_hash = job._commit_hash if hasattr(job, '_commit_hash') else None
                            
                            # If specific commit hash provided, get that commit and recent ones
                            if specific_hash:
                                try:
                                    # Get the specific commit
                                    commit_info = git_service.get_commit_info(repo_path, specific_hash)
                                    commits = [{
                                        'hash': commit_info['hash'],
                                        'short_hash': commit_info['short_hash'],
                                        'author': commit_info['author'],
                                        'date': commit_info['date'].isoformat() if isinstance(commit_info['date'], datetime) else str(commit_info['date']),
                                        'message': commit_info['message']
                                    }]
                                    # Also get recent commits
                                    recent_commits = git_service.get_commit_history(
                                        repo_path, 
                                        max_count=commit_count - 1,
                                        since=since_date,
                                        until=until_date
                                    )
                                    # Combine and remove duplicates
                                    seen_hashes = {commit_info['hash']}
                                    for c in recent_commits:
                                        if c['hash'] not in seen_hashes:
                                            commits.append({
                                                'hash': c['hash'],
                                                'short_hash': c['hash'][:7],
                                                'author': c['author'],
                                                'date': c['date'].isoformat() if isinstance(c['date'], datetime) else str(c['date']),
                                                'message': c['message']
                                            })
                                            seen_hashes.add(c['hash'])
                                except Exception as e:
                                    logger.warning(f"Could not get specific commit {specific_hash}: {e}, fetching recent commits instead")
                                    commits = git_service.get_commit_history(
                                        repo_path,
                                        max_count=commit_count,
                                        since=since_date,
                                        until=until_date
                                    )
                                    commits = [{
                                        'hash': c['hash'],
                                        'short_hash': c['hash'][:7],
                                        'author': c['author'],
                                        'date': c['date'].isoformat() if isinstance(c['date'], datetime) else str(c['date']),
                                        'message': c['message']
                                    } for c in commits]
                            else:
                                # Get recent commits based on configuration
                                logger.info(f"Fetching {commit_count} commits (since: {since_date}, until: {until_date})")
                                commits_raw = git_service.get_commit_history(
                                    repo_path,
                                    max_count=commit_count,
                                    since=since_date,
                                    until=until_date
                                )
                                logger.info(f"Git service returned {len(commits_raw)} commits")
                                
                                # Convert to serializable format
                                commits = []
                                for c in commits_raw:
                                    try:
                                        date_val = c.get('date')
                                        if isinstance(date_val, datetime):
                                            date_str = date_val.isoformat()
                                        else:
                                            date_str = str(date_val) if date_val else ''
                                        
                                        commits.append({
                                            'hash': c.get('hash', ''),
                                            'short_hash': c.get('hash', '')[:7] if c.get('hash') else '',
                                            'author': c.get('author', 'Unknown'),
                                            'date': date_str,
                                            'message': c.get('message', 'No message')
                                        })
                                    except Exception as e:
                                        logger.warning(f"Error processing commit: {e}")
                                        continue
                                
                                logger.info(f"Processed {len(commits)} commits for display")
                            
                            job.commits = commits
                            logger.info(f"Fetched {len(commits)} commits from repository")
                            
                        except Exception as clone_error:
                            error_msg = str(clone_error)
                            logger.error(f"Failed to clone repository: {error_msg}")
                            
                            # Provide helpful error messages
                            if "Permission denied" in error_msg or "authentication" in error_msg.lower():
                                raise Exception(
                                    f"Authentication failed for repository: {repo_url}. "
                                    "For private repositories, ensure SSH keys are configured or use HTTPS with credentials."
                                )
                            elif "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                                raise Exception(
                                    f"Repository not found: {repo_url}. "
                                    "Please verify the URL is correct and the repository exists."
                                )
                            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                                raise Exception(
                                    f"Network error while cloning: {repo_url}. "
                                    "Please check your internet connection and try again."
                                )
                            else:
                                raise Exception(
                                    f"Failed to clone repository: {error_msg}. "
                                    "Please verify the URL is correct and accessible."
                                )
                        
                        # Run SZZ analysis with progress callback
                        def update_szz_progress(progress, message):
                            """Update job progress during SZZ analysis."""
                            if job.cancelled:
                                raise Exception("Training cancelled by user")
                            job.progress = progress
                            job.message = message
                            logger.info(f"SZZ Progress: {progress:.1f}% - {message}")
                        
                        job.message = "Starting SZZ analysis..."
                        job.progress = 10.0
                        logger.info("Starting SZZ analysis...")
                        szz_results = szz_service.analyze_repository(
                            repo_path, 
                            max_commits=1000,
                            progress_callback=update_szz_progress
                        )
                        logger.info(f"SZZ analysis found {len(szz_results.get('bug_fix_commits', []))} bug-fix commits")
                        
                        # Log Jira statistics if available
                        szz_stats = szz_results.get('statistics', {})
                        if szz_stats.get('jira_enabled'):
                            jira_enhanced = szz_stats.get('jira_enhanced_commits', 0)
                            logger.info(f"Jira integration enhanced {jira_enhanced} bug-fix commit detections")
                            job.message = f"SZZ complete: {len(szz_results.get('bug_fix_commits', []))} bug-fix commits found ({jira_enhanced} enhanced by Jira)"
                        else:
                            job.message = f"SZZ complete: {len(szz_results.get('bug_fix_commits', []))} bug-fix commits found"
                        
                        # Create labeled dataset
                        job.message = "Creating labeled dataset from SZZ results..."
                        job.progress = 20.0
                        logger.info("Creating labeled dataset from SZZ results...")
                        # Get function_level from job or use parameter from outer scope
                        use_function_level = job._function_level if hasattr(job, '_function_level') else function_level
                        dataset_path = szz_service.create_labeled_dataset(
                            repo_path,
                            szz_results,
                            output_path=os.path.join(BASE_DIR, "szz_labeled_data.csv"),
                            function_level=use_function_level
                        )
                        
                        if dataset_path and os.path.exists(dataset_path):
                            logger.info(f"Labeled dataset created: {dataset_path}")
                            # Check dataset size and merge with default dataset
                            try:
                                df_szz = pd.read_csv(dataset_path)
                                szz_samples = len(df_szz)
                                logger.info(f"SZZ dataset contains {szz_samples} samples")
                                
                                # Always merge with default dataset for better training
                                default_path = os.path.join(BASE_DIR, "all_projects_combined.csv")
                                if os.path.exists(default_path):
                                    logger.info(f"Merging SZZ dataset with default dataset: {default_path}")
                                    df_default = pd.read_csv(default_path)
                                    default_samples = len(df_default)
                                    logger.info(f"Default dataset contains {default_samples} samples")
                                    
                                    # Ensure both datasets have same columns
                                    # Get all unique columns from both datasets
                                    all_cols = set(df_default.columns) | set(df_szz.columns)
                                    
                                    # Ensure 'label' column exists in both
                                    if 'label' not in df_szz.columns:
                                        df_szz['label'] = 'CLEAN'  # Default if missing
                                    if 'label' not in df_default.columns:
                                        logger.warning("Default dataset missing 'label' column")
                                    
                                    # Add missing columns with appropriate default values
                                    for col in all_cols:
                                        if col not in df_szz.columns:
                                            # Use 0 for numeric columns, '' for string columns
                                            if col == 'label':
                                                df_szz[col] = 'CLEAN'
                                            elif col in ['file_path', 'commit_hash', 'commit_message', 'author', 'date']:
                                                df_szz[col] = ''
                                            else:
                                                df_szz[col] = 0
                                        
                                        if col not in df_default.columns:
                                            if col == 'label':
                                                df_default[col] = 'CLEAN'
                                            elif col in ['file_path', 'commit_hash', 'commit_message', 'author', 'date']:
                                                df_default[col] = ''
                                            else:
                                                df_default[col] = 0
                                    
                                    # Reorder columns to match default dataset order (for consistency)
                                    if 'label' in df_default.columns:
                                        # Ensure label column is at the end or appropriate position
                                        cols_order = [c for c in df_default.columns if c != 'label'] + ['label']
                                        df_default = df_default[cols_order]
                                        df_szz = df_szz[cols_order]
                                    
                                    # Merge datasets
                                    df_combined = pd.concat([df_default, df_szz], ignore_index=True)
                                    combined_samples = len(df_combined)
                                    logger.info(f"Combined dataset contains {combined_samples} samples (default: {default_samples}, SZZ: {szz_samples})")
                                    
                                    # Save merged dataset
                                    merged_path = os.path.join(BASE_DIR, "combined_training_data.csv")
                                    df_combined.to_csv(merged_path, index=False)
                                    dataset_path = merged_path
                                    job.message = f"Merged dataset ready: {default_samples} default + {szz_samples} SZZ samples = {combined_samples} total"
                                else:
                                    logger.warning("Default dataset not found, using only SZZ dataset")
                                    if szz_samples < 50:
                                        logger.warning(f"SZZ dataset is very small ({szz_samples} samples). Results may not be reliable.")
                                
                            except Exception as e:
                                logger.error(f"Error processing SZZ dataset: {e}", exc_info=True)
                                dataset_path = None
                        else:
                            logger.warning("Dataset path is None or file does not exist")
                            dataset_path = None
                        
                        # Cleanup (ignore errors on Windows due to file locking)
                        try:
                            git_service.cleanup_repo(repo_path)
                            logger.info("Repository cleanup completed")
                        except Exception as cleanup_error:
                            logger.warning(f"Could not cleanup repository (this is OK on Windows): {cleanup_error}")
                        
                        if dataset_path:
                            job.message = "SZZ analysis complete. Starting training..."
                            job.progress = 25.0
                        else:
                            job.message = "SZZ analysis completed but no dataset created. Using default dataset..."
                            job.progress = 25.0
                            dataset_path = None
                        
                    except Exception as e:
                        error_msg = str(e)
                        # Check if this is a cancellation exception
                        if "Training cancelled by user" in error_msg or job.cancelled:
                            logger.info("SZZ analysis cancelled by user")
                            job.status = "cancelled"
                            job.message = "Training cancelled by user"
                            job.end_time = datetime.now()
                            return  # Exit the training thread immediately
                        
                        logger.error(f"SZZ analysis failed: {error_msg}", exc_info=True)
                        import traceback
                        full_error = traceback.format_exc()
                        logger.error(f"Full SZZ error traceback:\n{full_error}")
                        
                        # Don't silently fail - report the error clearly
                        job.message = f"SZZ analysis failed: {error_msg}. Check logs for details. Using default dataset..."
                        job.progress = 25.0
                        # Continue with existing dataset but log the failure clearly
                        dataset_path = None  # Will use default dataset
                        logger.warning("Falling back to default dataset due to SZZ failure")
                
                # Check if cancelled before proceeding
                if job.cancelled:
                    job.status = "cancelled"
                    job.message = "Training cancelled by user"
                    job.end_time = datetime.now()
                    return
                
                job.message = "Loading and preparing data..."
                job.progress = 30.0
                
                # Load data
                X, y, feature_cols, df = self.load_and_prepare_data(dataset_path)
                
                # Check if cancelled after loading
                if job.cancelled:
                    job.status = "cancelled"
                    job.message = "Training cancelled by user"
                    job.end_time = datetime.now()
                    return
                
                job.message = "Splitting data..."
                job.progress = 35.0
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
                )
                
                # Convert to DataFrame for feature selection
                X_train_df = pd.DataFrame(X_train, columns=feature_cols)
                X_test_df = pd.DataFrame(X_test, columns=feature_cols)
                
                # Automated feature selection
                job.message = "Selecting best features..."
                job.progress = 36.0
                
                try:
                    # Try to find best feature selection method
                    X_train_selected, selected_feature_names, best_method, feature_selector = self.select_best_feature_method(
                        X_train_df, y_train, X_test_df, y_test, 
                        methods=['selectkbest', 'rfe', 'importance'],
                        job=job
                    )
                    
                    # Apply same selection to test set
                    if feature_selector is not None and hasattr(feature_selector, 'transform'):
                        X_test_selected = pd.DataFrame(
                            feature_selector.transform(X_test_df),
                            columns=selected_feature_names
                        )
                    else:
                        # Manual selection based on feature names
                        X_test_selected = X_test_df[selected_feature_names]
                    
                    # Update feature columns
                    feature_cols = selected_feature_names
                    logger.info(f"Selected {len(feature_cols)} features using method: {best_method}")
                    logger.info(f"Selected features: {feature_cols}")
                    
                    # Convert back to numpy for resampling
                    X_train = X_train_selected.values if isinstance(X_train_selected, pd.DataFrame) else X_train_selected
                    X_test = X_test_selected.values if isinstance(X_test_selected, pd.DataFrame) else X_test_selected
                    
                except Exception as e:
                    logger.warning(f"Feature selection failed: {e}. Using all features.")
                    # Fallback: use all features
                    X_train = X_train_df.values if isinstance(X_train_df, pd.DataFrame) else X_train
                    X_test = X_test_df.values if isinstance(X_test_df, pd.DataFrame) else X_test
                
                # Check if cancelled before balancing
                if job.cancelled:
                    job.status = "cancelled"
                    job.message = "Training cancelled by user"
                    job.end_time = datetime.now()
                    return
                
                job.message = "Balancing classes..."
                job.progress = 38.0
                
                # Balance classes
                sampler = SMOTETomek(random_state=RANDOM_STATE)
                X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
                
                # Check if cancelled after balancing
                if job.cancelled:
                    job.status = "cancelled"
                    job.message = "Training cancelled by user"
                    job.end_time = datetime.now()
                    return
                
                job.message = "Scaling features..."
                job.progress = 40.0
                
                # Check if cancelled before scaling
                if job.cancelled:
                    job.status = "cancelled"
                    job.message = "Training cancelled by user"
                    job.end_time = datetime.now()
                    return
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_resampled)
                X_test_scaled = scaler.transform(X_test)
                
                # Check if cancelled before training
                if job.cancelled:
                    job.status = "cancelled"
                    job.message = "Training cancelled by user"
                    job.end_time = datetime.now()
                    return
                
                # Train models
                results = self.train_all_models(X_train_scaled, y_train_resampled, X_test_scaled, y_test, job=job)
                
                # Check if cancelled after training
                if job.cancelled:
                    job.status = "cancelled"
                    job.message = "Training cancelled by user"
                    job.end_time = datetime.now()
                    return
                results_sorted = sorted(results, key=lambda x: x['f1'], reverse=True)
                
                # Save models
                self.save_models(results_sorted, scaler, feature_cols, job=job)
                
                # Trigger model reload in Flask app (if running)
                # Use a callback approach to avoid circular imports
                try:
                    import sys
                    if 'app' in sys.modules:
                        app_module = sys.modules['app']
                        if hasattr(app_module, 'reload_models'):
                            app_module.reload_models()
                            logger.info("Models reloaded in Flask app")
                except Exception as e:
                    logger.warning(f"Could not reload models in Flask app: {e}")
                
                # Prepare results
                job.results = {
                    'best_model': results_sorted[0]['model_name'],
                    'best_f1': float(results_sorted[0]['f1']),
                    'best_accuracy': float(results_sorted[0]['accuracy']),
                    'best_recall': float(results_sorted[0]['recall']),
                    'best_precision': float(results_sorted[0]['precision']),
                    'best_roc_auc': float(results_sorted[0]['roc_auc']),
                    'total_models': len(results_sorted),
                    'all_models': [
                        {
                            'name': r['model_name'],
                            'f1': float(r['f1']),
                            'accuracy': float(r['accuracy']),
                            'recall': float(r['recall']),
                            'precision': float(r['precision']),
                        }
                        for r in results_sorted
                    ]
                }
                
                if job.cancelled:
                    job.status = "cancelled"
                    job.message = "Training cancelled by user"
                else:
                    job.status = "completed"
                    job.message = "Training completed successfully!"
                job.end_time = datetime.now()
                
            except Exception as e:
                if job.cancelled:
                    job.status = "cancelled"
                    job.message = "Training cancelled by user"
                else:
                    job.status = "failed"
                    job.error = str(e)
                    job.message = f"Training failed: {str(e)}"
                    import traceback
                    job.error = traceback.format_exc()
                job.end_time = datetime.now()
        
        job.thread = threading.Thread(target=training_thread, daemon=True)
        job.thread.start()
        
        return job
    
    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get status of a training job."""
        with self.jobs_lock:
            return self.jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job."""
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            
            if job.status in ["completed", "failed", "cancelled"]:
                return False  # Cannot cancel already finished jobs
            
            job.cancelled = True
            job.status = "cancelled"
            job.message = "Training cancelled by user"
            job.end_time = datetime.now()
            logger.info(f"Job {job_id} cancelled by user")
            return True
    
    def list_jobs(self) -> List[Dict]:
        """List all training jobs."""
        with self.jobs_lock:
            return [job.to_dict() for job in self.jobs.values()]


# Global training service instance
_training_service = None

def get_training_service() -> TrainingService:
    """Get or create global training service instance."""
    global _training_service
    if _training_service is None:
        _training_service = TrainingService()
    return _training_service

