import os
import re
import uuid
import logging
from typing import Dict, List, Tuple, Optional

from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np
import pandas as pd
from feature_extractor import SAFE_FEATURES, build_feature_vector
from training_service import get_training_service

logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "truly_clean_models")


def load_models() -> Dict[str, Dict]:
    """Load the saved model artifacts if they exist."""
    models: Dict[str, Dict] = {}
    if not os.path.isdir(MODELS_DIR):
        logger.warning(f"Models directory not found: {MODELS_DIR}")
        return models

    loaded_count = 0
    for filename in os.listdir(MODELS_DIR):
        if not filename.endswith(".pkl"):
            continue
        path = os.path.join(MODELS_DIR, filename)
        try:
            artifact = joblib.load(path)
            
            # Verify artifact has required components
            if not isinstance(artifact, dict):
                logger.warning(f"Invalid artifact format in {filename}")
                continue
                
            model = artifact.get("model")
            scaler = artifact.get("scaler")
            
            if model is None:
                logger.warning(f"Model is None in {filename}")
                continue
            if scaler is None:
                logger.warning(f"Scaler is None in {filename}")
                continue

            metrics = artifact.get("metrics", {})
            pretty_name = metrics.get("model_name", filename.replace(".pkl", "").replace("model_", "").replace("_", " "))
            key = re.sub(r"[^a-z0-9_]+", "_", pretty_name.lower())
            
            # Ensure unique key
            original_key = key
            counter = 1
            while key in models:
                key = f"{original_key}_{counter}"
                counter += 1
            
            models[key] = {
                "artifact": artifact,
                "pretty_name": pretty_name,
                "metrics": metrics,
            }
            loaded_count += 1
            logger.info(f"Loaded model: {pretty_name} (key: {key})")
        except Exception as e:
            logger.error(f"Failed to load model from {filename}: {e}", exc_info=True)
            continue

    logger.info(f"Successfully loaded {loaded_count} models from {MODELS_DIR}")
    return models

def to_serializable(obj):
    """Recursively convert numpy/pandas scalars to native Python types."""
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(to_serializable(item) for item in obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray, pd.Series)):
        return [to_serializable(item) for item in obj.tolist()]
    return obj


app = Flask(__name__, template_folder="templates", static_folder="static")
MODEL_STORE = load_models()


def reload_models():
    """Reload models from disk."""
    global MODEL_STORE
    MODEL_STORE = load_models()


def _make_prediction_for_features(features_df: pd.DataFrame, stats: Dict, selection: str, model_store: Dict) -> Optional[Dict]:
    """
    Helper function to make predictions for given features.
    
    Args:
        features_df: DataFrame with features
        stats: Dictionary with additional stats
        selection: Model selection ('all' or specific model key)
        model_store: Dictionary of loaded models
    
    Returns:
        Dictionary with predictions or None if error
    """
    if not model_store:
        return None
    
    # Handle model selection
    if selection == "all":
        selected_models = list(model_store.keys())
    else:
        if selection not in model_store:
            return None
        selected_models = [selection]
    
    predictions = {}
    seen_model_names = set()  # Track seen model names to prevent duplicates
    
    for model_id in selected_models:
        model_info = model_store.get(model_id)
        if not model_info:
            continue
        
        artifact = model_info["artifact"]
        model = artifact.get("model")
        scaler = artifact.get("scaler")
        
        if model is None or scaler is None:
            continue
        
        # Deduplicate by model name - keep only the first occurrence
        model_name = model_info["pretty_name"]
        if model_name in seen_model_names:
            logger.debug(f"Skipping duplicate model: {model_name} (key: {model_id})")
            continue
        seen_model_names.add(model_name)
        
        try:
            feature_cols = artifact.get("feature_columns", SAFE_FEATURES)
            
            # Check if all required features are present
            missing_features = [f for f in feature_cols if f not in features_df.columns]
            if missing_features:
                for feat in missing_features:
                    features_df[feat] = 0.0
            
            # Ensure features are in correct order
            features_for_pred = features_df[feature_cols].copy()
            
            # Validate feature values
            for col in features_for_pred.columns:
                if features_for_pred[col].dtype == 'object':
                    features_for_pred[col] = pd.to_numeric(features_for_pred[col], errors='coerce')
            
            features_for_pred = features_for_pred.replace([np.inf, -np.inf], np.nan)
            features_for_pred = features_for_pred.fillna(0)
            
            # Scale features
            features_scaled = scaler.transform(features_for_pred)
            
            # Make prediction
            if hasattr(model, "predict_proba"):
                proba_array = model.predict_proba(features_scaled)
                if proba_array.shape[1] >= 2:
                    probability = float(proba_array[0][1])
                else:
                    probability = float(proba_array[0][0])
            else:
                raw = float(model.predict(features_scaled)[0])
                probability = float(raw) if raw in [0, 1] else max(0.0, min(1.0, raw))
            
            label = "BUGGY" if probability >= 0.5 else "CLEAN"
            
            predictions[model_id] = {
                "name": model_info["pretty_name"],
                "probability": probability,
                "label": label,
            }
            
        except Exception as e:
            logger.error(f"Error making prediction with {model_info['pretty_name']}: {e}")
            continue
    
    if not predictions:
        return None
    
    return {
        "stats": to_serializable(stats),
        "predictions": to_serializable(predictions)
    }


@app.route("/", methods=["GET"])
def index():
    # Ensure unique model options (by label) to prevent duplicates
    seen_labels = set()
    model_options = []
    for key, info in MODEL_STORE.items():
        label = info["pretty_name"]
        if label not in seen_labels:
            seen_labels.add(label)
            model_options.append({"id": key, "label": label})
    model_options.sort(key=lambda item: item["label"])
    return render_template("index.html", model_options=model_options)


@app.route("/api/train", methods=["POST"])
def train():
    """Start training models on a dataset."""
    try:
        from datetime import datetime
        
        # Get dataset path and repo URL from request
        if request.is_json:
            data = request.json
            dataset_path = data.get("dataset_path")
            repo_url = data.get("repo_url")
            max_commits = data.get("max_commits")
            start_date = data.get("start_date")
            end_date = data.get("end_date")
            commit_hash = data.get("commit_hash")
            granularity = data.get("granularity", "file")
            jira_url = data.get("jira_url")
            jira_username = data.get("jira_username")
            jira_api_token = data.get("jira_api_token")
            enable_jira = data.get("enable_jira", False)
        else:
            dataset_path = request.form.get("dataset_path")
            repo_url = request.form.get("repo_url")
            max_commits = request.form.get("max_commits")
            start_date = request.form.get("start_date")
            end_date = request.form.get("end_date")
            commit_hash = request.form.get("commit_hash")
            granularity = request.form.get("granularity", "file")
            jira_url = request.form.get("jira_url")
            jira_username = request.form.get("jira_username")
            jira_api_token = request.form.get("jira_api_token")
            enable_jira = request.form.get("enable_jira", "false").lower() == "true"
        
        # Parse advanced configuration
        max_commits_int = None
        if max_commits:
            try:
                max_commits_int = int(max_commits)
                # Ensure it's a positive number
                if max_commits_int <= 0:
                    max_commits_int = None
            except (ValueError, TypeError):
                pass
        
        # If max_commits not provided or invalid, default to 10
        if max_commits_int is None:
            max_commits_int = 10
        
        since_date = None
        if start_date:
            try:
                since_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                try:
                    since_date = datetime.strptime(start_date, '%Y-%m-%d')
                except ValueError:
                    pass
        
        until_date = None
        if end_date:
            try:
                until_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                try:
                    until_date = datetime.strptime(end_date, '%Y-%m-%d')
                except ValueError:
                    pass
        
        if not dataset_path:
            # Default to the existing dataset
            dataset_path = os.path.join(BASE_DIR, "all_projects_combined.csv")
        
        # Convert to absolute path if relative
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(BASE_DIR, dataset_path)
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            return jsonify({
                "error": f"Dataset file not found: {dataset_path}. Please provide a valid path.",
                "status": "error"
            }), 400
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Get granularity setting
        function_level = (granularity == "function")
        
        # Start training
        training_service = get_training_service()
        job = training_service.train_models(
            dataset_path, 
            job_id, 
            repo_url=repo_url,
            max_commits=max_commits_int,
            since=since_date,
            until=until_date,
            commit_hash=commit_hash,
            function_level=function_level,
            jira_url=jira_url if enable_jira and jira_url else None,
            jira_username=jira_username if enable_jira and jira_username else None,
            jira_api_token=jira_api_token if enable_jira and jira_api_token else None
        )
        
        return jsonify({
            "job_id": job_id,
            "status": "started",
            "message": "Training job started",
            "dataset_path": dataset_path
        }), 202
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


@app.route("/api/training/status/<job_id>", methods=["GET"])
def training_status(job_id):
    """Get status of a training job."""
    try:
        training_service = get_training_service()
        job = training_service.get_job_status(job_id)
        
        if not job:
            return jsonify({
                "error": f"Job {job_id} not found",
                "status": "not_found"
            }), 404
        
        return jsonify(to_serializable(job.to_dict())), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


@app.route("/api/training/stop/<job_id>", methods=["POST"])
def stop_training(job_id):
    """Stop/cancel a running training job."""
    try:
        training_service = get_training_service()
        success = training_service.cancel_job(job_id)
        
        if not success:
            return jsonify({
                "error": f"Job {job_id} not found or cannot be cancelled",
                "status": "error"
            }), 404
        
        return jsonify({
            "message": "Training job cancelled successfully",
            "status": "cancelled",
            "job_id": job_id
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


@app.route("/api/training/jobs", methods=["GET"])
def list_training_jobs():
    """List all training jobs."""
    try:
        training_service = get_training_service()
        jobs = training_service.list_jobs()
        
        return jsonify({
            "jobs": [to_serializable(job) for job in jobs],
            "count": len(jobs)
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


@app.route("/api/repo/commits", methods=["GET"])
def get_repo_commits():
    """Get commits from a repository."""
    try:
        repo_url = request.args.get("repo_url")
        job_id = request.args.get("job_id")
        max_commits = request.args.get("max_commits", 10, type=int)
        
        if not repo_url and not job_id:
            return jsonify({
                "error": "Either repo_url or job_id must be provided",
                "status": "error"
            }), 400
        
        # If job_id provided, get commits from job
        if job_id:
            training_service = get_training_service()
            job = training_service.get_job_status(job_id)
            if job and job.commits:
                return jsonify({
                    "commits": to_serializable(job.commits),
                    "count": len(job.commits),
                    "repo_url": job.repo_url
                }), 200
            else:
                return jsonify({
                    "error": f"Job {job_id} not found or has no commits",
                    "status": "error"
                }), 404
        
        # If repo_url provided, get commits directly
        if repo_url:
            try:
                from git_service import GitService
            except ImportError:
                from .git_service import GitService
            
            git_service = GitService()
            repo_name = git_service._extract_repo_name(repo_url)
            repo_path = os.path.join(git_service.base_dir, repo_name)
            
            if not os.path.exists(repo_path):
                return jsonify({
                    "error": f"Repository not cloned yet. Please start training first.",
                    "status": "error"
                }), 404
            
            commits = git_service.get_commit_history(repo_path, max_count=max_commits)
            commits_serialized = [{
                'hash': c['hash'],
                'short_hash': c['hash'][:7],
                'author': c['author'],
                'date': c['date'].isoformat() if hasattr(c['date'], 'isoformat') else str(c['date']),
                'message': c['message']
            } for c in commits]
            
            return jsonify({
                "commits": to_serializable(commits_serialized),
                "count": len(commits_serialized),
                "repo_url": repo_url
            }), 200
        
    except Exception as e:
        logger.error(f"Error getting commits: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    if "code_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file_storage = request.files["code_file"]
    selection = request.form.get("model", "all")
    code_bytes = file_storage.read()
    base_file = request.files.get("base_code")
    commit_message = request.form.get("commit_message", "")
    repo_url = request.form.get("repo_url") or (request.json.get("repo_url") if request.is_json else None)
    file_path = request.form.get("file_path") or (request.json.get("file_path") if request.is_json else None)

    if not code_bytes:
        return jsonify({"error": "Uploaded file is empty"}), 400

    try:
        code_text = code_bytes.decode("utf-8")
    except UnicodeDecodeError:
        code_text = code_bytes.decode("latin-1")

    before_text = ""
    if base_file and base_file.filename:
        base_bytes = base_file.read()
        if base_bytes:
            try:
                before_text = base_bytes.decode("utf-8")
            except UnicodeDecodeError:
                before_text = base_bytes.decode("latin-1")

    # Check if function-level prediction is requested
    function_level = request.form.get("function_level", "false").lower() == "true" or \
                    (request.is_json and request.json.get("function_level", False))
    
    # Build features (file-level or function-level)
    if function_level:
        try:
            from feature_extractor import build_function_level_features
            from function_extractor import FunctionExtractor
            
            # Detect language from filename
            language = None
            if file_storage.filename:
                ext = file_storage.filename.split('.')[-1].lower()
                lang_map = {'py': 'python', 'java': 'java', 'js': 'javascript', 'cpp': 'cpp', 'c': 'c'}
                language = lang_map.get(ext, 'python')
            
            # Extract function-level features
            function_features = build_function_level_features(
                before_text,
                code_text,
                commit_message,
                language=language,
                filename=file_storage.filename
            )
            
            # Make predictions for each function
            function_predictions = []
            for features_df, func_stats, func_info in function_features:
                # Make prediction for this function
                func_prediction_result = _make_prediction_for_features(
                    features_df, func_stats, selection, MODEL_STORE
                )
                if func_prediction_result and func_prediction_result.get('predictions'):
                    # Extract predictions from the result
                    predictions_dict = func_prediction_result['predictions']
                    
                    # Create a prediction entry for each model
                    for model_id, pred_data in predictions_dict.items():
                        function_pred = {
                            'function_name': func_info.name,
                            'function_start_line': func_info.start_line,
                            'function_end_line': func_info.end_line,
                            'is_method': func_info.is_method,
                            'class_name': func_info.class_name or "",
                            'name': pred_data.get('name', 'Unknown Model'),
                            'probability': pred_data.get('probability', 0.0),
                            'label': pred_data.get('label', 'CLEAN'),
                            'model_id': model_id
                        }
                        function_predictions.append(to_serializable(function_pred))
            
            if function_predictions:
                return jsonify({
                    "filename": file_storage.filename,
                    "function_level": True,
                    "function_count": len(function_predictions),
                    "predictions": function_predictions,
                    "stats": func_stats if 'func_stats' in locals() else {}
                })
            else:
                # Fallback to file-level if no functions found
                function_level = False
        
        except Exception as e:
            logger.warning(f"Function-level prediction failed: {e}. Falling back to file-level.")
            function_level = False
    
    if not function_level:
        # Build base features (file-level)
        features_df, stats = build_feature_vector(before_text, code_text, commit_message)
    
    # Add process metrics if repo URL and file path provided
    if repo_url and file_path:
        try:
            try:
                from git_service import GitService
                from process_metrics import ProcessMetricsService
            except ImportError:
                from .git_service import GitService
                from .process_metrics import ProcessMetricsService
            
            git_service = GitService()
            metrics_service = ProcessMetricsService(git_service)
            
            # Try to find cloned repo or clone it
            repo_name = git_service._extract_repo_name(repo_url)
            repo_path = os.path.join(git_service.base_dir, repo_name)
            
            if os.path.exists(repo_path):
                process_metrics = metrics_service.get_all_process_metrics(
                    repo_path,
                    file_path
                )
                # Add process metrics to stats
                stats.update({
                    'churn_30d': process_metrics.get('churn', 0),
                    'developer_count_90d': process_metrics.get('developer_count', 0),
                    'commit_frequency_30d': process_metrics.get('commit_frequency', 0.0),
                    'file_age_days': process_metrics.get('file_age_days', 0)
                })
        except Exception as e:
            logger.warning(f"Could not add process metrics: {e}")
            # Continue without process metrics

    if not MODEL_STORE:
        return jsonify({"error": "No trained models available. Train models first."}), 500

    # Handle model selection - ensure it's a valid key
    if selection == "all":
        selected_models = list(MODEL_STORE.keys())
    else:
        # Validate the selected model exists
        if selection not in MODEL_STORE:
            logger.warning(f"Selected model '{selection}' not found. Available models: {list(MODEL_STORE.keys())}")
            return jsonify({"error": f"Model '{selection}' not found. Please select a valid model."}), 400
        selected_models = [selection]

    predictions = {}
    seen_model_names = set()  # Track seen model names to prevent duplicates
    
    for model_id in selected_models:
        model_info = MODEL_STORE.get(model_id)
        if not model_info:
            logger.warning(f"Model {model_id} not found in MODEL_STORE")
            continue

        artifact = model_info["artifact"]
        model = artifact.get("model")
        scaler = artifact.get("scaler")

        if model is None or scaler is None:
            logger.warning(f"Model or scaler is None for {model_info['pretty_name']}")
            continue

        # Deduplicate by model name - keep only the first occurrence
        model_name = model_info["pretty_name"]
        if model_name in seen_model_names:
            logger.debug(f"Skipping duplicate model: {model_name} (key: {model_id})")
            continue
        seen_model_names.add(model_name)

        try:
            # Ensure all required features are present and in correct order
            feature_cols = artifact.get("feature_columns", SAFE_FEATURES)
            
            # Log feature columns for debugging
            logger.debug(f"Model {model_info['pretty_name']} expects features: {feature_cols}")
            logger.debug(f"Available features in input: {list(features_df.columns)}")
            
            # Check if all required features are present
            missing_features = [f for f in feature_cols if f not in features_df.columns]
            if missing_features:
                logger.warning(f"Missing features in prediction for {model_info['pretty_name']}: {missing_features}")
                # Fill missing features with zeros
                for feat in missing_features:
                    features_df[feat] = 0.0
                    logger.info(f"Added missing feature '{feat}' with default value 0.0")
            
            # Ensure features are in the correct order (must match training order)
            features_for_pred = features_df[feature_cols].copy()
            
            # Validate feature values
            for col in features_for_pred.columns:
                if features_for_pred[col].dtype == 'object':
                    logger.warning(f"Feature {col} is object type, converting to numeric")
                    features_for_pred[col] = pd.to_numeric(features_for_pred[col], errors='coerce')
            
            # Validate feature values (replace inf/nan)
            features_for_pred = features_for_pred.replace([np.inf, -np.inf], np.nan)
            features_for_pred = features_for_pred.fillna(0)
            
            # Scale features
            features_scaled = scaler.transform(features_for_pred)
            
            if hasattr(model, "predict_proba"):
                proba_array = model.predict_proba(features_scaled)
                # Ensure we get probability of class 1 (BUGGY)
                # proba_array shape: (n_samples, n_classes) = (1, 2)
                # proba_array[0] = [prob_class_0, prob_class_1]
                if proba_array.shape[1] >= 2:
                    probability = float(proba_array[0][1])  # Probability of class 1 (BUGGY)
                else:
                    # Binary classification with single output
                    probability = float(proba_array[0][0])
            else:
                # Model only has predict, not predict_proba
                raw = float(model.predict(features_scaled)[0])
                # If raw is 0 or 1, convert to probability
                if raw in [0, 1]:
                    probability = float(raw)
                else:
                    # Assume it's already a probability
                    probability = max(0.0, min(1.0, raw))
            
            # Label mapping: probability >= 0.5 means BUGGY (class 1)
            # This matches training: label_binary = (label == 'BUGGY').astype(int)
            # So class 1 = BUGGY, class 0 = CLEAN
            # Using 0.5 threshold, but we could make this configurable
            label = "BUGGY" if probability >= 0.5 else "CLEAN"
            
            # Log prediction details for debugging
            logger.debug(f"Prediction for {model_info['pretty_name']}: probability={probability:.4f}, label={label}")
            
            prediction_payload = {
                "name": model_info["pretty_name"],
                "probability": probability,
                "label": label,
            }
            predictions[model_id] = to_serializable(prediction_payload)
            
        except Exception as e:
            logger.error(f"Error making prediction with {model_info['pretty_name']}: {e}", exc_info=True)
            continue

    if not predictions:
        return jsonify({"error": "Selected model(s) unavailable."}), 400

    # Get confusion matrix and feature importance if available
    confusion_matrix = None
    feature_importance = None
    
    if selection != "all" and selection in MODEL_STORE:
        model_info = MODEL_STORE[selection]
        metrics = model_info.get("metrics", {})
        
        # Try to get confusion matrix from metrics
        if 'tp' in metrics and 'tn' in metrics and 'fp' in metrics and 'fn' in metrics:
            confusion_matrix = [
                metrics['tn'],
                metrics['fp'],
                metrics['fn'],
                metrics['tp']
            ]
        
        # Try to get feature importance (if tree-based model)
        try:
            artifact = model_info["artifact"]
            model = artifact.get("model")
            if hasattr(model, 'feature_importances_'):
                feature_cols = artifact.get("feature_columns", SAFE_FEATURES)
                feature_importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
        except:
            pass
    
    response_payload = {
        "filename": file_storage.filename,
        "stats": to_serializable(stats),
        "predictions": to_serializable(predictions),
    }
    
    if confusion_matrix:
        response_payload["confusion_matrix"] = to_serializable(confusion_matrix)
    
    if feature_importance:
        response_payload["feature_importance"] = to_serializable(feature_importance)
    
    return jsonify(to_serializable(response_payload))


if __name__ == "__main__":
    app.run(debug=True)

