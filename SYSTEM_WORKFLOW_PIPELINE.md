# System Workflow & Pipeline Documentation

## 📋 **Overview**

This document describes the complete workflow and pipeline of the Performance Bug Detection System, from data collection to model training to prediction.

---

## 🔄 **Complete System Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Web App)                      │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │   Train Models       │      │   Predict Bugs        │        │
│  └──────────────────────┘      └──────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND SERVICES                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Training     │  │ SZZ          │  │ Jira         │         │
│  │ Service      │  │ Service      │  │ Service      │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Feature      │  │ Function     │  │ Git          │         │
│  │ Extractor    │  │ Extractor    │  │ Service      │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Labeled      │  │ Feature      │  │ Model        │         │
│  │ Dataset      │  │ Vectors      │  │ Artifacts    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 **WORKFLOW 1: Training Pipeline**

### **Phase 1: User Input & Configuration**

```
User Actions:
├── Enters Git Repository URL (optional)
├── Selects Model Type (All Models / Specific Model)
├── Configures Advanced Settings:
│   ├── Time/Commit Range (start date, end date, commit hash)
│   ├── Bug Keywords (custom keywords for SZZ)
│   ├── Granularity (File-Level / Function-Level)
│   ├── Feature Selection (checkboxes for features)
│   └── Jira Integration (URL, username, API token)
└── Clicks "Start Analysis & Training"
```

**Backend Processing:**
- Receives request at `/api/train` endpoint
- Validates input parameters
- Generates unique Job ID
- Creates `TrainingJob` object
- Starts background training thread

---

### **Phase 2: Data Collection & Preparation**

#### **2.1 Repository Cloning (if repo URL provided)**

```
Progress: 0-5%
├── Validate repository URL format
├── Clone repository to local storage
├── Extract repository name
└── Store repository path in job
```

**Services Used:**
- `GitService.clone_repository()`

---

#### **2.2 Commit History Fetching**

```
Progress: 5-10%
├── Fetch commits based on configuration:
│   ├── Max commits limit (default: 10)
│   ├── Date range (since/until)
│   ├── Specific commit hash
│   └── Time-based filtering
├── Parse commit metadata:
│   ├── Commit hash
│   ├── Author
│   ├── Date
│   ├── Message
│   └── File changes
└── Store commits in job object
```

**Services Used:**
- `GitService.get_commits()`
- `ProcessMetricsService` (for process metrics)

---

#### **2.3 SZZ Algorithm Analysis**

```
Progress: 10-25%
├── Initialize SZZ Service
│   └── Optionally initialize Jira Service (if configured)
│
├── Step 1: Identify Bug-Fix Commits (10-15%)
│   ├── Analyze commit messages for keywords:
│   │   ├── General: "fix", "bug", "error", "defect", "issue"
│   │   └── Performance: "performance", "slow", "latency", "bottleneck"
│   ├── Extract Jira issue keys (if Jira enabled)
│   ├── Fetch Jira issue details (if Jira enabled)
│   ├── Verify bug-fix status from Jira
│   └── Mark commits as bug-fix commits
│
├── Step 2: Find Bug-Inducing Commits (15-20%)
│   ├── For each bug-fix commit:
│   │   ├── Get diff (changes in bug-fix commit)
│   │   ├── Identify lines that were fixed
│   │   ├── Use `git blame` to trace buggy lines
│   │   ├── Find commits that introduced those lines
│   │   └── Mark as bug-inducing commits
│   └── Create mapping: bug-fix → bug-inducing commits
│
└── Step 3: Create Labeled Dataset (20-25%)
    ├── For each commit in repository:
    │   ├── Check if it's a bug-inducing commit
    │   ├── Label: "BUGGY" if bug-inducing, "CLEAN" otherwise
    │   └── Extract features (see Phase 3)
    ├── Handle granularity:
    │   ├── File-Level: One row per file change
    │   └── Function-Level: One row per function/method change
    └── Save to CSV: "szz_labeled_data.csv"
```

**Services Used:**
- `SZZService.analyze_repository()`
- `SZZService.identify_bug_fix_commits()`
- `SZZService.find_bug_inducing_commits()`
- `SZZService.create_labeled_dataset()`
- `JiraService` (optional, for enhanced detection)

**Key Features:**
- **Jira Integration**: Enhances bug-fix detection by verifying issues in Jira
- **Performance Keywords**: Specifically identifies performance-related bugs
- **Function-Level**: Extracts functions/methods for granular analysis

---

### **Phase 3: Feature Extraction**

```
Progress: 25-35%
├── Load labeled dataset
├── For each commit/function:
│   ├── Extract Static Code Metrics:
│   │   ├── NLOC (Number of Lines of Code)
│   │   ├── Complexity (cyclomatic complexity)
│   │   ├── Token Count
│   │   ├── Added Lines
│   │   ├── Deleted Lines
│   │   └── Net Lines (added - deleted)
│   │
│   ├── Extract Commit Message Metrics:
│   │   ├── Commit Message Length
│   │   ├── Number of Words
│   │   └── Keyword Counts
│   │
│   ├── Extract Process Metrics (if repo available):
│   │   ├── Churn (30 days)
│   │   ├── Developer Count (90 days)
│   │   ├── Commit Frequency (30 days)
│   │   └── File Age (days)
│   │
│   ├── Extract Performance-Aware Metrics:
│   │   ├── Synchronization Constructs Count
│   │   ├── Max Loop Nesting Depth
│   │   └── Nested Loops Count
│   │
│   └── Create feature vector
│
└── Combine with default dataset (if exists)
```

**Services Used:**
- `FeatureExtractor.build_feature_vector()`
- `FeatureExtractor.build_function_level_features()` (if function-level)
- `ProcessMetricsService.get_all_process_metrics()`

**Feature Categories:**
1. **Static Metrics**: Code complexity, size, structure
2. **Process Metrics**: Development history, team activity
3. **Performance Metrics**: Synchronization, loop complexity
4. **Commit Metrics**: Message analysis, change patterns

---

### **Phase 4: Data Preprocessing**

```
Progress: 35-40%
├── Split Data:
│   ├── Train Set (80%)
│   └── Test Set (20%)
│
├── Handle Missing Values:
│   ├── Fill NaN with 0
│   └── Replace inf/-inf with 0
│
├── Feature Selection (if enabled):
│   ├── Methods Available:
│   │   ├── SelectKBest (Mutual Information, Chi-squared, F-test)
│   │   ├── RFE (Recursive Feature Elimination)
│   │   └── Feature Importance (Random Forest)
│   ├── Select best features automatically
│   └── Reduce feature space
│
└── Class Balancing:
    ├── Check class distribution
    ├── Apply SMOTETomek (if imbalanced):
    │   ├── SMOTE: Oversample minority class
    │   └── Tomek Links: Remove borderline samples
    └── Balance BUGGY vs CLEAN classes
```

**Services Used:**
- `TrainingService.perform_feature_selection()`
- `SMOTETomek` from imblearn

---

### **Phase 5: Feature Scaling**

```
Progress: 40-45%
├── Initialize StandardScaler
├── Fit scaler on training data
├── Transform training features
├── Transform test features
└── Store scaler for prediction use
```

**Purpose:**
- Normalize features to same scale
- Improve model performance
- Required for distance-based algorithms

---

### **Phase 6: Model Training**

```
Progress: 45-95%
├── Train Multiple Models (in parallel):
│   ├── Random Forest
│   ├── XGBoost
│   ├── LightGBM
│   ├── CatBoost
│   ├── Gradient Boosting
│   ├── Decision Tree
│   ├── Logistic Regression
│   ├── Naive Bayes
│   ├── Deep Learning (FFN)
│   └── Ensemble (Top 3)
│
├── For Each Model:
│   ├── Train on training set
│   ├── Predict on test set
│   ├── Calculate Metrics:
│   │   ├── Accuracy
│   │   ├── Precision
│   │   ├── Recall
│   │   ├── F1-Score
│   │   ├── ROC-AUC
│   │   ├── MCC (Matthews Correlation Coefficient)
│   │   └── Confusion Matrix
│   └── Store model + metrics
│
└── Select Best Model (by F1-Score)
```

**Services Used:**
- `TrainingService.train_models()`
- Various ML libraries (scikit-learn, XGBoost, LightGBM, etc.)

**Model Selection:**
- Best model determined by F1-Score
- All models saved for ensemble predictions

---

### **Phase 7: Model Saving & Results**

```
Progress: 95-100%
├── Save Model Artifacts:
│   ├── Trained model (pickle/joblib)
│   ├── Scaler (for feature normalization)
│   ├── Feature columns (for prediction)
│   ├── Metrics (accuracy, precision, recall, etc.)
│   └── Metadata (model name, training date)
│
├── Save to Directory:
│   └── "truly_clean_models/"
│
├── Generate Results Summary:
│   ├── Model comparison table
│   ├── Best model metrics
│   ├── Feature importance (if available)
│   └── Confusion matrix
│
└── Return Results to UI:
    ├── Job status: "completed"
    ├── Best model metrics
    ├── All model results
    └── Training statistics
```

**Output:**
- Model files: `model_<name>.pkl`
- Results CSV: `model_results.csv`
- JSON response with metrics

---

## 🔮 **WORKFLOW 2: Prediction Pipeline**

### **Phase 1: User Input**

```
User Actions:
├── Uploads Code File (current version)
├── Optionally uploads Base Code File (previous version)
├── Optionally enters Commit Message
├── Selects Model(s):
│   ├── Single Model (specific model)
│   └── All Models (ensemble prediction)
├── Optionally checks "Function-Level Analysis"
└── Clicks "Predict Performance Bugs"
```

**Backend Processing:**
- Receives request at `/predict` endpoint
- Validates file upload
- Reads file content
- Determines analysis granularity

---

### **Phase 2: Feature Extraction**

```
├── Determine Granularity:
│   ├── File-Level (default)
│   └── Function-Level (if checkbox checked)
│
├── If Function-Level:
│   ├── Extract functions/methods from code
│   ├── For each function:
│   │   ├── Extract function features
│   │   ├── Get function metadata (name, class, lines)
│   │   └── Create feature vector
│   └── Return list of function features
│
└── If File-Level:
    ├── Extract file-level features
    ├── Get code statistics
    └── Create single feature vector
```

**Services Used:**
- `FeatureExtractor.build_feature_vector()` (file-level)
- `FeatureExtractor.build_function_level_features()` (function-level)
- `FunctionExtractor.extract_functions_from_diff()` (function-level)

**Features Extracted:**
- Same as training pipeline (static, process, performance metrics)

---

### **Phase 3: Model Loading**

```
├── Load models from "truly_clean_models/" directory
├── For each model file:
│   ├── Load model artifact
│   ├── Extract:
│   │   ├── Trained model
│   │   ├── Scaler
│   │   ├── Feature columns
│   │   └── Model metadata
│   └── Store in MODEL_STORE
│
└── Filter by selection:
    ├── Single Model: Load only selected model
    └── All Models: Load all available models
```

**Services Used:**
- `load_models()` function in `app.py`

---

### **Phase 4: Feature Preparation**

```
For each feature vector (file or function):
├── Check required features:
│   ├── Compare with model's expected features
│   ├── Add missing features (set to 0)
│   └── Remove extra features
│
├── Order features:
│   └── Match model's feature column order
│
├── Validate feature values:
│   ├── Convert object types to numeric
│   ├── Replace inf/-inf with 0
│   └── Fill NaN with 0
│
└── Scale features:
    └── Apply same scaler used during training
```

**Purpose:**
- Ensure features match training format
- Normalize features for prediction

---

### **Phase 5: Prediction**

```
For each model (if "All Models" selected):
├── Make Prediction:
│   ├── Use model.predict_proba() (if available)
│   │   └── Get probability of class 1 (BUGGY)
│   └── Or use model.predict() (binary)
│       └── Convert to probability
│
├── Determine Label:
│   ├── Probability >= 0.5 → "BUGGY"
│   └── Probability < 0.5 → "CLEAN"
│
└── Store Result:
    ├── Model name
    ├── Probability
    ├── Label (BUGGY/CLEAN)
    └── Function metadata (if function-level)
```

**Deduplication:**
- Remove duplicate models (same name, different keys)
- Keep only first occurrence

---

### **Phase 6: Results Formatting**

```
├── Structure Results:
│   ├── File-Level:
│   │   ├── Filename
│   │   ├── Code statistics
│   │   └── Predictions (one per model)
│   │
│   └── Function-Level:
│       ├── Filename
│       ├── Function count
│       └── Predictions (one per function per model)
│
└── Return JSON Response:
    ├── filename
    ├── function_level (boolean)
    ├── predictions (array or object)
    ├── stats (code metrics)
    └── function_count (if function-level)
```

---

### **Phase 7: UI Rendering**

```
Frontend Processing:
├── Receive JSON response
├── Determine View Mode:
│   ├── Card View (default)
│   └── Table View (toggle)
│
├── Render Results:
│   ├── File-Level Card View:
│   │   ├── Code metrics grid
│   │   └── Prediction cards (one per model)
│   │       ├── Model name
│   │       ├── Bug probability (%)
│   │       ├── Progress bar
│   │       └── Prediction button (BUGGY/CLEAN)
│   │
│   ├── Function-Level Card View:
│   │   ├── Function sections (one per function)
│   │   ├── Function metadata (name, class, lines)
│   │   └── Prediction cards (one per model per function)
│   │
│   └── Table View:
│       ├── Sortable columns
│       ├── Filterable rows
│       └── Detailed predictions
│
└── Display to User
```

**Features:**
- Real-time updates
- Interactive filtering/sorting
- Visual progress bars
- Color-coded predictions

---

## 🔗 **Integration Points**

### **SZZ ↔ Jira Integration**

```
SZZ Algorithm:
├── Analyzes commit messages
├── Extracts Jira issue keys (e.g., "PROJ-123")
│
Jira Service:
├── Fetches issue details from Jira API
├── Verifies issue type (Bug/Performance)
├── Checks issue status (Resolved/Closed)
└── Returns issue information
│
SZZ Algorithm:
└── Uses Jira data to enhance bug-fix detection
    ├── More accurate identification
    └── Performance issue detection
```

### **Training ↔ Prediction Integration**

```
Training Pipeline:
├── Trains models
├── Saves models with:
│   ├── Model artifact
│   ├── Scaler
│   ├── Feature columns
│   └── Metadata
│
Prediction Pipeline:
├── Loads saved models
├── Uses same scaler
├── Uses same feature columns
└── Ensures consistency
```

---

## 📊 **Data Flow Diagram**

```
┌─────────────┐
│ Git Repo    │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│ Git Service │────▶│ SZZ Service │
└─────────────┘     └──────┬──────┘
                            │
                            ▼
                    ┌─────────────┐
                    │ Labeled     │
                    │ Dataset     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Feature     │
                    │ Extractor   │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Training    │
                    │ Service     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Trained     │
                    │ Models      │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Prediction  │
                    │ Service     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Results     │
                    │ (UI)        │
                    └─────────────┘
```

---

## ⚙️ **Configuration Options**

### **Training Configuration**

| Option | Description | Default |
|--------|-------------|---------|
| `repo_url` | Git repository URL | None |
| `model_type` | Model to train | "all" |
| `granularity` | File or function level | "file" |
| `max_commits` | Max commits to analyze | 10 |
| `start_date` | Start date filter | None |
| `end_date` | End date filter | None |
| `commit_hash` | Specific commit | None |
| `bug_keywords` | Custom keywords | Default list |
| `jira_url` | Jira instance URL | None |
| `jira_username` | Jira username | None |
| `jira_api_token` | Jira API token | None |
| `enable_jira` | Enable Jira integration | False |

### **Prediction Configuration**

| Option | Description | Default |
|--------|-------------|---------|
| `code_file` | Code file to analyze | Required |
| `base_code` | Previous version (optional) | None |
| `commit_message` | Commit message (optional) | None |
| `model` | Model selection | "all" |
| `function_level` | Function-level analysis | False |

---

## 🎯 **Key Features**

### **1. Multi-Granularity Analysis**
- **File-Level**: Analyze entire files
- **Function-Level**: Analyze individual functions/methods
- Automatic function extraction (Python, Java, C++, etc.)

### **2. Multiple ML Models**
- 10+ different algorithms
- Ensemble predictions
- Automatic model selection

### **3. Enhanced Bug Detection**
- SZZ algorithm for bug-inducing commit detection
- Jira integration for verification
- Performance-specific keyword detection

### **4. Comprehensive Features**
- Static code metrics
- Process metrics (churn, developer count)
- Performance-aware metrics (synchronization, loops)
- Commit message analysis

### **5. Real-Time Progress**
- Background job processing
- Progress updates via polling
- Cancellable training jobs

---

## 📈 **Performance Metrics**

### **Training Metrics**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- MCC (Matthews Correlation Coefficient)

### **System Metrics**
- Training time per model
- Prediction latency
- Memory usage
- Model file sizes

---

## 🔄 **Error Handling & Recovery**

### **Training Errors**
- Invalid repository URL → Error message
- Jira connection failure → Fallback to keyword detection
- Model training failure → Continue with other models
- Cancellation → Graceful stop, cleanup

### **Prediction Errors**
- Missing features → Fill with defaults
- Invalid model → Skip, continue with others
- File parsing error → Error message to user

---

## 🚦 **Status Flow**

### **Training Job Status**

```
pending → running → completed
              │
              ├─→ cancelled
              │
              └─→ error
```

### **Progress Indicators**

- **0-5%**: Repository cloning
- **5-10%**: Commit fetching
- **10-15%**: Bug-fix identification
- **15-20%**: Bug-inducing detection
- **20-25%**: Dataset creation
- **25-35%**: Feature extraction
- **35-40%**: Data preprocessing
- **40-45%**: Feature scaling
- **45-95%**: Model training
- **95-100%**: Model saving

---

## 📝 **Summary**

The system provides a **complete end-to-end pipeline** for:

1. **Collecting** code changes from Git repositories
2. **Labeling** commits using SZZ algorithm + Jira integration
3. **Extracting** comprehensive features (static, process, performance)
4. **Training** multiple ML models with automatic selection
5. **Predicting** bug probability for new code
6. **Visualizing** results in interactive UI

The pipeline is **modular**, **scalable**, and **production-ready**, with support for:
- Multiple granularity levels
- External integrations (Jira)
- Real-time progress tracking
- Cancellable operations
- Comprehensive error handling

---

---

## 🖥️ **WORKFLOW 3: User Interface Workflow**

### **Phase 1: Application Initialization**

```
User Opens Web Application:
├── Flask server running on http://localhost:5000
├── Browser loads index.html
├── Static assets loaded:
│   ├── styles.css (styling)
│   ├── app.js (main JavaScript)
│   └── results-table.js (table functionality)
│
└── UI Components Initialized:
    ├── Training form
    ├── Prediction form
    ├── Progress dashboard (hidden)
    ├── Results container (hidden)
    └── Event listeners attached
```

**Initial State:**
- Training form: Visible, ready for input
- Prediction form: Visible, ready for file upload
- Progress dashboard: Hidden (`display: none`)
- Results section: Hidden/empty

---

### **Phase 2: Training Workflow (UI Perspective)**

#### **2.1 User Input & Form Submission**

```
User Actions:
├── Navigate to "Train Models" section
├── Fill in Repository URL (optional):
│   └── Example: https://github.com/user/repo.git
│
├── Select Model Type:
│   ├── "All Models" (default)
│   └── Or specific model (Random Forest, XGBoost, etc.)
│
├── (Optional) Expand "Advanced Configuration":
│   ├── Click "▶ Advanced Configuration" button
│   ├── Section expands (display: block)
│   │
│   ├── Time/Commit Range:
│   │   ├── Start Date: [date picker]
│   │   ├── End Date: [date picker]
│   │   ├── Max Commits: [number input, default: 10]
│   │   └── Commit Hash: [text input]
│   │
│   ├── Bug Keywords:
│   │   └── [text input with default keywords]
│   │
│   ├── Analysis Granularity:
│   │   ├── ○ File-Level (default)
│   │   └── ○ Function/Method-Level
│   │
│   ├── Feature Selection:
│   │   ├── ☑ Added Lines
│   │   ├── ☑ Deleted Lines
│   │   ├── ☑ NLOC
│   │   ├── ☑ Complexity
│   │   └── ... (other features)
│   │
│   └── Jira Integration:
│       ├── Jira URL: [text input]
│       ├── Username: [text input]
│       ├── API Token: [password input]
│       └── ☐ Enable Jira integration
│
└── Click "Start Analysis & Training" button
```

**Form Validation:**
- Repository URL: Validated if provided (format check)
- Model Type: Always valid (dropdown selection)
- Advanced fields: Optional, validated if provided

---

#### **2.2 Form Submission & API Call**

```
JavaScript Event Handler:
├── Prevent default form submission
├── Collect form data:
│   ├── repo_url
│   ├── model_type
│   ├── start_date, end_date
│   ├── commit_hash
│   ├── bug_keywords
│   ├── granularity
│   ├── selected_features (checkboxes)
│   ├── max_commits
│   ├── jira_url, jira_username, jira_api_token
│   └── enable_jira (checkbox)
│
├── Create FormData object
├── Disable form (prevent multiple submissions)
├── Show loading state
│
└── Send POST request to /api/train:
    ├── Method: POST
    ├── Body: FormData
    ├── Headers: (auto-set by browser)
    └── Response: JSON with job_id
```

**UI State Changes:**
- Training form: Disabled (button disabled, inputs disabled)
- Progress dashboard: Shown (`display: block`)
- Progress bar: Reset to 0%
- Progress text: "Starting training..."
- Stop button: Shown, enabled

---

#### **2.3 Progress Tracking (Real-Time Updates)**

```
Polling Mechanism:
├── Store job_id from API response
├── Set up polling interval (every 2 seconds)
│
└── Poll GET /api/training/status/<job_id>:
    ├── Every 2 seconds
    ├── Parse JSON response:
    │   ├── status: "running" | "completed" | "error" | "cancelled"
    │   ├── progress: 0-100 (float)
    │   ├── message: "Current step description"
    │   ├── metrics: {accuracy, precision, recall, f1, ...}
    │   └── results: {best_model, all_models, ...}
    │
    └── Update UI:
        ├── Progress bar: Update width to progress%
        ├── Progress text: Update to message
        ├── Progress percentage: Update to progress%
        └── Status indicator: Update color/icon
```

**Progress Indicators:**

| Progress | Status Message | Visual Indicator |
|----------|---------------|------------------|
| 0-5% | "Cloning repository..." | Blue progress bar |
| 5-10% | "Fetching commits..." | Blue progress bar |
| 10-15% | "Identifying bug-fix commits..." | Blue progress bar |
| 15-20% | "Finding bug-inducing commits..." | Blue progress bar |
| 20-25% | "Creating labeled dataset..." | Blue progress bar |
| 25-35% | "Extracting features..." | Blue progress bar |
| 35-40% | "Preprocessing data..." | Blue progress bar |
| 40-45% | "Scaling features..." | Blue progress bar |
| 45-95% | "Training models..." | Blue progress bar |
| 95-100% | "Saving models..." | Blue progress bar |
| 100% | "Training completed!" | Green progress bar |

**Stop Button Functionality:**
- User clicks "Stop Training" button
- Sends POST request to `/api/training/stop/<job_id>`
- Button text changes to "Stopping..."
- Button disabled
- Polling continues until status is "cancelled"
- UI updates to show cancellation

---

#### **2.4 Results Display**

```
When Status = "completed":
├── Stop polling
├── Hide progress dashboard
├── Show training results section
│
├── Display Metrics Grid:
│   ├── Best Model Name
│   ├── Accuracy: XX.XX%
│   ├── Precision: XX.XX%
│   ├── Recall: XX.XX%
│   ├── F1-Score: XX.XX%
│   └── ROC-AUC: XX.XX%
│
├── Display Model Comparison Table:
│   ├── Model Name
│   ├── Accuracy
│   ├── Precision
│   ├── Recall
│   ├── F1-Score
│   └── ROC-AUC
│   (Sorted by F1-Score, descending)
│
├── Display Visualizations (if available):
│   ├── Performance Metrics Chart
│   ├── Confusion Matrix
│   └── Feature Importance Chart
│
└── Re-enable form (for new training)
```

**UI Components:**
- **Metrics Grid**: Card-based layout with metric cards
- **Results Table**: Sortable, filterable table
- **Charts**: Interactive visualizations (if Chart.js available)
- **Success Message**: Green banner with completion message

---

### **Phase 3: Prediction Workflow (UI Perspective)**

#### **3.1 User Input & File Upload**

```
User Actions:
├── Navigate to "Predict Bugs" section
├── Upload Code File:
│   ├── Click "Choose File" button
│   ├── File picker opens
│   ├── Select code file (.py, .java, .js, .cpp, etc.)
│   └── File name displayed: "Selected: filename.ext"
│
├── (Optional) Upload Base Code File:
│   ├── Click "Choose File" button
│   ├── Select previous version of file
│   └── File name displayed: "Selected: base_filename.ext"
│
├── (Optional) Enter Commit Message:
│   └── [Text area] Enter commit message
│
├── Select Model(s):
│   ├── Radio buttons in grid layout:
│   │   ├── ○ All Models (default)
│   │   ├── ○ Random Forest
│   │   ├── ○ XGBoost
│   │   ├── ○ LightGBM
│   │   └── ... (other models)
│   │
│   └── Only one selection allowed
│
├── (Optional) Check Function-Level Analysis:
│   └── ☐ Function-Level Analysis
│       └── Tooltip: "Analyze code at function/method level"
│
└── Click "Predict Performance Bugs" button
```

**Form Validation:**
- Code file: Required, must be selected
- Base file: Optional
- Commit message: Optional
- Model selection: Always valid (radio buttons)
- Function-level: Optional checkbox

---

#### **3.2 Form Submission & API Call**

```
JavaScript Event Handler:
├── Prevent default form submission
├── Validate file selected
│   └── If no file: Alert "Please choose a code file"
│
├── Collect form data:
│   ├── code_file (File object)
│   ├── base_code (File object, if provided)
│   ├── commit_message (string)
│   ├── model (radio button value)
│   └── function_level (checkbox checked state)
│
├── Create FormData object
│   └── Append all form fields
│
├── Disable form
├── Show loading state:
│   └── Results container shows: "Running predictions…"
│
└── Send POST request to /predict:
    ├── Method: POST
    ├── Body: FormData
    ├── Headers: (auto-set by browser)
    └── Response: JSON with predictions
```

**UI State Changes:**
- Prediction form: Disabled
- Results container: Shown, displays loading message
- File labels: Show selected file names

---

#### **3.3 Results Processing & Display**

```
When Response Received:
├── Parse JSON response
├── Store data: window.lastPredictionData = payload
│
├── Determine View Mode:
│   ├── Check window.currentViewMode
│   ├── Default: 'card' (card view)
│   └── Or: 'table' (table view)
│
└── Render Results:
    ├── If function_level = true:
    │   ├── Group predictions by function
    │   ├── Display function sections
    │   ├── Show function metadata (name, class, lines)
    │   └── Show prediction cards per function
    │
    └── If function_level = false (file-level):
        ├── Display code metrics grid
        └── Display prediction cards (one per model)
```

**Card View Layout:**

```
┌─────────────────────────────────────────┐
│  Results for: filename.ext              │
│  [Card View] [Table View] ← Toggle     │
├─────────────────────────────────────────┤
│  Code Metrics:                          │
│  ┌──────┐ ┌──────┐ ┌──────┐           │
│  │ NLOC │ │Complex│ │Tokens│           │
│  │ 150  │ │  12  │ │ 450  │           │
│  └──────┘ └──────┘ └──────┘           │
├─────────────────────────────────────────┤
│  Predictions:                           │
│  ┌──────────────────────────────────┐  │
│  │ Random Forest                     │  │
│  │ Bug Probability: 15.23%          │  │
│  │ [Progress Bar: 15%]              │  │
│  │ [CLEAN Button]                   │  │
│  └──────────────────────────────────┘  │
│  ┌──────────────────────────────────┐  │
│  │ XGBoost                           │  │
│  │ Bug Probability: 8.45%           │  │
│  │ [Progress Bar: 8%]               │  │
│  │ [CLEAN Button]                   │  │
│  └──────────────────────────────────┘  │
│  ... (more models)                      │
└─────────────────────────────────────────┘
```

**Table View Layout:**

```
┌─────────────────────────────────────────┐
│  Results for: filename.ext              │
│  [Card View] [Table View] ← Toggle     │
├─────────────────────────────────────────┤
│  Filter: [All Results ▼]                │
│  Sort by: [Probability ▼]               │
├─────────────────────────────────────────┤
│  Model        │ Probability │ Prediction│
│  Random Forest│   15.23%    │  [CLEAN]  │
│  XGBoost      │    8.45%    │  [CLEAN]  │
│  LightGBM     │   22.10%    │  [CLEAN]  │
│  ...                                    │
└─────────────────────────────────────────┘
```

**Function-Level View:**

```
┌─────────────────────────────────────────┐
│  Function-Level Results for: file.java  │
│  Found 5 function(s) with 45 prediction(s)│
│  [Card View] [Table View]               │
├─────────────────────────────────────────┤
│  Function: add                           │
│  Class: Calculator                      │
│  Lines: 10-15                           │
│  ┌──────────────────────────────────┐  │
│  │ Random Forest    │ 12.5% │ [CLEAN]│  │
│  │ XGBoost         │  8.2% │ [CLEAN]│  │
│  └──────────────────────────────────┘  │
├─────────────────────────────────────────┤
│  Function: subtract                     │
│  Class: Calculator                      │
│  Lines: 17-22                           │
│  ... (more functions)                   │
└─────────────────────────────────────────┘
```

---

#### **3.4 View Toggle Functionality**

```
User Clicks View Toggle:
├── Card View Button:
│   ├── Set window.currentViewMode = 'card'
│   ├── Update button states (active/inactive)
│   └── Re-render results in card format
│
└── Table View Button:
    ├── Set window.currentViewMode = 'table'
    ├── Update button states (active/inactive)
    └── Re-render results in table format
        ├── Initialize ResultsTable instance
        ├── Render sortable table
        ├── Add filter controls
        └── Display predictions in rows
```

**Table Features:**
- **Sorting**: Click column headers to sort
- **Filtering**: Dropdown to filter by prediction (All/Buggy/Clean)
- **Search**: (if implemented) Search by model name
- **Responsive**: Adapts to screen size

---

### **Phase 4: UI Component States**

#### **4.1 Training Form States**

```
State 1: Initial (Ready)
├── Form: Enabled
├── Button: "Start Analysis & Training" (enabled)
├── Inputs: All enabled
└── Progress: Hidden

State 2: Submitting
├── Form: Disabled
├── Button: "Start Analysis & Training" (disabled)
├── Inputs: All disabled
└── Progress: Visible, 0%

State 3: Training
├── Form: Disabled
├── Button: "Start Analysis & Training" (disabled)
├── Stop Button: Visible, enabled
├── Progress: Visible, updating (0-100%)
└── Message: Dynamic status messages

State 4: Completed
├── Form: Enabled (ready for new training)
├── Button: "Start Analysis & Training" (enabled)
├── Progress: Hidden
├── Results: Visible
└── Stop Button: Hidden

State 5: Error
├── Form: Enabled
├── Button: "Start Analysis & Training" (enabled)
├── Progress: Hidden
└── Error Message: Displayed in red banner
```

---

#### **4.2 Prediction Form States**

```
State 1: Initial (Ready)
├── Form: Enabled
├── Button: "Predict Performance Bugs" (enabled)
├── File Inputs: Enabled
└── Results: Hidden/Empty

State 2: Submitting
├── Form: Disabled
├── Button: "Predict Performance Bugs" (disabled)
├── File Inputs: Disabled
└── Results: Loading message

State 3: Results Displayed
├── Form: Enabled (ready for new prediction)
├── Button: "Predict Performance Bugs" (enabled)
├── Results: Visible with predictions
└── View Toggle: Active

State 4: Error
├── Form: Enabled
├── Button: "Predict Performance Bugs" (enabled)
└── Error Message: Displayed in red banner
```

---

### **Phase 5: Interactive Features**

#### **5.1 Model Selection (Radio Buttons)**

```
Layout: Grid of radio buttons
├── Visual Design:
│   ├── Rounded corners
│   ├── Hover effects
│   ├── Selected state: Purple border
│   └── Unselected state: Gray border
│
├── Behavior:
│   ├── Only one selection allowed
│   ├── Click to select
│   └── Visual feedback on selection
│
└── Options:
    ├── "All Models" (default)
    ├── Individual models (if available)
    └── Models loaded from MODEL_STORE
```

---

#### **5.2 Progress Bar Animation**

```
Visual Elements:
├── Container: Light gray background
├── Fill: Colored bar (blue during progress)
├── Percentage Text: Overlay on bar
└── Status Message: Below bar

Animation:
├── Smooth width transition (CSS transition)
├── Updates every 2 seconds (polling)
└── Color changes:
    ├── Blue: 0-99% (in progress)
    └── Green: 100% (completed)
```

---

#### **5.3 Prediction Cards**

```
Card Structure:
├── Header:
│   └── Model Name (e.g., "Random Forest")
│
├── Content:
│   ├── Probability Label: "Bug Probability:"
│   ├── Probability Value: "15.23%"
│   ├── Progress Bar:
│   │   ├── Width: probability%
│   │   └── Color: 
│   │       ├── Red: >= 50% (high risk)
│   │       ├── Yellow: 25-50% (medium risk)
│   │       └── Green: < 25% (low risk)
│   │
│   └── Prediction Button:
│       ├── Text: "BUGGY" or "CLEAN"
│       ├── Color:
│       │   ├── Red: BUGGY
│       │   └── Green: CLEAN
│       └── Rounded corners
│
└── Styling:
    ├── White background
    ├── Shadow for depth
    ├── Rounded corners
    └── Hover effects
```

---

#### **5.4 Advanced Configuration Toggle**

```
Button: "▶ Advanced Configuration"
├── Initial State:
│   ├── Icon: ▶ (right arrow)
│   ├── Content: Hidden (display: none)
│   └── Button: Clickable
│
├── On Click:
│   ├── Toggle content visibility
│   ├── Change icon:
│   │   ├── ▶ → ▼ (if expanding)
│   │   └── ▼ → ▶ (if collapsing)
│   └── Smooth animation (CSS transition)
│
└── Content Sections:
    ├── Time/Commit Range
    ├── Bug Keywords
    ├── Granularity
    ├── Feature Selection
    └── Jira Integration
```

---

### **Phase 6: Error Handling (UI)**

#### **6.1 Error Display**

```
Error Scenarios:
├── Network Error:
│   ├── Display: "Network error. Please check your connection."
│   └── Location: Red banner at top
│
├── Validation Error:
│   ├── Display: "Please choose a code file."
│   └── Location: Alert dialog or inline message
│
├── Server Error:
│   ├── Display: "Server error: [error message]"
│   └── Location: Red banner in results section
│
└── Training Error:
    ├── Display: "Training failed: [error details]"
    └── Location: Progress dashboard or results section
```

**Error Banner Design:**
- Background: Light red (#fee2e2)
- Border: Red (#ef4444)
- Text: Dark red
- Icon: ⚠️ Warning icon
- Dismissible: (optional) X button

---

#### **6.2 Loading States**

```
Loading Indicators:
├── Training:
│   ├── Progress bar with percentage
│   ├── Spinning icon (optional)
│   └── Status message
│
└── Prediction:
    ├── Text: "Running predictions…"
    ├── Spinner (optional)
    └── Card with loading message
```

---

### **Phase 7: Responsive Design**

```
Screen Size Adaptations:
├── Desktop (> 1024px):
│   ├── Two-column layout
│   ├── Full feature set visible
│   └── Side-by-side forms
│
├── Tablet (768px - 1024px):
│   ├── Single column layout
│   ├── Stacked forms
│   └── Reduced padding
│
└── Mobile (< 768px):
    ├── Single column
    ├── Full-width inputs
    ├── Stacked cards
    └── Touch-friendly buttons
```

---

### **Phase 8: User Journey Examples**

#### **Example 1: Complete Training Workflow**

```
1. User opens application
   └── Sees training form

2. User enters repository URL
   └── https://github.com/user/repo.git

3. User expands Advanced Configuration
   └── Sees additional options

4. User selects "Function-Level" granularity
   └── Radio button selected

5. User enables Jira integration
   └── Fills in Jira credentials, checks checkbox

6. User clicks "Start Analysis & Training"
   └── Form submits, progress dashboard appears

7. User watches progress (0-100%)
   └── Real-time updates every 2 seconds

8. Training completes (100%)
   └── Results section appears with metrics

9. User reviews results
   └── Sees best model, all model metrics, charts

10. User ready for new training
    └── Form re-enabled, can start again
```

---

#### **Example 2: Complete Prediction Workflow**

```
1. User navigates to "Predict Bugs" section
   └── Sees prediction form

2. User uploads code file
   └── "Selected: MyClass.java" appears

3. User uploads base file (optional)
   └── "Selected: MyClass_old.java" appears

4. User enters commit message (optional)
   └── "Fix performance issue in calculation"

5. User selects "All Models"
   └── Radio button selected

6. User checks "Function-Level Analysis"
   └── Checkbox checked

7. User clicks "Predict Performance Bugs"
   └── Form submits, loading message appears

8. Results appear (function-level)
   └── Shows 5 functions with predictions

9. User toggles to Table View
   └── Sees sortable table with all predictions

10. User filters by "Buggy Only"
    └── Table shows only BUGGY predictions

11. User clicks column header to sort
    └── Table sorted by probability (descending)

12. User ready for new prediction
    └── Can upload new file and predict again
```

---

### **Phase 9: UI Event Flow Diagram**

```
┌─────────────────────────────────────────┐
│         User Interaction                │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Event Listeners (app.js)            │
│  ┌──────────────┐  ┌──────────────┐     │
│  │ Training     │  │ Prediction  │     │
│  │ Form Submit  │  │ Form Submit  │     │
│  └──────┬───────┘  └──────┬───────┘     │
└─────────┼──────────────────┼────────────┘
           │                  │
           ▼                  ▼
┌──────────────────┐  ┌──────────────────┐
│  POST /api/train │  │  POST /predict   │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│  Background Job  │  │  Immediate       │
│  (Training)      │  │  Response         │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│  Polling         │  │  Render Results  │
│  (Every 2s)      │  │  (Card/Table)    │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│  Update UI       │  │  User Views      │
│  (Progress)      │  │  Predictions     │
└──────────────────┘  └──────────────────┘
```

---

### **Phase 10: UI Component Hierarchy**

```
index.html
├── Header
│   └── Title, Navigation
│
├── Hero Section
│   └── Welcome message, description
│
├── Train Models Section
│   ├── Training Form
│   │   ├── Repository URL Input
│   │   ├── Model Type Dropdown
│   │   ├── Advanced Configuration (Collapsible)
│   │   │   ├── Time/Commit Range
│   │   │   ├── Bug Keywords
│   │   │   ├── Granularity Radio Buttons
│   │   │   ├── Feature Selection Checkboxes
│   │   │   └── Jira Integration Fields
│   │   └── Submit Button
│   │
│   ├── Progress Dashboard (Hidden by default)
│   │   ├── Progress Bar
│   │   ├── Progress Text
│   │   ├── Progress Percentage
│   │   ├── Stop Button
│   │   └── Status Message
│   │
│   └── Training Results (Hidden by default)
│       ├── Metrics Grid
│       ├── Model Comparison Table
│       └── Visualizations
│
├── Predict Bugs Section
│   ├── Prediction Form
│   │   ├── Code File Input
│   │   ├── Base File Input
│   │   ├── Commit Message Textarea
│   │   ├── Model Selection (Radio Grid)
│   │   ├── Function-Level Checkbox
│   │   └── Submit Button
│   │
│   └── Results Container
│       ├── Results Header
│       │   ├── Title
│       │   └── View Toggle Buttons
│       │
│       ├── Card View
│       │   ├── Code Metrics Grid
│       │   └── Prediction Cards Grid
│       │
│       └── Table View
│           ├── Filter Controls
│           ├── Sort Controls
│           └── Results Table
│
└── Footer
    └── Copyright, links
```

---

## 📱 **Mobile Responsiveness**

### **Breakpoints**

```
Mobile: < 768px
├── Single column layout
├── Full-width inputs
├── Stacked cards
└── Touch-friendly buttons (min 44px height)

Tablet: 768px - 1024px
├── Two-column (if space allows)
├── Responsive grid
└── Adjusted padding

Desktop: > 1024px
├── Multi-column layout
├── Side-by-side forms
└── Full feature set
```

---

## 🎨 **UI/UX Best Practices Implemented**

1. **Clear Visual Hierarchy**
   - Important actions: Large, prominent buttons
   - Secondary actions: Smaller, less prominent
   - Information: Organized in cards/sections

2. **Feedback & Status**
   - Loading states for all async operations
   - Progress indicators for long-running tasks
   - Success/error messages clearly displayed

3. **Accessibility**
   - Semantic HTML
   - ARIA labels (where applicable)
   - Keyboard navigation support
   - Color contrast compliance

4. **Error Prevention**
   - Form validation before submission
   - Clear error messages
   - Disabled states prevent double-submission

5. **Performance**
   - Lazy loading of results
   - Efficient polling (2-second intervals)
   - Minimal DOM manipulation

---

**Last Updated**: 2024
**Version**: 1.0

