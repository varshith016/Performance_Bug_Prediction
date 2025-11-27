# Performance Bug Detection System

A machine learning-powered system for detecting performance bugs in code commits using multiple ML models and the SZZ (Sliwerski-Zimmermann-Zeller) algorithm.

Analyze code commits and predict performance bugs using multiple machine learning models. Train custom models on your repositories or use pre-trained systems to identify potential issues before deployment.

## Features

- **Multiple ML Models**: 10+ machine learning algorithms (Random Forest, XGBoost, LightGBM, CatBoost, Deep Learning, etc.)
- **SZZ Algorithm**: Automatic bug-inducing commit detection
- **Function-Level Analysis**: Granular code analysis at function/method level
- **Jira Integration**: Enhanced bug detection with Jira issue tracker
- **Real-Time Training**: Background job processing with progress tracking
- **Interactive UI**: Modern web interface with card and table views

## Requirements

- Python 3.8+
- Git
- See `requirements.txt` for Python dependencies

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/varshith016/Performance_Bug_Prediction.git
   cd Performance_Bug_Prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or use the batch script (Windows):
   ```bash
   install_dependencies.bat
   ```

3. **Run the application:**
   ```bash
   cd webapp
   python app.py
   ```

4. **Access the web interface:**
   Open your browser and navigate to `http://localhost:5000`

## Usage

### Training Models

1. Open the web interface
2. Navigate to "Train Models" section
3. (Optional) Enter a Git repository URL for SZZ analysis
4. Select model type and configure advanced settings
5. Click "Start Analysis & Training"
6. Monitor progress in real-time

### Predicting Bugs

1. Navigate to "Predict Bugs" section
2. Upload a code file
3. (Optional) Upload base file and enter commit message
4. Select model(s) to use
5. (Optional) Enable function-level analysis
6. Click "Predict Performance Bugs"
7. View results in card or table view

## Configuration

### Jira Integration

To enable Jira integration for enhanced bug detection:

1. Go to "Train Models" → "Advanced Configuration"
2. Enter your Jira credentials:
   - Jira URL (e.g., `https://yourcompany.atlassian.net`)
   - Username/Email
   - API Token (get from https://id.atlassian.com/manage-profile/security/api-tokens)
3. Check "Enable Jira integration"
4. Start training

See `JIRA_INTEGRATION_GUIDE.md` for detailed instructions.

## Project Structure

```
.
├── webapp/                 # Main application
│   ├── app.py             # Flask application
│   ├── training_service.py # ML training service
│   ├── szz_service.py     # SZZ algorithm
│   ├── jira_service.py    # Jira integration
│   ├── feature_extractor.py # Feature extraction
│   ├── function_extractor.py # Function-level extraction
│   ├── git_service.py     # Git operations
│   ├── process_metrics.py # Process metrics
│   ├── templates/         # HTML templates
│   └── static/           # CSS, JavaScript
├── truly_clean_models/    # Trained models (excluded from git)
├── cloned_repos/          # Cloned repositories (excluded from git)
├── requirements.txt       # Python dependencies
├── Sampleeight.py         # Training script
└── README.md             # This file
```

## Documentation

- `SYSTEM_WORKFLOW_PIPELINE.md` - Complete system workflow and pipeline documentation
- `JIRA_INTEGRATION_GUIDE.md` - Jira integration guide

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


## Team
Varshith Bolloju
Sai Teja Rudroju
Firas 
Nivas Varma
Balaji Ramolla
Rishitha
Harika
---

**Note**: Trained models and cloned repositories are excluded from git due to size. Team members can train their own models or use the existing dataset.
