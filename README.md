# Hate Speech Detector

An intelligent machine learning system that detects and classifies hate speech in text using a hybrid approach combining ML and rule-based detection.

## Features

- Advanced hate speech detection using machine learning (Logistic Regression with probability calibration)
- Rule-based detection with comprehensive hate speech dictionary
- Real-time text analysis with detailed probability scores
- Balanced dataset from Twitter for training
- Pre-trained model with 2,860 balanced examples
- Clean and modular Python implementation

## Installation

### 1. Clone the Repository

```bash
git clone https://gitlab.com/pranavkumarnarayanan/HateSpeechDetector.git
cd HateSpeechDetector
```

### 2. Create Virtual Environment

#### Windows
```powershell
# Using Python's built-in venv
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate

# If you encounter execution policy issues, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### macOS
```bash
# Using Python's built-in venv
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

#### Linux
```bash
# Using Python's built-in venv
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip to the latest version
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### 4. Download NLTK Resources (if needed)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 5. Run the Hate Speech Detector

```bash
# Interactive mode
python hate_speech_detector.py

# Generate model analysis visualizations
python model_analysis.py
```

## Usage

Run the interactive detector:
```bash
python hate_speech_detector.py
```

The system will prompt you to enter text for analysis. For each input, it provides:
- Binary classification (hate speech or not)
- Confidence score
- Detailed breakdown of ML and rule-based scores

Example output:
```
Analysis Results:
Text: <your-input-text>
Is Hate Speech: True/False
Confidence: 85%

Detailed Scores:
- ML Model Score: 80%
- Rule-based Score: 90%
- Final Score: 85%
- Threshold: 35%
```

## Project Structure

- `hate_speech_detector.py`: Main detection system
- `online_dataset_collector.py`: Dataset collection and processing
- `model_analysis.py`: Model performance visualization
- `datasets/`: Contains processed datasets
- `visualizations/`: Contains model performance visualizations

## Model Performance Visualizations
After running `model_analysis.py`, check the `visualizations/` directory for:
- `confusion_matrix.png`: Model prediction accuracy
- `roc_curve.png`: Model discrimination ability
- `feature_importance.png`: Top influential features
- `class_distribution.png`: Class distribution comparison
- `classification_report.txt`: Detailed performance metrics

## Troubleshooting

### Common Issues
1. **Virtual Environment Not Activating**
   - Ensure you're using the correct activation command for your OS
   - Check Python installation and path settings

2. **Dependency Installation Failures**
   - Upgrade pip: `pip install --upgrade pip`
   - Install dependencies individually if batch install fails
   - Check system Python version matches requirements

3. **NLTK Resource Download**
   - Run the download commands in a Python interactive shell
   - Ensure you have internet connectivity

## Technical Details

- Model: Calibrated Logistic Regression
- Features: TF-IDF Vectorization (n-grams up to 5)
- Threshold: 0.35 (optimized for precision)
- Hybrid scoring:
  - 50% Machine Learning probability
  - 50% Rule-based scoring

## Dataset

The system uses a balanced dataset of 2,860 examples from Twitter:
- 1,430 hate speech examples
- 1,430 non-hate speech examples

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT Licensed - Feel free to use, modify, and share!

## Contact
Pranav Kumar Narayanan - pranavkumarsankar@gmail.com

Project Link: [https://gitlab.com/pranavkumarnarayanan/HateSpeechDetector](https://gitlab.com/pranavkumarnarayanan/HateSpeechDetector)

## 📞 Let's Connect!
- **Creator**: Pranav Kumar Narayanan
- **Email**: pranavkumarsankar@gmail.com
- **GitLab**: [@pranavkumarnarayanan](https://gitlab.com/pranavkumarnarayanan)

## 🌟 Fun Facts
- This project has analyzed thousands of comments
- It can detect subtle forms of hate speech
- It's constantly learning and improving
- It's my first major ML project!

---

*Built with ❤️ and a commitment to making the internet a better place*
