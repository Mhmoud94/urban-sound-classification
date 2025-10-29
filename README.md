# 🎵 Urban Sound Classification Project

An audio data collection and analysis project using the **UrbanSound8K** dataset. This project demonstrates the complete pipeline of audio processing, from data collection to feature extraction and visualization.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![librosa](https://img.shields.io/badge/librosa-0.10.0-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Author](#author)

## 🎯 Overview

This project performs a comprehensive audio analysis of urban environmental sounds. It includes:
- Automated audio file organization
- Feature extraction using librosa
- Spectrogram generation
- Data cleaning and CSV export

The project processes **80 audio files** across **4 sound classes**:
- 🐕 Dog Bark
- 🚗 Car Horn
- 👶 Children Playing
- 🚨 Siren

## 📊 Dataset

**UrbanSound8K Dataset**
- 8,732 labeled sound excerpts from urban environments
- Each clip is less than 4 seconds
- 10 different sound classes
- Organized into 10 folds for cross-validation

🔗 [Download Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)

## ✨ Features

### Audio Processing
- ✅ Automated file organization by class
- ✅ Quality assessment and validation
- ✅ Batch processing of multiple audio files

### Feature Extraction
The following features are extracted for each audio file:
- **Time-domain features:**
  - Duration
  - RMS Energy
  - Zero-Crossing Rate
  
- **Frequency-domain features:**
  - 13 MFCCs (Mel-Frequency Cepstral Coefficients)
  - Spectral Centroid
  - Spectral Rolloff
  - Spectral Bandwidth
  - Spectral Contrast (7 bands)
  - Chroma Features (12 pitch classes)

**Total: 57 features per audio file**

### Visualization
- 📈 Spectrogram generation for each class
- 🎨 High-quality plots (300 DPI)
- 📊 Frequency vs. Time representation

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies
```bash
pip install librosa pandas numpy matplotlib
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Mhmoud94/urban-sound-classification.git
cd urban-sound-classification
```

### 2. Download the Dataset

Download the UrbanSound8K dataset and extract it to your preferred location.

### 3. Update File Paths

Edit `complete_audio_processor.py` and update these lines:
```python
URBANSOUND_PATH = r'C:\path\to\your\UrbanSound8K'
METADATA_FILE = r'C:\path\to\your\UrbanSound8K\metadata\UrbanSound8K.csv'
```

### 4. Run the Script
```bash
python complete_audio_processor.py
```

### 5. Output

The script will create:
- `my_audio_data/` - Organized audio files (80 files in 4 folders)
- `audio_features.csv` - Extracted features (80 rows × 57 columns)
- `spectrograms/` - Spectrogram images (4 PNG files)

## 📈 Results

### Dataset Statistics

| Class | Files | Avg Duration | Avg RMS Energy | Avg Spectral Centroid |
|-------|-------|--------------|----------------|----------------------|
| Dog Bark | 20 | 3.26s | 0.0748 | 2741 Hz |
| Car Horn | 20 | 1.31s | 0.0845 | 2211 Hz |
| Children Playing | 20 | 4.00s | 0.0281 | 2999 Hz |
| Siren | 20 | 4.00s | 0.0374 | 2241 Hz |

### Spectrograms

The spectrograms reveal distinct acoustic signatures for each class:

- **Dog Bark:** Short, impulsive events with energy concentrated in 0-5000 Hz
- **Car Horn:** Sustained tonal sounds with clear harmonic structure
- **Children Playing:** Complex, broadband spectral content with high variability
- **Siren:** Concentrated energy in lower frequencies with periodic patterns

## 📁 Project Structure
```
urban-sound-classification/
│
├── complete_audio_processor.py    # Main processing script
├── audio_features.csv             # Extracted features (output)
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
└── spectrograms/                  # Generated spectrograms
    ├── spectrogram_dog_bark.png
    ├── spectrogram_car_horn.png
    ├── spectrogram_children_playing.png
    └── spectrogram_siren.png
```

## 🔧 Technologies Used

- **Python 3.12** - Programming language
- **librosa** - Audio analysis and feature extraction
- **pandas** - Data manipulation and CSV handling
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization and spectrogram generation

## 📚 Key Learnings

Through this project, I gained hands-on experience with:
- Audio signal processing and analysis
- Feature engineering for machine learning
- Batch processing and automation
- Data organization and documentation
- Scientific Python libraries (librosa, pandas, numpy)

## 🎓 Academic Context

This project was completed as part of the **Data Collection and Data Quality** course in the Master's program in Data Science at **Dalarna University**. The lab covered:
- Audio data collection methodologies
- Feature extraction techniques
- Data quality assessment
- Ethical considerations in audio recording
- Protocol development for research

## 🔮 Future Improvements

Potential enhancements for this project:
- [ ] Implement machine learning classification models
- [ ] Add deep learning approaches (CNN, RNN)
- [ ] Expand to all 10 UrbanSound8K classes
- [ ] Create interactive web dashboard
- [ ] Add real-time audio classification
- [ ] Implement data augmentation techniques

## 📄 License

This project is for educational purposes. The UrbanSound8K dataset has its own license - please refer to the [official dataset page](https://urbansounddataset.weebly.com/urbansound8k.html) for usage terms.

## 👤 Author

**Mhmoud Ahmad**

Master's Student in Data Science  
Dalarna University, Sweden

- GitHub: [@Mhmoud94](https://github.com/Mhmoud94)
- LinkedIn: [Mhmoud Ahmad](https://linkedin.com/in/mhmoudahmad)
- Email: mhmoud.94ahmad@gmail.com

## 🙏 Acknowledgments

- J. Salamon, C. Jacoby, and J. P. Bello - UrbanSound8K Dataset creators
- librosa development team
- Dalarna University - Data Collection and Data Quality course instructors
- Master's program in the Data Science faculty

## 📞 Contact

For questions or feedback about this project, feel free to reach out:
- 📧 Email: mhmoud.94ahmad@gmail.com
- 💼 LinkedIn: [Connect with me](https://linkedin.com/in/mhmoudahmad)
- 🐙 GitHub: [View my repositories](https://github.com/Mhmoud94)

---

⭐ **If you found this project helpful, please give it a star!** ⭐

---

*Project completed as part of Master's coursework at Dalarna University*  
*Last Updated: October 2025*
