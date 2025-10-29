

import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path

print("🎵 Audio Lab - Complete Automated Processor")
print("="*70)

# ==================== SETTINGS ====================
# Selected classes
SELECTED_CLASSES = ['dog_bark', 'car_horn', 'children_playing', 'siren']

# Paths - YOUR CORRECT PATHS
URBANSOUND_PATH = r'C:\Users\Mhmou\Desktop\Round 3\D.collection\lab report 3\UrbanSound8K (1)\UrbanSound8K'
METADATA_FILE = r'C:\Users\Mhmou\Desktop\Round 3\D.collection\lab report 3\UrbanSound8K (1)\UrbanSound8K\metadata\UrbanSound8K.csv'

# Output folder
OUTPUT_FOLDER = 'my_audio_data'
FILES_PER_CLASS = 20

# ==================== STEP 1: ORGANIZE FILES ====================

def organize_audio_files():
    """Copy and organize audio files from UrbanSound8K"""
    print("\n" + "="*70)
    print("STEP 1: Organizing Audio Files")
    print("="*70)
    
    # Read metadata
    print(f"\n📖 Reading metadata from: {METADATA_FILE}")
    df = pd.read_csv(METADATA_FILE)
    
    # Create output folder
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    
    copied_files = []
    
    for class_name in SELECTED_CLASSES:
        print(f"\n🎯 Processing class: {class_name}")
        
        # Create class folder
        class_folder = os.path.join(OUTPUT_FOLDER, class_name)
        Path(class_folder).mkdir(exist_ok=True)
        
        # Get files for this class
        class_files = df[df['class'] == class_name].head(FILES_PER_CLASS)
        
        print(f"   Found {len(class_files)} files to copy")
        
        # Copy files
        for idx, row in class_files.iterrows():
            source_file = os.path.join(URBANSOUND_PATH, 'audio', f"fold{row['fold']}", row['slice_file_name'])
            dest_file = os.path.join(class_folder, row['slice_file_name'])
            
            if os.path.exists(source_file):
                shutil.copy2(source_file, dest_file)
                copied_files.append({
                    'filename': row['slice_file_name'],
                    'class': class_name,
                    'source': source_file,
                    'destination': dest_file
                })
                print(f"   ✓ Copied: {row['slice_file_name']}")
            else:
                print(f"   ✗ Not found: {source_file}")
    
    print(f"\n✅ Total files copied: {len(copied_files)}")
    return copied_files

# ==================== STEP 2: EXTRACT FEATURES ====================

def extract_features(audio_path, class_name):
    """Extract features from audio file"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Basic features
        duration = len(y) / sr
        rms = np.sqrt(np.mean(y**2))
        zcr_mean = np.mean(librosa.zero_crossings(y))
        
        # MFCC (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Compile features
        features = {
            'filename': os.path.basename(audio_path),
            'class': class_name,
            'duration': duration,
            'sample_rate': sr,
            'rms_energy': rms,
            'zero_crossing_rate': zcr_mean,
            'spectral_centroid_mean': np.mean(spec_cent),
            'spectral_centroid_std': np.std(spec_cent),
            'spectral_rolloff_mean': np.mean(spec_roll),
            'spectral_rolloff_std': np.std(spec_roll),
            'spectral_bandwidth_mean': np.mean(spec_bw),
            'spectral_bandwidth_std': np.std(spec_bw),
        }
        
        # Add MFCC
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = mfcc_mean[i]
            features[f'mfcc_{i+1}_std'] = mfcc_std[i]
        
        # Add Spectral Contrast
        for i in range(min(7, spec_contrast.shape[0])):
            features[f'spectral_contrast_{i+1}'] = np.mean(spec_contrast[i])
        
        # Add Chroma
        for i in range(12):
            features[f'chroma_{i+1}'] = np.mean(chroma[i])
        
        return features
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def create_spectrogram(audio_path, class_name, output_folder='spectrograms'):
    """Create and save spectrogram"""
    try:
        Path(output_folder).mkdir(exist_ok=True)
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Compute STFT
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram - {class_name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        
        # Save
        filename = os.path.basename(audio_path).replace('.wav', '')
        output_path = os.path.join(output_folder, f'spectrogram_{class_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    except Exception as e:
        print(f"   ❌ Error creating spectrogram: {e}")
        return None


def process_all_files():
    """Process all audio files"""
    print("\n" + "="*70)
    print("STEP 2: Extracting Features & Creating Spectrograms")
    print("="*70)
    
    all_features = []
    spectrograms_created = 0
    
    # Process each class
    for class_name in SELECTED_CLASSES:
        print(f"\n🎯 Processing class: {class_name}")
        print("-"*70)
        
        class_folder = os.path.join(OUTPUT_FOLDER, class_name)
        
        if not os.path.exists(class_folder):
            print(f"⚠️  Warning: Folder {class_folder} not found!")
            continue
        
        # Get WAV files
        audio_files = [f for f in os.listdir(class_folder) 
                      if f.endswith('.wav') or f.endswith('.WAV')]
        
        print(f"   Found {len(audio_files)} files")
        
        # Process each file
        for i, audio_file in enumerate(audio_files, 1):
            audio_path = os.path.join(class_folder, audio_file)
            
            print(f"   [{i:2d}/{len(audio_files)}] {audio_file[:50]:<50}", end=' ')
            
            # Extract features
            features = extract_features(audio_path, class_name)
            if features:
                all_features.append(features)
                print("✓")
            else:
                print("✗")
            
            # Create spectrogram for first file only
            if i == 1:
                print(f"   📊 Creating spectrogram...", end=' ')
                spec_path = create_spectrogram(audio_path, class_name)
                if spec_path:
                    spectrograms_created += 1
                    print(f"✓ Saved to {spec_path}")
                else:
                    print("✗")
    
    return all_features, spectrograms_created


def save_results(all_features):
    """Save features to CSV"""
    print("\n" + "="*70)
    print("STEP 3: Saving Results")
    print("="*70)
    
    if all_features:
        df = pd.DataFrame(all_features)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total rows: {len(df)}")
        print(f"   Total columns: {len(df.columns)}")
        
        # Save to CSV
        output_csv = 'audio_features.csv'
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Features saved to: {output_csv}")
        
        print(f"\n📊 Class Distribution:")
        print(df['class'].value_counts())
        
        print(f"\n📋 Sample Features (first 5 columns):")
        print(df.columns[:5].tolist())
        
        return df
    else:
        print("❌ No features extracted!")
        return None


# ==================== MAIN EXECUTION ====================

def main():
    print("\n🚀 Starting Automated Processing...")
    print("="*70)
    
    # Check paths
    print("\n🔍 Checking paths...")
    if not os.path.exists(URBANSOUND_PATH):
        print(f"\n❌ ERROR: UrbanSound8K path not found!")
        print(f"   Expected: {URBANSOUND_PATH}")
        return
    
    if not os.path.exists(METADATA_FILE):
        print(f"\n❌ ERROR: Metadata file not found!")
        print(f"   Expected: {METADATA_FILE}")
        return
    
    print("✅ Paths look good!")
    
    # Step 1: Organize files
    copied_files = organize_audio_files()
    
    if not copied_files:
        print("\n❌ No files were copied. Please check your paths!")
        return
    
    # Step 2: Process files
    all_features, spectrograms_created = process_all_files()
    
    # Step 3: Save results
    df = save_results(all_features)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ PROCESSING COMPLETE!")
    print("="*70)
    print("\n📦 What you got:")
    print(f"   ✓ {OUTPUT_FOLDER}/ folder with {len(copied_files)} organized audio files")
    print(f"   ✓ audio_features.csv with {len(all_features)} rows of extracted features")
    print(f"   ✓ spectrograms/ folder with {spectrograms_created} spectrogram images")
    
    print("\n📝 Next Steps:")
    print("   1. Open audio_features.csv and review the data")
    print("   2. Check spectrograms/ folder for the images")
    print("   3. Open the report template and start filling it")
    print("   4. Zip the my_audio_data/ folder for submission")
    
    print("\n🎉 Great job! You're almost done with the lab!")

if __name__ == "__main__":
    main()