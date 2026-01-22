"""
Script untuk training model CNN klasifikasi ikan air tawar
Ikan yang didukung: Lele, Patin, Nila, Gurame

Cara menggunakan:
1. Siapkan dataset di folder dataset/train/ dan dataset/validation/
2. Jalankan: python train_model.py
3. Model akan disimpan di backend/models/fish_classifier.h5
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# TensorFlow dan Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Konfigurasi
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Will be adjusted based on dataset size
EPOCHS = 50
NUM_CLASSES = 4  # Lele, Patin, Nila, Gurame

# Paths
ROOT_DIR = Path(__file__).parent
DATASET_DIR = ROOT_DIR / 'dataset'
TRAIN_DIR = DATASET_DIR / 'train'
VAL_DIR = DATASET_DIR / 'validation'
MODEL_DIR = ROOT_DIR / 'backend' / 'models'
MODEL_PATH = MODEL_DIR / 'fish_classifier.h5'

# Create models directory if not exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def check_dataset():
    """Cek apakah dataset tersedia"""
    print("\n" + "="*60)
    print("CHECKING DATASET")
    print("="*60)

    if not TRAIN_DIR.exists():
        print(f"❌ ERROR: Training directory tidak ditemukan: {TRAIN_DIR}")
        print("\nSilakan buat folder dataset dengan struktur:")
        print("dataset/")
        print("  ├── train/")
        print("  │   ├── Lele/")
        print("  │   ├── Patin/")
        print("  │   ├── Nila/")
        print("  │   └── Gurame/")
        print("  └── validation/")
        print("      ├── Lele/")
        print("      ├── Patin/")
        print("      ├── Nila/")
        print("      └── Gurame/")
        return False

    classes = ['Lele', 'Patin', 'Nila', 'Gurame']

    print("\nDataset Training:")
    train_counts = {}
    for cls in classes:
        class_dir = TRAIN_DIR / cls
        if class_dir.exists():
            count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png'))) + len(list(class_dir.glob('*.jpeg')))
            train_counts[cls] = count
            print(f"  {cls}: {count} gambar")
        else:
            print(f"  {cls}: ❌ Folder tidak ditemukan")
            train_counts[cls] = 0

    print("\nDataset Validation:")
    val_counts = {}
    for cls in classes:
        class_dir = VAL_DIR / cls
        if class_dir.exists():
            count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png'))) + len(list(class_dir.glob('*.jpeg')))
            val_counts[cls] = count
            print(f"  {cls}: {count} gambar")
        else:
            print(f"  {cls}: ❌ Folder tidak ditemukan")
            val_counts[cls] = 0

    # Check if we have enough data
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())

    print(f"\nTotal Training: {total_train}")
    print(f"Total Validation: {total_val}")

    if total_train == 0:
        print("\n❌ ERROR: Tidak ada data training!")
        print("Silakan tambahkan gambar ikan ke folder dataset/train/")
        return False

    # Warning for minimal dataset
    if total_train < 40:  # Less than 10 images per class
        print("\n❌ ERROR: Data terlalu sedikit!")
        print(f"Minimal 10 gambar per kelas (40 total), saat ini: {total_train}")
        return False
    elif total_train < 200:  # Less than 50 images per class
        print("\n⚠️  WARNING: Dataset Minimal!")
        print(f"Total gambar: {total_train}")
        print("Dengan dataset minimal (10-50 per kelas):")
        print("  - Akurasi yang diharapkan: 60-75%")
        print("  - Model mungkin overfitting")
        print("  - Akan menggunakan data augmentation agresif")
        print("  - Disarankan tambah data bertahap untuk hasil lebih baik")
        print("\nUntuk hasil optimal (85-95% akurasi):")
        print("  - Minimal: 100-200 gambar per kelas")
        print("  - Ideal: 500-1000 gambar per kelas")

    if total_val == 0:
        print("\n⚠️  INFO: Tidak ada data validation terpisah")
        print("Training akan menggunakan 20% dari training data untuk validation")

    return True

def create_data_generators():
    """Buat data generators untuk training dan validation"""
    print("\n" + "="*60)
    print("CREATING DATA GENERATORS")
    print("="*60)

    # Count total images to determine augmentation level
    total_images = 0
    for cls in ['Lele', 'Patin', 'Nila', 'Gurame']:
        class_dir = TRAIN_DIR / cls
        if class_dir.exists():
            total_images += len(list(class_dir.glob('*.jpg'))) + \
                           len(list(class_dir.glob('*.png'))) + \
                           len(list(class_dir.glob('*.jpeg')))

    # Use aggressive augmentation for small datasets
    if total_images < 200:  # Minimal dataset (10-50 per class)
        print("📊 Dataset Mode: MINIMAL (using aggressive augmentation)")
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,           # Increased from 20
            width_shift_range=0.3,       # Increased from 0.2
            height_shift_range=0.3,      # Increased from 0.2
            shear_range=0.3,             # Increased from 0.2
            zoom_range=0.3,              # Increased from 0.2
            horizontal_flip=True,
            vertical_flip=True,          # Added
            brightness_range=[0.7, 1.3], # Added
            fill_mode='nearest',
            validation_split=0.2
        )
    else:  # Normal dataset
        print("📊 Dataset Mode: NORMAL (standard augmentation)")
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )

    # Hanya rescaling untuk validation
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Adjust batch size for small datasets
    batch_size = BATCH_SIZE
    if total_images < 200:
        batch_size = min(8, total_images // 8)  # Smaller batch for small datasets
        print(f"Adjusted batch size: {batch_size} (optimal for small dataset)")

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training' if not VAL_DIR.exists() else None
    )

    # Validation generator
    if VAL_DIR.exists() and len(list(VAL_DIR.glob('*/*.jpg'))) > 0:
        print("Using separate validation directory")
        validation_generator = val_datagen.flow_from_directory(
            VAL_DIR,
            target_size=IMG_SIZE,
            batch_size=batch_size,
            class_mode='categorical'
        )
    else:
        print("Using validation split from training data (20%)")
        validation_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=IMG_SIZE,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

    print(f"\nClasses: {train_generator.class_indices}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")

    return train_generator, validation_generator

def build_model(use_transfer_learning=True):
    """
    Buat model CNN

    Args:
        use_transfer_learning: Jika True, gunakan MobileNetV2 pre-trained
                              Jika False, buat CNN dari scratch
    """
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)

    if use_transfer_learning:
        print("Using Transfer Learning with MobileNetV2")

        # Load MobileNetV2 pre-trained on ImageNet
        base_model = MobileNetV2(
            input_shape=IMG_SIZE + (3,),
            include_top=False,
            weights='imagenet'
        )

        # Freeze base model layers
        base_model.trainable = False

        # Add custom classification head with more dropout for small datasets
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),  # Increased from 0.3 for regularization
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),  # Increased from 0.3
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])

    else:
        print("Building CNN from scratch")

        model = keras.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SIZE + (3,)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),

            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),

            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),

            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),

            # Classification head
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Summary:")
    model.summary()

    return model

def train(model, train_generator, validation_generator):
    """Training model"""
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    # Callbacks
    callbacks = [
        # Save best model
        ModelCheckpoint(
            str(MODEL_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),

        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    return history

def plot_training_history(history):
    """Plot training history"""
    print("\n" + "="*60)
    print("PLOTTING TRAINING HISTORY")
    print("="*60)

    # Create plots directory
    plots_dir = ROOT_DIR / 'training_plots'
    plots_dir.mkdir(exist_ok=True)

    # Plot accuracy
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.grid(True)

    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = plots_dir / f'training_history_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nTraining history plot saved to: {plot_path}")

    # Print final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"\nModel saved to: {MODEL_PATH}")

def evaluate_model(model, val_dir, batch_size=32):
    """
    Evaluasi model dengan Confusion Matrix dan Classification Report
    """
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)

    if not val_dir.exists():
        print("Validation directory not found, skipping detailed evaluation.")
        return

    # Create evaluation generator (shuffle=False is CRITICAL for confusion matrix)
    eval_datagen = ImageDataGenerator(rescale=1./255)
    
    eval_generator = eval_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Important!
    )

    print("\nGenerating predictions...")
    # Get predictions
    predictions = model.predict(eval_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = eval_generator.classes
    class_labels = list(eval_generator.class_indices.keys())

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save Confusion Matrix plot
    plots_dir = ROOT_DIR / 'training_plots'
    plots_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cm_plot_path = plots_dir / f'confusion_matrix_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(cm_plot_path)
    print(f"\nConfusion Matrix plot saved to: {cm_plot_path}")

    # --- NEW: Metrics Bar Chart ---
    # Get metrics dict
    report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    
    # Extract overall metrics
    accuracy = report_dict['accuracy']
    precision = report_dict['weighted avg']['precision']
    recall = report_dict['weighted avg']['recall']
    f1 = report_dict['weighted avg']['f1-score']
    
    metrics = [accuracy, precision, recall, f1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['blue', 'green', 'orange', 'purple']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metric_names, metrics, color=colors)
    
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)  # Scale from 0 to 1
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    
    # Save Metrics plot
    metrics_plot_path = plots_dir / f'performance_metrics_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(metrics_plot_path)
    print(f"Performance Metrics bar chart saved to: {metrics_plot_path}")

def main():
    """Main training function"""
    print("\n" + "="*60)
    print("FISH CLASSIFIER TRAINING SCRIPT")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    # Check dataset
    if not check_dataset():
        print("\n❌ Dataset check failed. Exiting...")
        return

    # Ask user if they want to continue
    print("\n" + "="*60)
    response = input("Lanjutkan training? (y/n): ")
    if response.lower() != 'y':
        print("Training dibatalkan.")
        return

    # Ask about transfer learning
    print("\n" + "="*60)
    print("PILIH METODE TRAINING:")
    print("1. Transfer Learning dengan MobileNetV2 (Recommended - lebih cepat & akurat)")
    print("2. CNN from Scratch (Membutuhkan lebih banyak data)")
    choice = input("Pilih (1/2): ")
    use_transfer_learning = (choice == '1')

    # Create data generators
    train_gen, val_gen = create_data_generators()

    # Build model
    model = build_model(use_transfer_learning=use_transfer_learning)

    # Train model
    history = train(model, train_gen, val_gen)

    # Plot results
    plot_training_history(history)

    # Evaluate model
    if VAL_DIR.exists():
         # Use the model to evaluate on validation set
         # We recreate generator to ensure shuffle=False
         evaluate_model(model, VAL_DIR, BATCH_SIZE)
    else:
        print("\nNote: Skipping detailed evaluation (Confusion Matrix) because no separate validation folder found.")
        print("To get detailed metrics, please organize data into train/ and validation/ folders.")

    print("\n" + "="*60)
    print("✓ TRAINING SELESAI!")
    print("="*60)
    print(f"\nModel telah disimpan di: {MODEL_PATH}")
    print("\nUntuk menggunakan model:")
    print("1. Restart backend server")
    print("2. Backend akan otomatis load model yang baru")
    print("3. Upload gambar ikan untuk klasifikasi")
    print("\n" + "="*60)

if __name__ == '__main__':
    main()
