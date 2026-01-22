"""
Script untuk Augmentasi Data Gambar Ikan
=========================================

Script ini akan membuat variasi gambar dari dataset yang ada menggunakan
berbagai teknik augmentasi untuk meningkatkan jumlah data training.

Fitur Augmentasi:
- Rotasi (berbagai sudut)
- Flip horizontal dan vertikal
- Zoom in/out
- Brightness, Contrast, Saturation adjustment
- Shear transformation
- Shift (horizontal dan vertikal)
- Gaussian blur dan Motion blur
- Color Space Transformations (HSV, RGB channel shifts)
- Noise Injection (Gaussian, Salt & Pepper, Speckle)
- Random Erasing
- Kombinasi dari teknik di atas

Cara menggunakan:
1. Pastikan dataset ada di folder dataset/train/
2. Jalankan: python augment_data.py
3. Gambar hasil augmentasi akan disimpan di folder yang sama

Author: Fish Classification System
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import argparse
from datetime import datetime
from tqdm import tqdm
import random
import cv2

# Konfigurasi
CLASSES = ['Lele', 'Patin', 'Nila', 'Gurame']
ROOT_DIR = Path(__file__).parent
DATASET_DIR = ROOT_DIR / 'dataset'
TRAIN_DIR = DATASET_DIR / 'train'


class ImageAugmentor:
    """Class untuk melakukan augmentasi gambar"""

    def __init__(self, seed=42):
        """
        Initialize augmentor

        Args:
            seed: Random seed untuk reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)

    def rotate(self, image, angle=None):
        """
        Rotasi gambar

        Args:
            image: PIL Image object
            angle: Sudut rotasi (jika None, akan random antara -30 sampai 30)
        """
        if angle is None:
            angle = random.uniform(-30, 30)
        return image.rotate(angle, fillcolor=(0, 0, 0), expand=False)

    def flip_horizontal(self, image):
        """Flip gambar secara horizontal"""
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    def flip_vertical(self, image):
        """Flip gambar secara vertikal"""
        return image.transpose(Image.FLIP_TOP_BOTTOM)

    def adjust_brightness(self, image, factor=None):
        """
        Adjust brightness gambar

        Args:
            image: PIL Image object
            factor: Brightness factor (jika None, akan random antara 0.7 sampai 1.3)
                   1.0 = original, < 1.0 = darker, > 1.0 = brighter
        """
        if factor is None:
            factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def adjust_contrast(self, image, factor=None):
        """
        Adjust contrast gambar

        Args:
            image: PIL Image object
            factor: Contrast factor (jika None, akan random antara 0.8 sampai 1.2)
        """
        if factor is None:
            factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def adjust_saturation(self, image, factor=None):
        """
        Adjust saturation gambar

        Args:
            image: PIL Image object
            factor: Saturation factor (jika None, akan random antara 0.8 sampai 1.2)
        """
        if factor is None:
            factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    def zoom(self, image, zoom_factor=None):
        """
        Zoom in/out gambar

        Args:
            image: PIL Image object
            zoom_factor: Zoom factor (jika None, akan random antara 0.8 sampai 1.2)
                        > 1.0 = zoom in, < 1.0 = zoom out
        """
        if zoom_factor is None:
            zoom_factor = random.uniform(0.8, 1.2)

        width, height = image.size
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)

        # Resize image
        resized = image.resize((new_width, new_height), Image.LANCZOS)

        if zoom_factor > 1.0:
            # Zoom in - crop center
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            return resized.crop((left, top, left + width, top + height))
        else:
            # Zoom out - paste on black background
            new_image = Image.new('RGB', (width, height), (0, 0, 0))
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            new_image.paste(resized, (left, top))
            return new_image

    def shift(self, image, horizontal=None, vertical=None):
        """
        Shift gambar horizontal/vertical

        Args:
            image: PIL Image object
            horizontal: Horizontal shift dalam pixels (jika None, akan random)
            vertical: Vertical shift dalam pixels (jika None, akan random)
        """
        width, height = image.size

        if horizontal is None:
            horizontal = random.randint(-int(width * 0.2), int(width * 0.2))
        if vertical is None:
            vertical = random.randint(-int(height * 0.2), int(height * 0.2))

        return image.transform(
            image.size,
            Image.AFFINE,
            (1, 0, -horizontal, 0, 1, -vertical),
            fillcolor=(0, 0, 0)
        )

    def blur(self, image, radius=None):
        """
        Apply gaussian blur

        Args:
            image: PIL Image object
            radius: Blur radius (jika None, akan random antara 0 sampai 2)
        """
        if radius is None:
            radius = random.uniform(0, 2)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def shear(self, image, shear_factor=None):
        """
        Apply shear transformation

        Args:
            image: PIL Image object
            shear_factor: Shear factor (jika None, akan random antara -0.2 sampai 0.2)
        """
        if shear_factor is None:
            shear_factor = random.uniform(-0.2, 0.2)

        width, height = image.size

        # Calculate shear matrix
        xshift = abs(shear_factor) * width
        new_width = width + int(round(xshift))

        return image.transform(
            (new_width, height),
            Image.AFFINE,
            (1, shear_factor, -xshift if shear_factor > 0 else 0, 0, 1, 0),
            Image.BICUBIC,
            fillcolor=(0, 0, 0)
        ).resize((width, height), Image.LANCZOS)

    def add_gaussian_noise(self, image, mean=0, sigma=None):
        """
        Tambahkan Gaussian noise ke gambar

        Args:
            image: PIL Image object
            mean: Mean of noise distribution
            sigma: Standard deviation (jika None, akan random antara 5 sampai 25)
        """
        if sigma is None:
            sigma = random.uniform(5, 25)

        # Convert to numpy array
        img_array = np.array(image)

        # Generate Gaussian noise
        gaussian_noise = np.random.normal(mean, sigma, img_array.shape)

        # Add noise to image
        noisy_image = img_array + gaussian_noise

        # Clip values to valid range [0, 255]
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_image)

    def add_salt_pepper_noise(self, image, salt_prob=None, pepper_prob=None):
        """
        Tambahkan Salt & Pepper noise ke gambar

        Args:
            image: PIL Image object
            salt_prob: Probability of salt noise (jika None, akan random)
            pepper_prob: Probability of pepper noise (jika None, akan random)
        """
        if salt_prob is None:
            salt_prob = random.uniform(0.001, 0.01)
        if pepper_prob is None:
            pepper_prob = random.uniform(0.001, 0.01)

        # Convert to numpy array
        img_array = np.array(image).copy()

        # Generate random matrix
        rnd = np.random.random(img_array.shape[:2])

        # Add salt noise (white pixels)
        img_array[rnd < salt_prob] = 255

        # Add pepper noise (black pixels)
        img_array[rnd > 1 - pepper_prob] = 0

        return Image.fromarray(img_array)

    def add_speckle_noise(self, image, variance=None):
        """
        Tambahkan Speckle noise ke gambar

        Args:
            image: PIL Image object
            variance: Variance of noise (jika None, akan random antara 0.01 sampai 0.1)
        """
        if variance is None:
            variance = random.uniform(0.01, 0.1)

        # Convert to numpy array
        img_array = np.array(image) / 255.0

        # Generate speckle noise
        noise = np.random.randn(*img_array.shape) * variance

        # Add noise
        noisy_image = img_array + img_array * noise

        # Clip and convert back
        noisy_image = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_image)

    def adjust_hsv(self, image, hue_shift=None, sat_shift=None, val_shift=None):
        """
        Adjust HSV color space

        Args:
            image: PIL Image object
            hue_shift: Hue shift (-180 to 180, jika None akan random)
            sat_shift: Saturation multiplier (jika None akan random 0.7-1.3)
            val_shift: Value multiplier (jika None akan random 0.7-1.3)
        """
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert to HSV
        img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Apply hue shift
        if hue_shift is None:
            hue_shift = random.uniform(-30, 30)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_shift) % 180

        # Apply saturation shift
        if sat_shift is None:
            sat_shift = random.uniform(0.7, 1.3)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * sat_shift, 0, 255)

        # Apply value shift
        if val_shift is None:
            val_shift = random.uniform(0.7, 1.3)
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * val_shift, 0, 255)

        # Convert back to RGB
        img_hsv = img_hsv.astype(np.uint8)
        img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        return Image.fromarray(img_rgb)

    def rgb_channel_shift(self, image, r_shift=None, g_shift=None, b_shift=None):
        """
        Shift individual RGB channels

        Args:
            image: PIL Image object
            r_shift, g_shift, b_shift: Shift values (-50 to 50, jika None akan random)
        """
        img_array = np.array(image).astype(np.float32)

        if r_shift is None:
            r_shift = random.uniform(-30, 30)
        if g_shift is None:
            g_shift = random.uniform(-30, 30)
        if b_shift is None:
            b_shift = random.uniform(-30, 30)

        img_array[:, :, 0] = np.clip(img_array[:, :, 0] + r_shift, 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] + g_shift, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] + b_shift, 0, 255)

        return Image.fromarray(img_array.astype(np.uint8))

    def motion_blur(self, image, kernel_size=None):
        """
        Apply motion blur effect

        Args:
            image: PIL Image object
            kernel_size: Size of blur kernel (jika None, akan random antara 5 sampai 15)
        """
        if kernel_size is None:
            kernel_size = random.choice([5, 7, 9, 11, 13, 15])

        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        # Apply motion blur
        blurred = cv2.filter2D(img_cv, -1, kernel)

        # Convert back to RGB
        img_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

        return Image.fromarray(img_rgb)

    def random_erasing(self, image, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        """
        Random Erasing augmentation (cutout)

        Args:
            image: PIL Image object
            probability: Probability of applying erasing
            sl: Min erasing area
            sh: Max erasing area
            r1: Aspect ratio
        """
        if random.random() > probability:
            return image

        img_array = np.array(image).copy()
        height, width, channels = img_array.shape
        area = height * width

        for _ in range(100):
            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1 / r1)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                x1 = random.randint(0, width - w)
                y1 = random.randint(0, height - h)

                # Fill with random color or mean color
                if random.random() > 0.5:
                    # Random gray value
                    img_array[y1:y1 + h, x1:x1 + w, :] = random.randint(0, 255)
                else:
                    # Mean color of image
                    mean_color = img_array.mean(axis=(0, 1))
                    img_array[y1:y1 + h, x1:x1 + w, :] = mean_color

                return Image.fromarray(img_array)

        return image

    def color_jitter(self, image):
        """
        Apply random color jittering (combination of brightness, contrast, saturation)
        """
        # Random brightness
        brightness_factor = random.uniform(0.7, 1.3)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)

        # Random contrast
        contrast_factor = random.uniform(0.7, 1.3)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        # Random saturation
        saturation_factor = random.uniform(0.7, 1.3)
        image = ImageEnhance.Color(image).enhance(saturation_factor)

        return image


class DataAugmentor:
    """Class untuk mengatur proses augmentasi dataset"""

    def __init__(self, augmentations_per_image=5, output_dir=None, classes=None):
        """
        Initialize data augmentor

        Args:
            augmentations_per_image: Jumlah augmentasi yang akan dibuat per gambar
            output_dir: Directory output (jika None, akan menyimpan di folder yang sama)
            classes: List of classes to augment (default: all classes)
        """
        self.augmentations_per_image = augmentations_per_image
        self.output_dir = Path(output_dir) if output_dir else None
        self.classes = classes if classes else CLASSES
        self.augmentor = ImageAugmentor()
        self.stats = {
            'total_original': 0,
            'total_augmented': 0,
            'per_class': {}
        }

    def get_augmentation_pipelines(self):
        """
        Definisi berbagai pipeline augmentasi

        Returns:
            List of augmentation pipelines (list of functions)
        """
        pipelines = [
            # Pipeline 1: Simple rotation + flip
            [
                lambda img: self.augmentor.rotate(img),
                lambda img: self.augmentor.flip_horizontal(img)
            ],

            # Pipeline 2: Brightness + contrast
            [
                lambda img: self.augmentor.adjust_brightness(img),
                lambda img: self.augmentor.adjust_contrast(img)
            ],

            # Pipeline 3: Zoom + brightness
            [
                lambda img: self.augmentor.zoom(img),
                lambda img: self.augmentor.adjust_brightness(img)
            ],

            # Pipeline 4: Rotation + brightness + slight blur
            [
                lambda img: self.augmentor.rotate(img, random.uniform(-20, 20)),
                lambda img: self.augmentor.adjust_brightness(img),
                lambda img: self.augmentor.blur(img, random.uniform(0, 1))
            ],

            # Pipeline 5: Shift + flip
            [
                lambda img: self.augmentor.shift(img),
                lambda img: self.augmentor.flip_vertical(img)
            ],

            # Pipeline 6: Shear + brightness
            [
                lambda img: self.augmentor.shear(img),
                lambda img: self.augmentor.adjust_brightness(img)
            ],

            # Pipeline 7: Multiple adjustments
            [
                lambda img: self.augmentor.rotate(img, random.uniform(-15, 15)),
                lambda img: self.augmentor.adjust_saturation(img),
                lambda img: self.augmentor.adjust_contrast(img)
            ],

            # Pipeline 8: Zoom out + shift + brightness
            [
                lambda img: self.augmentor.zoom(img, random.uniform(0.8, 0.95)),
                lambda img: self.augmentor.shift(img),
                lambda img: self.augmentor.adjust_brightness(img, random.uniform(0.8, 1.2))
            ],

            # Pipeline 9: Heavy rotation + adjustments
            [
                lambda img: self.augmentor.rotate(img, random.uniform(-30, 30)),
                lambda img: self.augmentor.adjust_brightness(img),
                lambda img: self.augmentor.adjust_contrast(img, random.uniform(0.9, 1.1))
            ],

            # Pipeline 10: Flip both + brightness
            [
                lambda img: self.augmentor.flip_horizontal(img),
                lambda img: self.augmentor.flip_vertical(img),
                lambda img: self.augmentor.adjust_brightness(img)
            ],

            # ===== NEW PIPELINES WITH NOISE & COLOR SPACE =====

            # Pipeline 11: Gaussian noise + slight rotation
            [
                lambda img: self.augmentor.add_gaussian_noise(img, sigma=random.uniform(10, 20)),
                lambda img: self.augmentor.rotate(img, random.uniform(-10, 10))
            ],

            # Pipeline 12: HSV adjustment + flip
            [
                lambda img: self.augmentor.adjust_hsv(img),
                lambda img: self.augmentor.flip_horizontal(img)
            ],

            # Pipeline 13: RGB channel shift + brightness
            [
                lambda img: self.augmentor.rgb_channel_shift(img),
                lambda img: self.augmentor.adjust_brightness(img)
            ],

            # Pipeline 14: Color jitter + zoom
            [
                lambda img: self.augmentor.color_jitter(img),
                lambda img: self.augmentor.zoom(img, random.uniform(0.85, 1.15))
            ],

            # Pipeline 15: Motion blur + rotation
            [
                lambda img: self.augmentor.motion_blur(img),
                lambda img: self.augmentor.rotate(img, random.uniform(-15, 15))
            ],

            # Pipeline 16: Salt & pepper noise + brightness
            [
                lambda img: self.augmentor.add_salt_pepper_noise(img),
                lambda img: self.augmentor.adjust_brightness(img)
            ],

            # Pipeline 17: Speckle noise + contrast
            [
                lambda img: self.augmentor.add_speckle_noise(img, variance=random.uniform(0.02, 0.05)),
                lambda img: self.augmentor.adjust_contrast(img)
            ],

            # Pipeline 18: Random erasing + flip
            [
                lambda img: self.augmentor.random_erasing(img, probability=0.5),
                lambda img: self.augmentor.flip_horizontal(img)
            ],

            # Pipeline 19: HSV + Gaussian noise + rotation (complex)
            [
                lambda img: self.augmentor.adjust_hsv(img, hue_shift=random.uniform(-20, 20)),
                lambda img: self.augmentor.add_gaussian_noise(img, sigma=random.uniform(5, 15)),
                lambda img: self.augmentor.rotate(img, random.uniform(-20, 20))
            ],

            # Pipeline 20: Color jitter + noise + blur (very complex)
            [
                lambda img: self.augmentor.color_jitter(img),
                lambda img: self.augmentor.add_speckle_noise(img, variance=0.03),
                lambda img: self.augmentor.blur(img, random.uniform(0.5, 1.5))
            ],

            # Pipeline 21: RGB shift + motion blur + shift
            [
                lambda img: self.augmentor.rgb_channel_shift(img),
                lambda img: self.augmentor.motion_blur(img, kernel_size=random.choice([5, 7, 9])),
                lambda img: self.augmentor.shift(img)
            ],

            # Pipeline 22: HSV + zoom + random erasing
            [
                lambda img: self.augmentor.adjust_hsv(img),
                lambda img: self.augmentor.zoom(img, random.uniform(0.9, 1.1)),
                lambda img: self.augmentor.random_erasing(img, probability=0.3)
            ],

            # Pipeline 23: Rotation + HSV + Gaussian noise
            [
                lambda img: self.augmentor.rotate(img, random.uniform(-25, 25)),
                lambda img: self.augmentor.adjust_hsv(img, sat_shift=random.uniform(0.8, 1.2)),
                lambda img: self.augmentor.add_gaussian_noise(img, sigma=10)
            ],

            # Pipeline 24: Shear + color jitter + flip
            [
                lambda img: self.augmentor.shear(img),
                lambda img: self.augmentor.color_jitter(img),
                lambda img: self.augmentor.flip_vertical(img)
            ],

            # Pipeline 25: All-in-one extreme augmentation
            [
                lambda img: self.augmentor.rotate(img, random.uniform(-20, 20)),
                lambda img: self.augmentor.adjust_hsv(img),
                lambda img: self.augmentor.add_gaussian_noise(img, sigma=random.uniform(8, 15)),
                lambda img: self.augmentor.zoom(img, random.uniform(0.85, 1.1)),
                lambda img: self.augmentor.adjust_brightness(img)
            ]
        ]

        return pipelines

    def augment_image(self, image_path, output_dir, num_augmentations):
        """
        Augmentasi satu gambar

        Args:
            image_path: Path ke gambar original
            output_dir: Directory untuk menyimpan hasil augmentasi
            num_augmentations: Jumlah augmentasi yang akan dibuat

        Returns:
            Number of augmented images created
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')

            # Get filename without extension
            filename = image_path.stem
            extension = image_path.suffix

            # Get augmentation pipelines
            pipelines = self.get_augmentation_pipelines()

            # Create augmentations
            created = 0
            for i in range(num_augmentations):
                # Pilih pipeline secara random
                pipeline = random.choice(pipelines)

                # Apply pipeline
                augmented = image.copy()
                for transform in pipeline:
                    augmented = transform(augmented)

                # Save augmented image
                output_filename = f"{filename}_aug_{i+1}{extension}"
                output_path = output_dir / output_filename
                augmented.save(output_path, quality=95)
                created += 1

            return created

        except Exception as e:
            print(f"Error augmenting {image_path}: {e}")
            return 0

    def augment_class(self, class_name):
        """
        Augmentasi semua gambar dalam satu kelas

        Args:
            class_name: Nama kelas (Lele, Patin, Nila, Gurame)

        Returns:
            Tuple of (original_count, augmented_count)
        """
        class_dir = TRAIN_DIR / class_name

        if not class_dir.exists():
            print(f"Warning: Directory {class_dir} tidak ditemukan")
            return 0, 0

        # Get output directory
        if self.output_dir:
            output_dir = self.output_dir / class_name
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = class_dir

        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            files = list(class_dir.glob(ext))
            # Filter out already augmented images
            files = [f for f in files if '_aug' not in f.stem]
            image_files.extend(files)

        original_count = len(image_files)
        augmented_count = 0

        print(f"\n{class_name}: {original_count} gambar original (excluding augmented)")

        # Augment each image
        for image_path in tqdm(image_files, desc=f"Augmenting {class_name}", unit="img"):
            created = self.augment_image(image_path, output_dir, self.augmentations_per_image)
            augmented_count += created

        return original_count, augmented_count

    def augment_all(self):
        """
        Augmentasi semua kelas

        Returns:
            Dictionary with augmentation statistics
        """
        print("\n" + "="*70)
        print("STARTING DATA AUGMENTATION")
        print("="*70)
        print(f"Augmentations per image: {self.augmentations_per_image}")
        print(f"Output directory: {self.output_dir if self.output_dir else 'Same as input'}")
        print("="*70)

        for class_name in self.classes:
            original, augmented = self.augment_class(class_name)

            self.stats['total_original'] += original
            self.stats['total_augmented'] += augmented
            self.stats['per_class'][class_name] = {
                'original': original,
                'augmented': augmented,
                'total': original + augmented
            }

        return self.stats

    def print_summary(self):
        """Print augmentation summary"""
        print("\n" + "="*70)
        print("AUGMENTATION SUMMARY")
        print("="*70)

        for class_name in self.classes:
            if class_name in self.stats['per_class']:
                stats = self.stats['per_class'][class_name]
                print(f"\n{class_name}:")
                print(f"  Original images:   {stats['original']:4d}")
                print(f"  Augmented images:  {stats['augmented']:4d}")
                print(f"  Total images:      {stats['total']:4d}")

        print(f"\n{'='*70}")
        print(f"TOTAL:")
        print(f"  Original images:   {self.stats['total_original']:4d}")
        print(f"  Augmented images:  {self.stats['total_augmented']:4d}")
        print(f"  Total images:      {self.stats['total_original'] + self.stats['total_augmented']:4d}")
        print("="*70)


def check_dataset():
    """Check if dataset exists"""
    if not TRAIN_DIR.exists():
        print(f"\nError: Training directory tidak ditemukan: {TRAIN_DIR}")
        print("\nPastikan struktur folder seperti ini:")
        print("dataset/")
        print("  └── train/")
        print("      ├── Lele/")
        print("      ├── Patin/")
        print("      ├── Nila/")
        print("      └── Gurame/")
        return False

    # Check each class
    total_images = 0
    for class_name in CLASSES:
        class_dir = TRAIN_DIR / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob('*.jpg'))) + \
                   len(list(class_dir.glob('*.jpeg'))) + \
                   len(list(class_dir.glob('*.png')))
            total_images += count

    if total_images == 0:
        print("\nError: Tidak ada gambar yang ditemukan di dataset/train/")
        return False

    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Augmentasi Data untuk Fish Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:

  # Buat 5 augmentasi per gambar (default)
  python augment_data.py

  # Buat 10 augmentasi per gambar
  python augment_data.py --num-aug 10

  # Simpan hasil ke folder terpisah
  python augment_data.py --output-dir dataset/augmented

  # Buat 3 augmentasi dan simpan ke folder terpisah
  python augment_data.py --num-aug 3 --output-dir dataset/augmented
        """
    )

    parser.add_argument(
        '--num-aug',
        type=int,
        default=5,
        help='Jumlah augmentasi per gambar (default: 5)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: save in same folder as original)'
    )

    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=CLASSES,
        choices=CLASSES,
        help='Kelas yang akan diaugmentasi (default: semua kelas)'
    )

    parser.add_argument(
        '--data',
        type=str,
        choices=CLASSES,
        help='Nama kelas ikan untuk augmentasi satu folder saja (alias for specific class)'
    )

    args = parser.parse_args()

    # Handle --data argument
    if args.data:
        args.classes = [args.data]

    print("\n" + "="*70)
    print("FISH CLASSIFICATION - DATA AUGMENTATION SCRIPT")
    print("="*70)

    # Check dataset
    if not check_dataset():
        print("\nDataset check failed. Exiting...")
        return

    # Konfigurasi
    print(f"\nKonfigurasi:")
    print(f"  - Augmentasi per gambar: {args.num_aug}")
    print(f"  - Output directory: {args.output_dir if args.output_dir else 'Same as input'}")
    print(f"  - Kelas: {', '.join(args.classes)}")

    # Create augmentor
    augmentor = DataAugmentor(
        augmentations_per_image=args.num_aug,
        output_dir=args.output_dir,
        classes=args.classes
    )

    response = input("\nLanjutkan augmentasi? (y/n): ")
    if response.lower() != 'y':
        print("Augmentasi dibatalkan.")
        return


    # Start augmentation
    start_time = datetime.now()
    stats = augmentor.augment_all()
    end_time = datetime.now()

    # Print summary
    augmentor.print_summary()

    duration = (end_time - start_time).total_seconds()
    print(f"\nWaktu yang dibutuhkan: {duration:.2f} detik")
    print("\n" + "="*70)
    print("AUGMENTASI SELESAI!")
    print("="*70)

    if args.output_dir:
        print(f"\nHasil augmentasi disimpan di: {args.output_dir}")
        print("\nUntuk menggunakan data augmentasi:")
        print("1. Copy folder augmented ke dataset/train/")
        print("2. Atau ubah TRAIN_DIR di train_model.py")
    else:
        print(f"\nHasil augmentasi disimpan di: {TRAIN_DIR}")
        print("\nData siap untuk training!")
        print("Jalankan: python train_model.py")


if __name__ == '__main__':
    main()
