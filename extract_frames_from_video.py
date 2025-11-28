#!/usr/bin/env python3
"""
Script untuk extract frames dari video ikan menjadi dataset
Usage: python extract_frames_from_video.py video_ikan.mp4 [--fps 2] [--quality 95]

Contoh:
  python extract_frames_from_video.py patin.mp4
  python extract_frames_from_video.py lele.mp4 --fps 3 --quality 90
  python extract_frames_from_video.py videos/*.mp4 --fps 1
"""

import cv2
import os
import sys
import argparse
from pathlib import Path
import numpy as np


def extract_frames(video_path, output_dir, fps=2, quality=95, similarity_threshold=0.95):
    """
    Extract frames dari video dan simpan sebagai images

    Args:
        video_path: Path ke video file
        output_dir: Directory untuk menyimpan frames
        fps: Berapa frame per detik yang diambil (default: 2)
        quality: JPEG quality 0-100 (default: 95)
        similarity_threshold: Skip frame jika terlalu mirip dengan frame sebelumnya (0-1)
    """
    # Buka video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ ERROR: Tidak bisa membuka video {video_path}")
        return 0

    # Dapatkan properties video
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"\n📹 Video Info:")
    print(f"   FPS: {video_fps:.2f}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Extracting {fps} frames per second...")

    # Hitung interval frame
    frame_interval = int(video_fps / fps) if fps > 0 else 1

    # Buat output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    saved_count = 0
    prev_frame_gray = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Ambil frame setiap interval
        if frame_count % frame_interval == 0:
            # Convert ke grayscale untuk similarity check
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Check similarity dengan frame sebelumnya
            save_frame = True
            if prev_frame_gray is not None:
                # Hitung structural similarity
                similarity = compute_similarity(prev_frame_gray, gray)

                if similarity > similarity_threshold:
                    save_frame = False  # Skip frame yang terlalu mirip

            if save_frame:
                # Generate filename
                filename = f"{output_dir.name}_{saved_count:04d}.jpeg"
                filepath = output_dir / filename

                # Save frame
                cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                saved_count += 1

                prev_frame_gray = gray

                # Progress indicator
                if saved_count % 10 == 0:
                    print(f"   Saved {saved_count} frames...", end='\r')

        frame_count += 1

    cap.release()

    print(f"\n✓ Selesai! Saved {saved_count} frames to {output_dir}")
    return saved_count


def compute_similarity(img1, img2):
    """Hitung similarity antara dua gambar (0-1, 1 = identik)"""
    # Resize jika ukuran berbeda
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Simple correlation-based similarity
    correlation = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
    return correlation


def get_fish_name_from_filename(video_filename):
    """
    Extract nama ikan dari filename video

    Examples:
        patin.mp4 -> Patin
        lele_001.mp4 -> Lele
        nila.avi -> Nila
        GURAME.MP4 -> Gurame
    """
    # Ambil nama file tanpa extension
    name = Path(video_filename).stem

    # Ambil kata pertama (sebelum underscore/angka)
    import re
    fish_name = re.split(r'[_\-\d]', name)[0]

    # Capitalize first letter
    fish_name = fish_name.capitalize()

    return fish_name


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames dari video ikan untuk dataset training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python extract_frames_from_video.py patin.mp4
  python extract_frames_from_video.py lele.mp4 --fps 3 --quality 90
  python extract_frames_from_video.py videos/*.mp4

Tips:
  - Gunakan --fps 1-3 untuk video yang bergerak cepat
  - Gunakan --fps 0.5 untuk video yang lambat/statis
  - Quality 90-95 recommended untuk training
        """
    )

    parser.add_argument('videos', nargs='+', help='Video file(s) untuk di-extract')
    parser.add_argument('--fps', type=float, default=2,
                       help='Berapa frame per detik yang diambil (default: 2)')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG quality 0-100 (default: 95)')
    parser.add_argument('--similarity', type=float, default=0.95,
                       help='Skip threshold untuk frame yang mirip 0-1 (default: 0.95)')
    parser.add_argument('--output', type=str, default='dataset/train',
                       help='Output directory base (default: dataset/train)')
    parser.add_argument('--validation', action='store_true',
                       help='Simpan ke dataset/validation instead of dataset/train')

    args = parser.parse_args()

    # Tentukan base output directory
    base_output = 'dataset/validation' if args.validation else args.output

    print("="*60)
    print("FISH VIDEO FRAME EXTRACTOR")
    print("="*60)
    print(f"Settings:")
    print(f"  FPS: {args.fps} frames/second")
    print(f"  Quality: {args.quality}")
    print(f"  Similarity threshold: {args.similarity}")
    print(f"  Output base: {base_output}")
    print("="*60)

    total_saved = 0

    for video_path in args.videos:
        video_path = Path(video_path)

        if not video_path.exists():
            print(f"\n⚠️  Video tidak ditemukan: {video_path}")
            continue

        # Dapatkan nama ikan dari filename
        fish_name = get_fish_name_from_filename(video_path.name)

        # Buat output directory
        output_dir = Path(base_output) / fish_name

        print(f"\n{'='*60}")
        print(f"Processing: {video_path.name}")
        print(f"Fish type: {fish_name}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")

        # Extract frames
        saved = extract_frames(
            video_path,
            output_dir,
            fps=args.fps,
            quality=args.quality,
            similarity_threshold=args.similarity
        )

        total_saved += saved

    print("\n" + "="*60)
    print(f"✓ SELESAI!")
    print(f"Total frames saved: {total_saved}")
    print("="*60)
    print("\nLangkah selanjutnya:")
    print("1. Cek hasil di folder", base_output)
    print("2. Hapus gambar yang blur/tidak jelas")
    print("3. Pastikan setiap class punya gambar cukup (min 50-100)")
    print("4. Jalankan: python train_model.py")
    print("="*60)


if __name__ == '__main__':
    main()
