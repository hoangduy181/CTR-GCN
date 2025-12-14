# File: eda/visualizations/images_to_gif.py
"""
Convert skeleton frame images to GIF animations.
"""

from pathlib import Path
import glob

import imageio.v2 as imageio
USE_IMAGEIO = True
# Try imageio first (preferred)
# try:
# except ImportError:
#     USE_IMAGEIO = False
#     try:
#         from PIL import Image
#         USE_PIL = True
#     except ImportError:
#         print("ERROR: Install imageio or pillow:")
#         print("  pip install imageio")
#         print("  or: pip install pillow")
#         exit(1)


def create_gif_with_imageio(frame_paths, output_path, fps=10):
    """Create GIF using imageio"""
    images = [imageio.imread(str(fp)) for fp in sorted(frame_paths)]
    duration = 1.0 / fps
    imageio.mimsave(str(output_path), images, duration=duration, loop=0, format='GIF')


def create_gif_with_pil(frame_paths, output_path, fps=10):
    """Create GIF using PIL/Pillow"""
    images = []
    for fp in sorted(frame_paths):
        img = Image.open(fp)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)
    
    duration = int(1000 / fps)  # milliseconds
    images[0].save(
        str(output_path),
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        format='GIF'
    )


def convert_folder_to_gif(folder_path, fps=10):
    """Convert all PNG frames in a folder to a GIF"""
    folder = Path(folder_path)
    frame_files = sorted(folder.glob('frame_*.png'))
    
    if len(frame_files) == 0:
        print(f"  No frames found in {folder.name}")
        return
    
    output_path = folder.parent / f"{folder.name}.gif"
    print(f"  Creating {output_path.name} from {len(frame_files)} frames...")
    
    try:
        create_gif_with_imageio(frame_files, output_path, fps=fps)
        # if USE_IMAGEIO:
        # else:
        #     create_gif_with_pil(frame_files, output_path, fps=fps)
        print(f"  ✓ Done!")
    except Exception as e:
        print(f"  ✗ Error: {e}")


def main():
    """Convert all action folders to GIFs"""
    base_dir = Path('.')
    action_folders = sorted([d for d in base_dir.iterdir() 
                            if d.is_dir() and d.name.startswith('action_')])
    
    print(f"Found {len(action_folders)} action folders")
    print(f"Using {'imageio' if USE_IMAGEIO else 'PIL/Pillow'}\n")
    
    for folder in action_folders:
        print(f"Processing {folder.name}...")
        convert_folder_to_gif(folder, fps=10)
    
    print("\n✓ All GIFs created!")


if __name__ == "__main__":
    main()