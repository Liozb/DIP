
# Digital Image Processing (DIP)

A collection of foundational **Digital Image Processing** implementations using only **NumPy** — no external image processing libraries like OpenCV or PIL are used.  
This repository is ideal for learning and experimenting with the basics of image manipulation and computer vision at the pixel level.

## Features

- 🧮 Implemented entirely with NumPy
- 🎨 Basic image processing operations:
  - Contrast expansion
  - 2D convolution
  - Gradient computation
  - Interpolation methods
  - Panoramic image generation
  - Spectrum analysis

## File Overview

- `contrast_expanding.py` – Enhances image contrast through expansion techniques.
- `conv2d.py` – Performs 2D convolution for filtering and edge detection.
- `gradients.py` – Calculates image gradients to detect edges and texture.
- `interpolations.py` – Implements interpolation methods (e.g., nearest, bilinear).
- `panoramic.py` – Builds panoramic images using simple stitching logic.
- `spectrum.py` – Computes and analyzes image frequency spectrum.
- `pic/` – Folder containing example image files used by the scripts.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/DIP.git
   cd DIP
   ```
2. Make sure you have NumPy installed:
   ```bash
   pip install numpy
   ```
3. Run any script of interest:
   ```bash
   python contrast_expanding.py
   ```

## License

This project is open-source and available for educational or research purposes. Contributions and improvements are welcome!

---
