# Sudoku Solver

A computer vision-based Sudoku puzzle solver that can recognize Sudoku puzzles from images and solve them automatically.

## Features

- Uses computer vision to extract Sudoku grids from images
- Employs a custom-trained CNN model for digit recognition
- Efficient backtracking algorithm for solving difficult Sudoku puzzles
- Supports standard 9×9 Sudoku puzzles

## Tech Stack

- Python
- TensorFlow (for digit recognition)
- OpenCV (for image processing)
- NumPy

## Installation

```bash
pip install tensorflow opencv-python numpy
```

## Usage

1. Save the Sudoku puzzle image locally
2. Modify the image path in `Sudoku.py`
3. Run the main program:

```bash
python Sudoku.py
```

## Project Structure

- `Sudoku.py` - Main program entry point
- `sudoku_solver.py` - Sudoku solving algorithm
- `sudoku_vision.py` - Computer vision module
- `mnist_model.h5` - Pre-trained digit recognition model
