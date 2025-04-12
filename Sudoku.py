import os
from sudoku_solver import SudokuSolver
from sudoku_vision import SudokuVision

def main():
    # Use fixed path
    image_path = "C:\\Users\\MarsWang\\Pictures\\Screenshots\\Screenshot 2025-01-05 152134.png"
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File does not exist - {image_path}")
        return
        
    try:
        # Process image
        vision = SudokuVision()
        sudoku_str = vision.process_image(image_path)
        
        # Solve Sudoku
        solver = SudokuSolver(sudoku_str)
        if solver.solve():
            print("\nSudoku Solution:")
            solver.print_grid()
        else:
            print("Unable to solve this Sudoku")
            
    except Exception as e:
        print(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()
