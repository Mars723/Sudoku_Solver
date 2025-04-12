import os
from sudoku_solver import SudokuSolver
from sudoku_vision import SudokuVision

def main():
    # 使用固定路径
    image_path = "C:\\Users\\MarsWang\\Pictures\\Screenshots\\Screenshot 2025-01-05 152134.png"
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        return
        
    try:
        # 处理图像
        vision = SudokuVision()
        sudoku_str = vision.process_image(image_path)
        
        # 解决数独
        solver = SudokuSolver(sudoku_str)
        if solver.solve():
            print("\n数独解决方案：")
            solver.print_grid()
        else:
            print("无法解决此数独")
            
    except Exception as e:
        print(f"处理出错: {str(e)}")

if __name__ == "__main__":
    main()
