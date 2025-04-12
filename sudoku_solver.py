class SudokuSolver:
    def __init__(self, grid_str):
        """Initialize Sudoku from a string
        Args:
            grid_str: Sudoku grid string, each row separated by '/', each digit directly input, 0 represents empty cell
        """
        # Parse the input grid string
        rows = grid_str.strip().split('/')
        self.grid = []
        for row in rows:
            if not row.strip():  # Skip empty rows
                continue
            # Convert each character to number
            numbers = [int(char) for char in row.strip()]
            self.grid.append(numbers)
            
        self.n = len(self.grid[0])  # Use the length of the first row as n
        self.box_size = int(self.n ** 0.5)  # Size of the sub-grid
        
        # Validate the input format
        if not all(len(row) == self.n for row in self.grid):
            raise ValueError("Each row must have the same number of digits")
    
    def is_valid(self, row, col, num):
        """Check if placing a number at the specified position is valid"""
        # Check row
        if num in self.grid[row]:
            return False
            
        # Check column
        if num in [self.grid[i][col] for i in range(self.n)]:
            return False
            
        # Check sub-grid
        box_row = (row // self.box_size) * self.box_size
        box_col = (col // self.box_size) * self.box_size
        for i in range(box_row, box_row + self.box_size):
            for j in range(box_col, box_col + self.box_size):
                if self.grid[i][j] == num:
                    return False
        
        return True
    
    def find_empty(self):
        """Find an empty cell (value 0)"""
        for i in range(self.n):
            for j in range(self.n):
                if self.grid[i][j] == 0:
                    return i, j
        return None
    
    def solve(self):
        """Solve the Sudoku puzzle"""
        empty = self.find_empty()
        if not empty:
            return True
        
        row, col = empty
        for num in range(1, self.n + 1):
            if self.is_valid(row, col, num):
                self.grid[row][col] = num
                if self.solve():
                    return True
                self.grid[row][col] = 0
        
        return False
    
    def print_grid(self):
        """Print the Sudoku grid"""
        for i in range(self.n):
            if i % self.box_size == 0 and i != 0:
                print("-" * (self.n * 2 + self.box_size + 1))
            
            for j in range(self.n):
                if j % self.box_size == 0 and j != 0:
                    print("|", end=" ")
                print(self.grid[i][j], end=" ")
            print() 