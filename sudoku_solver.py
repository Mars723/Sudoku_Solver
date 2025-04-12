class SudokuSolver:
    def __init__(self, grid_str):
        """通过字符串初始化数独
        Args:
            grid_str: 数独表格字符串，每行用/分隔，每个数字直接输入，0表示空位
        """
        # 解析输入的grid字符串
        rows = grid_str.strip().split('/')
        self.grid = []
        for row in rows:
            if not row.strip():  # 跳过空行
                continue
            # 将每个字符转换为数字
            numbers = [int(char) for char in row.strip()]
            self.grid.append(numbers)
            
        self.n = len(self.grid[0])  # 使用第一行的长度作为n
        self.box_size = int(self.n ** 0.5)  # 小方格的大小
        
        # 验证输入的格式是否正确
        if not all(len(row) == self.n for row in self.grid):
            raise ValueError("每行的数字数量必须相同")
    
    def is_valid(self, row, col, num):
        """检查在指定位置放置数字是否有效"""
        # 检查行
        if num in self.grid[row]:
            return False
            
        # 检查列
        if num in [self.grid[i][col] for i in range(self.n)]:
            return False
            
        # 检查小方格
        box_row = (row // self.box_size) * self.box_size
        box_col = (col // self.box_size) * self.box_size
        for i in range(box_row, box_row + self.box_size):
            for j in range(box_col, box_col + self.box_size):
                if self.grid[i][j] == num:
                    return False
        
        return True
    
    def find_empty(self):
        """找到一个空位置（值为0的格子）"""
        for i in range(self.n):
            for j in range(self.n):
                if self.grid[i][j] == 0:
                    return i, j
        return None
    
    def solve(self):
        """解决数独"""
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
        """打印数独网格"""
        for i in range(self.n):
            if i % self.box_size == 0 and i != 0:
                print("-" * (self.n * 2 + self.box_size + 1))
            
            for j in range(self.n):
                if j % self.box_size == 0 and j != 0:
                    print("|", end=" ")
                print(self.grid[i][j], end=" ")
            print() 