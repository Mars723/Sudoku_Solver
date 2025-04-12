import cv2
import numpy as np
import tensorflow as tf

class SudokuVision:
    def __init__(self):
        # 加载预训练的MNIST模型
        try:
            self.model = tf.keras.models.load_model('mnist_model.h5')
        except:
            print("正在训练MNIST模型...")
            self.train_mnist_model()
    
    def train_mnist_model(self):
        """训练一个更复杂的MNIST模型"""
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # 数据预处理
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        
        # 构建更复杂的模型
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # 编译模型
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        # 训练模型（增加训练轮数）
        model.fit(x_train.reshape(-1, 28, 28, 1), y_train, 
                  validation_data=(x_test.reshape(-1, 28, 28, 1), y_test),
                  epochs=10,
                  batch_size=128)
        
        # 保存模型
        model.save('mnist_model.h5')
        self.model = model

    def preprocess_image(self, image_path):
        """预处理图像"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("无法读取图像")
        
        # 调整图像大小，保持宽高比
        max_size = 1000
        h, w = img.shape[:2]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        # 转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 增加对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 高斯模糊
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        return thresh, img

    def find_largest_square(self, thresh):
        """找到最大的正方形"""
        # 反转图像（确保边框是白色）
        thresh = cv2.bitwise_not(thresh)
        
        # 使用形态学操作强化边框
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        # 找到所有轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 按面积排序
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # 遍历最大的几个轮廓
        for contour in contours[:5]:  # 只检查最大的5个轮廓
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # 如果是四边形
            if len(approx) == 4:
                # 检查是否接近正方形
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w)/h
                if 0.8 <= aspect_ratio <= 1.2:  # 放宽比例限制
                    return approx
        
        # 如果没有找到合适的轮廓，尝试使用整个图像
        h, w = thresh.shape[:2]
        return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    def enhance_cell(self, cell):
        """增强单元格图像预处理"""
        if cell.size == 0:
            return np.zeros((28, 28), dtype=np.uint8)
        
        # 转换为灰度图
        if len(cell.shape) == 3:
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        
        # 使用CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        cell = clahe.apply(cell)
        
        # 降噪处理
        cell = cv2.fastNlMeansDenoising(cell, None, 10, 7, 21)
        
        # 锐化处理
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        cell = cv2.filter2D(cell, -1, kernel)
        
        # 自适应二值化
        cell = cv2.adaptiveThreshold(
            cell, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            blockSize=13,
            C=4
        )
        
        # 形态学操作
        kernel_size = (2,2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        
        # 去除小噪点
        cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, kernel)
        
        # 填充小空洞
        cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)
        
        # 使用腐蚀操作代替细化
        kernel = np.ones((2,2), np.uint8)
        cell = cv2.erode(cell, kernel, iterations=1)
        
        return cell

    def extract_digits(self, img, square_contour):
        """从正方形中提取数字"""
        # 获取正方形的四个角点
        pts = square_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        # 按顺序排列角点（左上，右上，右下，左下）
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上
        rect[3] = pts[np.argmax(diff)]  # 左下
        
        # 计算正方形的宽度
        width = int(max(
            np.linalg.norm(rect[0] - rect[1]),
            np.linalg.norm(rect[2] - rect[3])
        ))
        
        # 创建透视变换矩阵
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, width - 1],
            [0, width - 1]
        ], dtype="float32")
        
        # 执行透视变换
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (width, width))
        
        # 分割成81个小格
        cells = []
        cell_size = width // 9
        
        for i in range(9):
            row = []
            for j in range(9):
                # 计算每个小格的边界
                x1 = j * cell_size
                y1 = i * cell_size
                x2 = (j + 1) * cell_size
                y2 = (i + 1) * cell_size
                
                # 提取小格并留出边距
                margin = int(cell_size * 0.18)  # 增加边距
                cell = warped[y1+margin:y2-margin, x1+margin:x2-margin]
                
                # 增强预处理
                cell = self.enhance_cell(cell)
                
                # 找到数字的边界框
                contours, _ = cv2.findContours(
                    cell, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    # 找到最大的轮廓
                    main_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(main_contour)
                    
                    # 面积阈值判断
                    min_area = cell.size * 0.03  # 最小面积阈值
                    max_area = cell.size * 0.8   # 最大面积阈值
                    
                    if min_area < area < max_area:
                        x, y, w, h = cv2.boundingRect(main_contour)
                        
                        # 提取数字区域并加入边距
                        padding = 4  # 增加padding
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(cell.shape[1] - x, w + 2*padding)
                        h = min(cell.shape[0] - y, h + 2*padding)
                        
                        # 确保提取的区域有效
                        if w > 0 and h > 0:
                            digit = cell[y:y+h, x:x+w]
                            
                            # 调整大小前先填充到正方形
                            size = max(h, w) + 4  # 增加额外边距
                            top = (size - h) // 2
                            bottom = size - h - top
                            left = (size - w) // 2
                            right = size - w - left
                            
                            digit = cv2.copyMakeBorder(
                                digit, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=0
                            )
                        else:
                            digit = np.zeros((28, 28), dtype=np.uint8)
                    else:
                        digit = np.zeros((28, 28), dtype=np.uint8)
                else:
                    digit = np.zeros((28, 28), dtype=np.uint8)
                
                # 调整到最终大小
                digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_CUBIC)
                
                # 最终的形态学处理
                kernel = np.ones((2,2), np.uint8)
                digit = cv2.morphologyEx(digit, cv2.MORPH_CLOSE, kernel)
                
                row.append(digit)
            cells.append(row)
        
        return cells

    def recognize_digit(self, cell, row=None, col=None):
        """使用MNIST模型识别数字，结合多重特征分析"""
        # 如果像素太少，认为是空白
        if cv2.countNonZero(cell) < 40:
            return 0
        
        # 确保是黑底白字
        if cv2.countNonZero(cell) > cell.size/2:
            cell = cv2.bitwise_not(cell)
        
        # 归一化
        normalized = cell.astype('float32') / 255
        
        # 多次预测，获取top3预测结果
        predictions = []
        for _ in range(3):  # 进行3次预测
            pred = self.model.predict(normalized.reshape(1, 28, 28, 1), verbose=0)
            predictions.append(pred[0])
        
        # 合并预测结果
        avg_prediction = np.mean(predictions, axis=0)
        top3_indices = np.argsort(avg_prediction)[-3:][::-1]
        confidences = avg_prediction[top3_indices]
        
        # 几何特征分析
        # 1. 轮廓分析
        contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            main_contour = max(contours, key=cv2.contourArea)
            # 计算轮廓特征
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # 计算最小外接矩形
            rect = cv2.minAreaRect(main_contour)
            box = cv2.boxPoints(rect)
            box = np.int_(box)
            width = min(rect[1])
            height = max(rect[1])
            aspect_ratio = height / width if width > 0 else 0
            
            # 计算凸包
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # 2. 区域分析
        h, w = cell.shape
        regions = []
        for i in range(3):
            for j in range(3):
                region = cell[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
                regions.append(cv2.countNonZero(region))
        
        # 3. 投影分析
        h_proj = np.sum(cell, axis=1)
        v_proj = np.sum(cell, axis=0)
        
        # 计算左右两侧的像素
        left_half = cell[:, :cell.shape[1]//2]
        right_half = cell[:, cell.shape[1]//2:]
        left_pixels = cv2.countNonZero(left_half)
        right_pixels = cv2.countNonZero(right_half)
        
        # 添加更多的图像分析
        def analyze_shape(img):
            # 计算图像的矩
            moments = cv2.moments(img)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                cx, cy = img.shape[1]//2, img.shape[0]//2
            
            # 计算图像的方向
            if moments['m20'] + moments['m02'] != 0:
                orientation = 0.5 * np.arctan2(2*moments['m11'], 
                                             moments['m20'] - moments['m02'])
            else:
                orientation = 0
                
            return cx, cy, orientation

        # 计算形状特征
        center_x, center_y, angle = analyze_shape(cell)
        is_centered = abs(center_x - cell.shape[1]//2) < cell.shape[1]//4
        is_vertical = abs(angle) < np.pi/6
        is_slanted = abs(angle) > np.pi/6
        
        # 特征向量构建
        def get_digit_features(digit):
            features = {
                1: {
                    'aspect_ratio': (2.0, 3.0),
                    'solidity': (0.4, 0.7),
                    'regions': [0,0,1, 0,1,0, 0,1,0],
                    'special_features': {
                        'vertical_line': True,
                        'no_horizontal': True,
                        'center_aligned': True
                    }
                },
                2: {
                    'aspect_ratio': (1.1, 1.8),
                    'solidity': (0.5, 0.8),
                    'regions': [1,1,1, 0,1,1, 1,1,0],
                    'special_features': {
                        'top_curve': True,
                        'bottom_left': True,
                        'middle_empty': True,
                        'right_top': True,
                        'diagonal_flow': True
                    }
                },
                3: {
                    'aspect_ratio': (1.1, 1.8),
                    'solidity': (0.5, 0.8),
                    'regions': [1,1,1, 0,1,1, 1,1,1],
                    'special_features': {
                        'right_curves': True,
                        'middle_connection': True,
                        'symmetric_sides': True
                    }
                },
                4: {
                    'aspect_ratio': (0.9, 1.6),
                    'solidity': (0.4, 0.7),
                    'regions': [1,0,1, 1,1,1, 0,0,1],
                    'special_features': {
                        'cross_point': True,
                        'vertical_right': True,
                        'horizontal_middle': True
                    }
                },
                5: {
                    'aspect_ratio': (1.1, 1.8),
                    'solidity': (0.5, 0.8),
                    'regions': [1,1,1, 1,1,0, 1,1,1],
                    'special_features': {
                        'top_horizontal': True,
                        'middle_bend': True,
                        'bottom_curve': True
                    }
                },
                6: {
                    'aspect_ratio': (1.1, 1.8),
                    'solidity': (0.6, 0.9),
                    'regions': [1,1,1, 1,0,0, 1,1,1],
                    'special_features': {
                        'left_vertical': True,
                        'bottom_loop': True,
                        'top_curve': True
                    }
                },
                7: {
                    'aspect_ratio': (1.2, 2.0),
                    'solidity': (0.4, 0.7),
                    'regions': [1,1,1, 0,0,1, 0,1,0],
                    'special_features': {
                        'top_horizontal': True,
                        'diagonal_line': True,
                        'right_slant': True
                    }
                },
                8: {
                    'aspect_ratio': (1.2, 1.8),
                    'solidity': (0.6, 0.9),
                    'regions': [1,1,1, 1,0,1, 1,1,1],
                    'special_features': {
                        'double_loop': True,
                        'center_pinch': True,
                        'symmetric': True
                    }
                },
                9: {
                    'aspect_ratio': (1.1, 1.8),
                    'solidity': (0.5, 0.8),
                    'regions': [1,1,1, 1,0,1, 0,1,1],
                    'special_features': {
                        'top_loop': True,
                        'right_line': True,
                        'bottom_curve': True,
                        'left_top': True
                    }
                }
            }
            return features.get(digit, {})
        
        # 特征检查函数
        def check_special_features(digit, sf):
            score = 0
            
            # 基本形状分析
            shape_score = 0
            if is_centered:
                shape_score += 0.2
            if is_vertical and digit in [1, 4, 6, 8, 9]:
                shape_score += 0.2
            if is_slanted and digit == 7:
                shape_score += 0.2
            
            # 区域连通性分析
            num_labels, labels = cv2.connectedComponents(cell)
            has_single_component = num_labels == 2  # 背景算一个component
            has_multiple_components = num_labels > 2
            
            if digit in [1, 7] and has_single_component:
                shape_score += 0.2
            if digit in [4, 8] and has_multiple_components:
                shape_score += 0.2
            
            score += shape_score

            # 数字特定特征检查
            if digit == 1:
                # 1的特征：单一垂直线，居中，几乎没有横线
                # 检查垂直线
                if sf['vertical_line']:
                    vertical_score = 0
                    # 检查中间区域的垂直连续性
                    middle_sum = sum(regions[1::3])  # 中间列的和
                    side_sum = sum(regions[0::3]) + sum(regions[2::3])  # 两侧列的和
                    if middle_sum > side_sum:  # 中间列比两侧列像素多
                        vertical_score += 0.5
                    # 检查垂直线的直度和位置
                    if self.has_vertical_line(v_proj, 'middle', 1.2):  # 降低阈值
                        vertical_score += 0.5
                    score += vertical_score

                # 检查是否居中
                if sf['center_aligned']:
                    center_score = 0
                    # 检查重心位置
                    if abs(center_x - cell.shape[1]//2) < cell.shape[1]//5:  # 放宽居中要求
                        center_score += 0.3
                    # 检查左右像素分布
                    if abs(left_pixels - right_pixels) < max(left_pixels, right_pixels) * 0.4:  # 放宽对称要求
                        center_score += 0.2
                    score += center_score

                # 检查是否缺少横线
                if sf['no_horizontal']:
                    h_score = 0
                    if max(h_proj) < np.mean(h_proj) * 1.8:  # 放宽横线限制
                        h_score += 0.3
                    # 检查上下部分是否较少像素
                    if sum(regions[0:3]) + sum(regions[6:]) < sum(regions[3:6]) * 0.6:  # 放宽比例要求
                        h_score += 0.2
                    score += h_score

                # 额外的1特征检查
                extra_score = 0
                # 检查高度是否足够
                if aspect_ratio > 1.5:  # 确保足够细长
                    extra_score += 0.2
                # 检查连通性
                if has_single_component:  # 应该是单个连通区域
                    extra_score += 0.2
                # 检查像素分布的连续性
                middle_pixels = [regions[1], regions[4], regions[7]]
                if all(p > 0 for p in middle_pixels):  # 中间列应该连续
                    extra_score += 0.1
                score += extra_score

            elif digit == 7:
                # 7的特征：上横线，右斜线，无底线
                # 检查上横线
                if sf['top_horizontal']:
                    top_score = 0
                    # 检查上部横线的强度和位置
                    top_line = max(h_proj[:5])
                    if top_line > np.mean(h_proj) * 1.2:  # 降低阈值
                        top_score += 0.3
                    # 确保是在最上方
                    top_line_pos = np.argmax(h_proj[:7])
                    if top_line_pos < 4:  # 放宽位置要求
                        top_score += 0.2
                    score += top_score

                # 检查斜线特征
                if sf['diagonal_line']:
                    diagonal_score = 0
                    # 检查从右上到左下的像素分布
                    if regions[2] > regions[0] and regions[5] > regions[3]:
                        diagonal_score += 0.3
                    # 检查斜线的连续性
                    if regions[2] > 0 and regions[5] > 0:  # 只需要主要部分连续
                        diagonal_score += 0.2
                    # 检查整体倾斜度
                    if is_slanted:
                        diagonal_score += 0.2
                    score += diagonal_score

                # 检查底部特征
                if sf['right_slant']:
                    bottom_score = 0
                    # 确保底部较少像素
                    if sum(regions[6:]) < sum(regions[0:3]) * 0.9:  # 放宽比例要求
                        bottom_score += 0.2
                    # 检查右下角的特征
                    if regions[8] > regions[6]:
                        bottom_score += 0.2
                    score += bottom_score

                # 额外的7特征检查
                extra_score = 0
                # 检查上部横线与斜线的关系
                if max(h_proj[:5]) > max(h_proj[5:]) * 1.1:
                    extra_score += 0.2
                # 检查整体形状
                if has_single_component:  # 应该是单个连通区域
                    extra_score += 0.1
                score += extra_score

            elif digit == 9:
                # 9的特征：上部闭环，右侧竖线，整体形状
                score = 0
                
                # 检查上部闭环
                if sf['top_loop']:
                    top_score = 0
                    # 检查上部区域的像素分布
                    top_region = sum(regions[0:3])
                    if top_region > sum(regions[6:9]) * 1.2:
                        top_score += 0.3
                    # 检查上部的圆形特征
                    if regions[1] > max(regions[0], regions[2]) * 0.8:
                        top_score += 0.2
                    # 检查上部连通性
                    if all(r > 0 for r in regions[0:3]):
                        top_score += 0.2
                    score += top_score

                # 检查右侧竖线
                if sf['right_line']:
                    right_score = 0
                    # 检查右侧列的像素分布
                    right_col = sum(regions[2::3])
                    if right_col > sum(regions[0::3]) * 1.1:
                        right_score += 0.3
                    # 检查右侧线的直度
                    if self.has_vertical_line(v_proj, 'right', 1.2):
                        right_score += 0.2
                    score += right_score

                # 检查整体形状特征
                shape_score = 0
                # 检查上下部分的比例
                if sum(regions[0:6]) > sum(regions[3:9]) * 1.1:
                    shape_score += 0.2
                # 检查左右对称性
                if abs(regions[0] - regions[2]) < np.mean(regions) * 0.4:
                    shape_score += 0.2
                # 检查底部弧线
                if regions[7] > regions[6] and regions[7] > regions[8]:
                    shape_score += 0.2
                score += shape_score

                # 额外的9特征检查
                extra_score = 0
                # 检查整体高宽比
                if 1.1 <= aspect_ratio <= 1.8:
                    extra_score += 0.1
                # 检查闭环特征
                if circularity > 0.4:
                    extra_score += 0.1
                # 检查连通性
                if has_single_component:
                    extra_score += 0.1
                score += extra_score

                # 位置相关的特征
                if row is not None and col is not None:
                    if row >= 6 and col <= 2:  # 左下角位置
                        score *= 1.1  # 略微提高得分

                return score / 3

            return score / 3
        
        # 计算特征匹配得分
        def get_feature_score(digit):
            features = get_digit_features(digit)
            if not features:
                return 0
            
            score = 0
            # 基本特征检查
            if features['aspect_ratio'][0] <= aspect_ratio <= features['aspect_ratio'][1]:
                score += 1
            if features['solidity'][0] <= solidity <= features['solidity'][1]:
                score += 1
            
            # 区域模式检查（更严格的匹配要求）
            region_pattern = features['regions']
            matches = 0
            total = 0
            for i in range(9):
                if region_pattern[i]:
                    total += 1
                    if regions[i] > np.mean(regions) * 0.8:  # 更严格的阈值
                        matches += 1
            if total > 0:
                score += matches / total
            
            # 特殊特征检查
            if 'special_features' in features:
                special_score = 0
                sf = features['special_features']
                
                special_score += check_special_features(digit, sf)
                
                score += special_score
            
            return score
        
        # 综合分析
        best_digit = None
        best_score = -1
        
        for digit, conf in zip(top3_indices, confidences):
            feature_score = get_feature_score(digit)
            position_score = 0
            
            # 位置相关的调整
            if row is not None and col is not None:
                if digit == 9 and row >= 6 and col <= 2:  # 左下角的9
                    position_score = 0.3
                    conf_threshold = 0.6  # 降低要求
                elif digit == 7 and is_slanted:  # 倾斜的7
                    position_score = 0.2
                    conf_threshold = 0.65
                else:
                    conf_threshold = 0.7
            
            # 综合评分
            total_score = (conf * 0.4 + 
                          feature_score * 0.4 + 
                          position_score * 0.2)
            
            if total_score > best_score and conf > conf_threshold:
                best_score = total_score
                best_digit = digit
        
        # 返回最佳匹配的数字
        if best_score > 0.65:
            return best_digit
        
        return 0

    def process_image(self, image_path):
        """处理图像并返回数独字符串"""
        try:
            thresh, img = self.preprocess_image(image_path)
            square = self.find_largest_square(thresh)
            cells = self.extract_digits(img, square)
            
            # 创建显示网格
            cell_size = 60
            gap = 2
            grid_size = 9 * cell_size + 8 * gap
            display_grid = np.full((grid_size, grid_size, 3), 255, dtype=np.uint8)
            
            # 识别数字并构建数独字符串
            recognized_grid = []
            for i in range(9):
                row = []
                for j in range(9):
                    cell = cells[i][j]
                    digit = self.recognize_digit(cell, row=i, col=j)
                    row.append(digit)
                    
                    # 显示单元格
                    y1 = i * (cell_size + gap)
                    x1 = j * (cell_size + gap)
                    cell_display = cv2.resize(cell, (cell_size, cell_size))
                    cell_display = cv2.cvtColor(cell_display, cv2.COLOR_GRAY2BGR)
                    display_grid[y1:y1+cell_size, x1:x1+cell_size] = cell_display
                    
                    if digit != 0:
                        cv2.putText(display_grid, str(digit), 
                                  (x1+15, y1+45), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, (0, 255, 0), 2)
                recognized_grid.append(row)
            
            # 显示识别结果
            print("\n识别到的数独：")
            self.print_grid(recognized_grid)
            
            # 创建窗口并显示
            cv2.namedWindow('Sudoku Recognition', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Sudoku Recognition', grid_size, grid_size)
            cv2.imshow('Sudoku Recognition', display_grid)
            cv2.waitKey(1)  # 确保窗口显示
            
            # 允许用户修改
            while True:
                choice = input("\n是否需要修改识别结果？(y/n): ").lower()
                if choice == 'n':
                    break
                elif choice == 'y':
                    try:
                        row = int(input("输入行号(1-9): ")) - 1
                        col = int(input("输入列号(1-9): ")) - 1
                        value = int(input("输入新的数字(0-9，0表示空): "))
                        
                        if 0 <= row < 9 and 0 <= col < 9 and 0 <= value <= 9:
                            recognized_grid[row][col] = value
                            # 更新显示
                            y1 = row * (cell_size + gap)
                            x1 = col * (cell_size + gap)
                            # 清除原有数字
                            cv2.rectangle(display_grid, 
                                        (x1+10, y1+10), 
                                        (x1+cell_size-10, y1+cell_size-10), 
                                        (255, 255, 255), 
                                        -1)
                            if value != 0:
                                cv2.putText(display_grid, str(value), 
                                          (x1+15, y1+45), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 
                                          1, (0, 0, 255), 2)  # 修改的数字用红色显示
                            
                            cv2.imshow('Sudoku Recognition', display_grid)
                            cv2.waitKey(1)  # 确保窗口更新
                            
                            print("\n当前数独：")
                            self.print_grid(recognized_grid)
                        else:
                            print("输入的数值超出范围！")
                    except ValueError:
                        print("请输入有效的数字！")
                else:
                    print("请输入 y 或 n")
            
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # 确保窗口完全关闭
            
            # 构建数独字符串
            sudoku_str = ""
            for i in range(9):
                if i > 0:
                    sudoku_str += "/"
                for j in range(9):
                    sudoku_str += str(recognized_grid[i][j])
            
            return sudoku_str
            
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            print("\n程序被用户中断")
            raise
        except Exception as e:
            cv2.destroyAllWindows()
            print(f"\n处理图像时出错: {str(e)}")
            raise

    def print_grid(self, grid):
        """打印数独网格"""
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("-" * 25)
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    print("|", end=" ")
                print(grid[i][j], end=" ")
            print()

    # 特征分析函数
    def has_horizontal_line(self, proj, pos='top', threshold=1.5):
        if pos == 'top':
            line = max(proj[:5])
        elif pos == 'middle':
            line = max(proj[12:16])
        else:  # bottom
            line = max(proj[-5:])
        return line > np.mean(proj) * threshold
    
    def has_vertical_line(self, proj, pos='middle', threshold=1.5):
        if pos == 'left':
            line = max(proj[:10])
        elif pos == 'middle':
            line = max(proj[12:16])
        else:  # right
            line = max(proj[-10:])
        return line > np.mean(proj) * threshold