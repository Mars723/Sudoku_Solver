import cv2
import numpy as np
import tensorflow as tf

class SudokuVision:
    def __init__(self):
        # Load pre-trained MNIST model
        try:
            self.model = tf.keras.models.load_model('mnist_model.h5')
        except:
            print("Training MNIST model...")
            self.train_mnist_model()
    
    def train_mnist_model(self):
        """Train a more complex MNIST model"""
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Data preprocessing
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        
        # Build a more complex model
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
        
        # Compile model
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Train model (increase training epochs)
        model.fit(x_train.reshape(-1, 28, 28, 1), y_train, 
                  validation_data=(x_test.reshape(-1, 28, 28, 1), y_test),
                  epochs=10,
                  batch_size=128)
        
        # Save model
        model.save('mnist_model.h5')
        self.model = model

    def preprocess_image(self, image_path):
        """Preprocess image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        
        # Resize image while maintaining aspect ratio
        max_size = 1000
        h, w = img.shape[:2]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        return thresh, img

    def find_largest_square(self, thresh):
        """Find the largest square"""
        # Invert image (ensure borders are white)
        thresh = cv2.bitwise_not(thresh)
        
        # Use morphological operations to strengthen borders
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        # Find all contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Iterate through the largest contours
        for contour in contours[:5]:  # Only check the 5 largest contours
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # If it's a quadrilateral
            if len(approx) == 4:
                # Check if it's close to a square
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w)/h
                if 0.8 <= aspect_ratio <= 1.2:  # Relax ratio constraint
                    return approx
        
        # If no suitable contour is found, try using the entire image
        h, w = thresh.shape[:2]
        return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    def enhance_cell(self, cell):
        """Enhance cell image preprocessing"""
        if cell.size == 0:
            return np.zeros((28, 28), dtype=np.uint8)
        
        # Convert to grayscale
        if len(cell.shape) == 3:
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        
        # Use CLAHE to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        cell = clahe.apply(cell)
        
        # Noise reduction
        cell = cv2.fastNlMeansDenoising(cell, None, 10, 7, 21)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        cell = cv2.filter2D(cell, -1, kernel)
        
        # Adaptive thresholding
        cell = cv2.adaptiveThreshold(
            cell, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            blockSize=13,
            C=4
        )
        
        # Morphological operations
        kernel_size = (2,2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        
        # Remove small noise
        cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, kernel)
        
        # Fill small holes
        cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)
        
        # Use erosion operation instead of thinning
        kernel = np.ones((2,2), np.uint8)
        cell = cv2.erode(cell, kernel, iterations=1)
        
        return cell

    def extract_digits(self, img, square_contour):
        """Extract digits from the square"""
        # Get the four corners of the square
        pts = square_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sort the corners (top-left, top-right, bottom-right, bottom-left)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        # Calculate the width of the square
        width = int(max(
            np.linalg.norm(rect[0] - rect[1]),
            np.linalg.norm(rect[2] - rect[3])
        ))
        
        # Create perspective transformation matrix
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, width - 1],
            [0, width - 1]
        ], dtype="float32")
        
        # Apply perspective transformation
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (width, width))
        
        # Split into 81 small cells
        cells = []
        cell_size = width // 9
        
        for i in range(9):
            row = []
            for j in range(9):
                # Calculate the boundaries of each small cell
                x1 = j * cell_size
                y1 = i * cell_size
                x2 = (j + 1) * cell_size
                y2 = (i + 1) * cell_size
                
                # Extract the cell and leave margin
                margin = int(cell_size * 0.18)  # Add margin
                cell = warped[y1+margin:y2-margin, x1+margin:x2-margin]
                
                # Enhance preprocessing
                cell = self.enhance_cell(cell)
                
                # Find the boundaries of the digit
                contours, _ = cv2.findContours(
                    cell, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    # Find the largest contour
                    main_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(main_contour)
                    
                    # Area threshold judgment
                    min_area = cell.size * 0.03  # Minimum area threshold
                    max_area = cell.size * 0.8   # Maximum area threshold
                    
                    if min_area < area < max_area:
                        x, y, w, h = cv2.boundingRect(main_contour)
                        
                        # Extract the digit area and add margin
                        padding = 4  # Add padding
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(cell.shape[1] - x, w + 2*padding)
                        h = min(cell.shape[0] - y, h + 2*padding)
                        
                        # Ensure the extracted area is valid
                        if w > 0 and h > 0:
                            digit = cell[y:y+h, x:x+w]
                            
                            # Fill to square before resizing
                            size = max(h, w) + 4  # Add extra margin
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
                
                # Adjust to final size
                digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_CUBIC)
                
                # Final morphological processing
                kernel = np.ones((2,2), np.uint8)
                digit = cv2.morphologyEx(digit, cv2.MORPH_CLOSE, kernel)
                
                row.append(digit)
            cells.append(row)
        
        return cells

    def recognize_digit(self, cell, row=None, col=None):
        """Use MNIST model to recognize digit, combined with multi-feature analysis"""
        # If pixel too few, consider it blank
        if cv2.countNonZero(cell) < 40:
            return 0
        
        # Ensure black background
        if cv2.countNonZero(cell) > cell.size/2:
            cell = cv2.bitwise_not(cell)
        
        # Normalize
        normalized = cell.astype('float32') / 255
        
        # Multiple predictions, get top3 prediction results
        predictions = []
        for _ in range(3):  # Perform 3 predictions
            pred = self.model.predict(normalized.reshape(1, 28, 28, 1), verbose=0)
            predictions.append(pred[0])
        
        # Merge prediction results
        avg_prediction = np.mean(predictions, axis=0)
        top3_indices = np.argsort(avg_prediction)[-3:][::-1]
        confidences = avg_prediction[top3_indices]
        
        # Geometric feature analysis
        # 1. Contour analysis
        contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            main_contour = max(contours, key=cv2.contourArea)
            # Calculate contour features
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # Calculate minimum bounding rectangle
            rect = cv2.minAreaRect(main_contour)
            box = cv2.boxPoints(rect)
            box = np.int_(box)
            width = min(rect[1])
            height = max(rect[1])
            aspect_ratio = height / width if width > 0 else 0
            
            # Calculate convex hull
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # 2. Area analysis
        h, w = cell.shape
        regions = []
        for i in range(3):
            for j in range(3):
                region = cell[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
                regions.append(cv2.countNonZero(region))
        
        # 3. Projection analysis
        h_proj = np.sum(cell, axis=1)
        v_proj = np.sum(cell, axis=0)
        
        # Calculate pixels on both sides
        left_half = cell[:, :cell.shape[1]//2]
        right_half = cell[:, cell.shape[1]//2:]
        left_pixels = cv2.countNonZero(left_half)
        right_pixels = cv2.countNonZero(right_half)
        
        # Add more image analysis
        def analyze_shape(img):
            # Calculate image moments
            moments = cv2.moments(img)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                cx, cy = img.shape[1]//2, img.shape[0]//2
            
            # Calculate image orientation
            if moments['m20'] + moments['m02'] != 0:
                orientation = 0.5 * np.arctan2(2*moments['m11'], 
                                             moments['m20'] - moments['m02'])
            else:
                orientation = 0
                
            return cx, cy, orientation

        # Calculate shape features
        center_x, center_y, angle = analyze_shape(cell)
        is_centered = abs(center_x - cell.shape[1]//2) < cell.shape[1]//4
        is_vertical = abs(angle) < np.pi/6
        is_slanted = abs(angle) > np.pi/6
        
        # Feature vector construction
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
        
        # Feature check function
        def check_special_features(digit, sf):
            score = 0
            
            # Basic shape analysis
            shape_score = 0
            if is_centered:
                shape_score += 0.2
            if is_vertical and digit in [1, 4, 6, 8, 9]:
                shape_score += 0.2
            if is_slanted and digit == 7:
                shape_score += 0.2
            
            # Area connectivity analysis
            num_labels, labels = cv2.connectedComponents(cell)
            has_single_component = num_labels == 2  # Background counts as one component
            has_multiple_components = num_labels > 2
            
            if digit in [1, 7] and has_single_component:
                shape_score += 0.2
            if digit in [4, 8] and has_multiple_components:
                shape_score += 0.2
            
            score += shape_score

            # Digital specific feature check
            if digit == 1:
                # 1's feature: Single vertical line, centered, almost no horizontal lines
                # Check vertical line
                if sf['vertical_line']:
                    vertical_score = 0
                    # Check vertical continuity in the middle area
                    middle_sum = sum(regions[1::3])  # Sum of middle column
                    side_sum = sum(regions[0::3]) + sum(regions[2::3])  # Sum of sides
                    if middle_sum > side_sum:  # Middle column has more pixels than sides
                        vertical_score += 0.5
                    # Check vertical line's straightness and position
                    if self.has_vertical_line(v_proj, 'middle', 1.2):  # Lower threshold
                        vertical_score += 0.5
                    score += vertical_score

                # Check if centered
                if sf['center_aligned']:
                    center_score = 0
                    # Check center of gravity position
                    if abs(center_x - cell.shape[1]//2) < cell.shape[1]//5:  # Relax center requirement
                        center_score += 0.3
                    # Check left and right pixel distribution
                    if abs(left_pixels - right_pixels) < max(left_pixels, right_pixels) * 0.4:  # Relax symmetry requirement
                        center_score += 0.2
                    score += center_score

                # Check if missing horizontal lines
                if sf['no_horizontal']:
                    h_score = 0
                    if max(h_proj) < np.mean(h_proj) * 1.8:  # Relax horizontal line restriction
                        h_score += 0.3
                    # Check if less pixels in upper and lower parts
                    if sum(regions[0:3]) + sum(regions[6:]) < sum(regions[3:6]) * 0.6:  # Relax ratio requirement
                        h_score += 0.2
                    score += h_score

                # Additional 1 feature check
                extra_score = 0
                # Check if height is sufficient
                if aspect_ratio > 1.5:  # Ensure sufficient elongation
                    extra_score += 0.2
                # Check connectivity
                if has_single_component:  # Should be single connected area
                    extra_score += 0.2
                # Check continuous pixel distribution
                middle_pixels = [regions[1], regions[4], regions[7]]
                if all(p > 0 for p in middle_pixels):  # Middle column should be continuous
                    extra_score += 0.1
                score += extra_score

            elif digit == 7:
                # 7's feature: Top horizontal line, right slant line, no bottom line
                # Check top horizontal line
                if sf['top_horizontal']:
                    top_score = 0
                    # Check top horizontal line strength and position
                    top_line = max(h_proj[:5])
                    if top_line > np.mean(h_proj) * 1.2:  # Lower threshold
                        top_score += 0.3
                    # Ensure it's at the top
                    top_line_pos = np.argmax(h_proj[:7])
                    if top_line_pos < 4:  # Relax position requirement
                        top_score += 0.2
                    score += top_score

                # Check slant line feature
                if sf['diagonal_line']:
                    diagonal_score = 0
                    # Check pixel distribution from top right to left bottom
                    if regions[2] > regions[0] and regions[5] > regions[3]:
                        diagonal_score += 0.3
                    # Check slant line continuity
                    if regions[2] > 0 and regions[5] > 0:  # Only need main part to be continuous
                        diagonal_score += 0.2
                    # Check overall slant
                    if is_slanted:
                        diagonal_score += 0.2
                    score += diagonal_score

                # Check bottom feature
                if sf['right_slant']:
                    bottom_score = 0
                    # Ensure bottom has less pixels
                    if sum(regions[6:]) < sum(regions[0:3]) * 0.9:  # Relax ratio requirement
                        bottom_score += 0.2
                    # Check right bottom feature
                    if regions[8] > regions[6]:
                        bottom_score += 0.2
                    score += bottom_score

                # Additional 7 feature check
                extra_score = 0
                # Check relationship between top horizontal line and slant line
                if max(h_proj[:5]) > max(h_proj[5:]) * 1.1:
                    extra_score += 0.2
                # Check overall shape
                if has_single_component:  # Should be single connected area
                    extra_score += 0.1
                score += extra_score

            elif digit == 9:
                # 9's feature: Top closed loop, right vertical line, overall shape
                score = 0
                
                # Check top closed loop
                if sf['top_loop']:
                    top_score = 0
                    # Check pixel distribution in top area
                    top_region = sum(regions[0:3])
                    if top_region > sum(regions[6:9]) * 1.2:
                        top_score += 0.3
                    # Check top circular feature
                    if regions[1] > max(regions[0], regions[2]) * 0.8:
                        top_score += 0.2
                    # Check top connectivity
                    if all(r > 0 for r in regions[0:3]):
                        top_score += 0.2
                    score += top_score

                # Check right vertical line
                if sf['right_line']:
                    right_score = 0
                    # Check pixel distribution in right column
                    right_col = sum(regions[2::3])
                    if right_col > sum(regions[0::3]) * 1.1:
                        right_score += 0.3
                    # Check right line straightness
                    if self.has_vertical_line(v_proj, 'right', 1.2):
                        right_score += 0.2
                    score += right_score

                # Check overall shape feature
                shape_score = 0
                # Check upper and lower part ratio
                if sum(regions[0:6]) > sum(regions[3:9]) * 1.1:
                    shape_score += 0.2
                # Check left and right symmetry
                if abs(regions[0] - regions[2]) < np.mean(regions) * 0.4:
                    shape_score += 0.2
                # Check bottom arc
                if regions[7] > regions[6] and regions[7] > regions[8]:
                    shape_score += 0.2
                score += shape_score

                # Additional 9 feature check
                extra_score = 0
                # Check overall aspect ratio
                if 1.1 <= aspect_ratio <= 1.8:
                    extra_score += 0.1
                # Check closed loop feature
                if circularity > 0.4:
                    extra_score += 0.1
                # Check connectivity
                if has_single_component:
                    extra_score += 0.1
                score += extra_score

                # Position related feature
                if row is not None and col is not None:
                    if row >= 6 and col <= 2:  # Left bottom position
                        score *= 1.1  # Slightly increase score

                return score / 3

                return score / 3
        
        # Calculate feature matching score
        def get_feature_score(digit):
            features = get_digit_features(digit)
            if not features:
                return 0
            
            score = 0
            # Basic feature check
            if features['aspect_ratio'][0] <= aspect_ratio <= features['aspect_ratio'][1]:
                score += 1
            if features['solidity'][0] <= solidity <= features['solidity'][1]:
                score += 1
            
            # Area pattern check (Stricter matching requirement)
            region_pattern = features['regions']
            matches = 0
            total = 0
            for i in range(9):
                if region_pattern[i]:
                    total += 1
                    if regions[i] > np.mean(regions) * 0.8:  # Strict threshold
                        matches += 1
            if total > 0:
                score += matches / total
            
            # Special feature check
            if 'special_features' in features:
                special_score = 0
                sf = features['special_features']
                
                special_score += check_special_features(digit, sf)
                
                score += special_score
            
            return score
        
        # Comprehensive analysis
        best_digit = None
        best_score = -1
        
        for digit, conf in zip(top3_indices, confidences):
            feature_score = get_feature_score(digit)
            position_score = 0
            
            # Position related adjustment
            if row is not None and col is not None:
                if digit == 9 and row >= 6 and col <= 2:  # Left bottom 9
                    position_score = 0.3
                    conf_threshold = 0.6  # Lower requirement
                elif digit == 7 and is_slanted:  # Slanted 7
                    position_score = 0.2
                    conf_threshold = 0.65
                else:
                    conf_threshold = 0.7
            
            # Comprehensive score
            total_score = (conf * 0.4 + 
                          feature_score * 0.4 + 
                          position_score * 0.2)
            
            if total_score > best_score and conf > conf_threshold:
                best_score = total_score
                best_digit = digit
        
        # Return best matched digit
        if best_score > 0.65:
            return best_digit
        
        return 0

    def process_image(self, image_path):
        """Process image and return Sudoku string"""
        try:
            thresh, img = self.preprocess_image(image_path)
            square = self.find_largest_square(thresh)
            cells = self.extract_digits(img, square)
            
            # Create display grid
            cell_size = 60
            gap = 2
            grid_size = 9 * cell_size + 8 * gap
            display_grid = np.full((grid_size, grid_size, 3), 255, dtype=np.uint8)
            
            # Recognize digits and build Sudoku string
            recognized_grid = []
            for i in range(9):
                row = []
                for j in range(9):
                    cell = cells[i][j]
                    digit = self.recognize_digit(cell, row=i, col=j)
                    row.append(digit)
                    
                    # Display cell
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
            
            # Display recognition result
            print("\nRecognized Sudoku:")
            self.print_grid(recognized_grid)
            
            # Create window and display
            cv2.namedWindow('Sudoku Recognition', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Sudoku Recognition', grid_size, grid_size)
            cv2.imshow('Sudoku Recognition', display_grid)
            cv2.waitKey(1)  # Ensure window display
            
            # Allow user to modify
            while True:
                choice = input("\nDo you want to modify the recognized result? (y/n): ").lower()
                if choice == 'n':
                    break
                elif choice == 'y':
                    try:
                        row = int(input("Enter row number (1-9): ")) - 1
                        col = int(input("Enter column number (1-9): ")) - 1
                        value = int(input("Enter new digit (0-9, 0 represents empty): "))
                        
                        if 0 <= row < 9 and 0 <= col < 9 and 0 <= value <= 9:
                            recognized_grid[row][col] = value
                            # Update display
                            y1 = row * (cell_size + gap)
                            x1 = col * (cell_size + gap)
                            # Clear existing digit
                            cv2.rectangle(display_grid, 
                                        (x1+10, y1+10), 
                                        (x1+cell_size-10, y1+cell_size-10), 
                                        (255, 255, 255), 
                                        -1)
                            if value != 0:
                                cv2.putText(display_grid, str(value), 
                                          (x1+15, y1+45), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 
                                          1, (0, 0, 255), 2)  # Modified digit displayed in red
                            
                            cv2.imshow('Sudoku Recognition', display_grid)
                            cv2.waitKey(1)  # Ensure window update
                            
                            print("\nCurrent Sudoku:")
                            self.print_grid(recognized_grid)
                        else:
                            print("Input value out of range!")
                    except ValueError:
                        print("Please enter a valid digit!")
                else:
                    print("Please enter y or n")
            
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Ensure window completely closed
            
            # Build Sudoku string
            sudoku_str = ""
            for i in range(9):
                if i > 0:
                    sudoku_str += "/"
                for j in range(9):
                    sudoku_str += str(recognized_grid[i][j])
            
            return sudoku_str
            
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            print("\nProgram interrupted by user")
            raise
        except Exception as e:
            cv2.destroyAllWindows()
            print(f"\nError processing image: {str(e)}")
            raise

    def print_grid(self, grid):
        """Print Sudoku grid"""
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("-" * 25)
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    print("|", end=" ")
                print(grid[i][j], end=" ")
            print()

    # Feature analysis function
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