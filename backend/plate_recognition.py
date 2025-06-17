import cv2
import numpy as np
import easyocr
from functools import lru_cache
import os
import time

class PlateRecognizer:
    _instance = None
    _reader = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PlateRecognizer, cls).__new__(cls)
            # 使用单例模式，确保只初始化一次 EasyOCR
            if cls._reader is None:
                print("初始化 EasyOCR...")
                try:
                    cls._reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)  # 如果有GPU则使用GPU
                    print("EasyOCR 初始化成功")
                except Exception as e:
                    print(f"EasyOCR 初始化失败: {str(e)}")
                    raise
        return cls._instance

    def __init__(self):
        # 定义车牌识别的一些参数
        self.min_area = 1000  # 最小车牌面积（降低以适应远距离车牌）
        self.max_area = 100000  # 最大车牌面积（增加以适应近距离车牌）
        self.min_ratio = 1.5  # 最小宽高比（降低以适应更多视角）
        self.max_ratio = 7.0  # 最大宽高比（增加以适应更多视角）
        
        # 多种颜色的HSV范围，适应不同类型的车牌
        self.color_ranges = [
            # 蓝色车牌
            (np.array([100, 70, 70]), np.array([140, 255, 255])),
            # 黄色车牌
            (np.array([15, 70, 70]), np.array([35, 255, 255])),
            # 绿色车牌（新能源）
            (np.array([35, 70, 70]), np.array([90, 255, 255])),
            # 白色车牌
            (np.array([0, 0, 200]), np.array([180, 30, 255])),
            # 黑色车牌
            (np.array([0, 0, 0]), np.array([180, 30, 70]))
        ]
        
        # 设置调试模式，保存中间处理图像
        self.debug = True
        self.debug_dir = "debug_images"
        if self.debug and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

    def preprocess_image(self, image_path):
        """
        增强的图像预处理，适应各种光线条件和车牌类型
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return None, None, "无法读取图片"

        # 保存原始图像
        original = image.copy()
        timestamp = int(time.time())
        
        # 调整图像大小以加快处理速度，但保持足够的细节
        height, width = image.shape[:2]
        max_dimension = 1280  # 增加分辨率
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)
            
        # 增强图像对比度
        enhanced = self.enhance_contrast(image)
        if self.debug:
            cv2.imwrite(f"{self.debug_dir}/{timestamp}_1_enhanced.jpg", enhanced)
            
        # 去噪
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        if self.debug:
            cv2.imwrite(f"{self.debug_dir}/{timestamp}_2_denoised.jpg", denoised)
            
        # 创建多个颜色掩码，适应不同类型的车牌
        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for i, (lower, upper) in enumerate(self.color_ranges):
            mask = cv2.inRange(hsv, lower, upper)
            if self.debug:
                cv2.imwrite(f"{self.debug_dir}/{timestamp}_3_mask_{i}.jpg", mask)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
            
        if self.debug:
            cv2.imwrite(f"{self.debug_dir}/{timestamp}_4_combined_mask.jpg", combined_mask)
            
        # 形态学操作，连接相近的区域
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # 闭操作连接字符
        morph = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
        # 开操作去除小噪点
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_open)
        
        if self.debug:
            cv2.imwrite(f"{self.debug_dir}/{timestamp}_5_morph.jpg", morph)
            
        # 边缘检测，增强轮廓特征
        edges = cv2.Canny(morph, 30, 200)
        if self.debug:
            cv2.imwrite(f"{self.debug_dir}/{timestamp}_6_edges.jpg", edges)
            
        # 再次进行形态学操作，连接边缘
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel_edge)
        
        if self.debug:
            cv2.imwrite(f"{self.debug_dir}/{timestamp}_7_dilated.jpg", edges)
            
        return edges, original, timestamp

    def enhance_contrast(self, image):
        """
        增强图像对比度
        """
        # 转换到LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # 分离通道
        l, a, b = cv2.split(lab)
        # 对亮度通道进行CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # 合并通道
        merged = cv2.merge((cl, a, b))
        # 转换回BGR
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return enhanced

    def find_plate_contours(self, binary, timestamp=None):
        """
        增强的车牌轮廓检测，使用多种方法提高准确率
        """
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选可能的车牌区域
        possible_plates = []
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                # 获取最小外接矩形
                rect = cv2.minAreaRect(contour)
                w, h = rect[1]
                if w == 0 or h == 0:
                    continue
                
                # 计算宽高比，考虑矩形可能的旋转
                ratio = max(w, h) / min(w, h)
                if self.min_ratio <= ratio <= self.max_ratio:
                    # 将候选区域转换为四边形坐标
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    # 计算矩形的周长
                    peri = cv2.arcLength(contour, True)
                    # 近似多边形
                    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                    
                    # 如果是四边形，更可能是车牌
                    if len(approx) >= 4 and len(approx) <= 8:
                        possible_plates.append(box)
                        if self.debug and timestamp:
                            # 创建一个临时图像显示找到的轮廓
                            temp_img = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
                            cv2.drawContours(temp_img, [box], 0, (0, 255, 0), 2)
                            cv2.imwrite(f"{self.debug_dir}/{timestamp}_8_contour_{idx}.jpg", temp_img)
        
        # 按面积排序，优先处理较大的区域
        possible_plates.sort(key=lambda box: cv2.contourArea(np.array([box])), reverse=True)
        return possible_plates

    def perspective_transform(self, image, points, timestamp=None, idx=0):
        """
        增强的透视变换，更好地处理倾斜的车牌
        """
        try:
            # 获取矩形四个角点，按照左上、右上、右下、左下排序
            rect = np.zeros((4, 2), dtype="float32")
            
            # 计算左上、右下
            s = points.sum(axis=1)
            rect[0] = points[np.argmin(s)]
            rect[2] = points[np.argmax(s)]
            
            # 计算右上、左下
            diff = np.diff(points, axis=1)
            rect[1] = points[np.argmin(diff)]
            rect[3] = points[np.argmax(diff)]
            
            # 计算新图像的宽度和高度
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            # 确保宽度大于高度（车牌通常是横向的）
            if maxHeight > maxWidth:
                maxWidth, maxHeight = maxHeight, maxWidth
                # 重新排序角点
                rect = np.array([rect[3], rect[0], rect[1], rect[2]], dtype="float32")
            
            # 变换后的坐标
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
            
            # 计算变换矩阵
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            
            if self.debug and timestamp:
                cv2.imwrite(f"{self.debug_dir}/{timestamp}_9_warped_{idx}.jpg", warped)
            
            return warped
        except Exception as e:
            print(f"透视变换失败: {str(e)}")
            # 如果透视变换失败，返回原始图像区域的矩形裁剪
            x, y, w, h = cv2.boundingRect(points)
            cropped = image[y:y+h, x:x+w]
            return cropped

    def enhance_plate_image(self, plate_img, timestamp=None, idx=0):
        """
        增强车牌图像质量
        """
        # 调整大小，确保车牌区域足够大
        h, w = plate_img.shape[:2]
        # 确保车牌图像大小适中，不要过大或过小
        target_width = 400
        scale = target_width / w
        plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale)
        
        if self.debug and timestamp:
            cv2.imwrite(f"{self.debug_dir}/{timestamp}_10_resized_{idx}.jpg", plate_img)
        
        # 转换为灰度图
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        if self.debug and timestamp:
            cv2.imwrite(f"{self.debug_dir}/{timestamp}_11_gray_{idx}.jpg", gray)
        
        # 自适应二值化
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        if self.debug and timestamp:
            cv2.imwrite(f"{self.debug_dir}/{timestamp}_12_binary_{idx}.jpg", binary)
        
        # 反转二值图像，确保字符为黑色背景为白色
        binary = cv2.bitwise_not(binary)
        
        if self.debug and timestamp:
            cv2.imwrite(f"{self.debug_dir}/{timestamp}_13_inverted_{idx}.jpg", binary)
        
        return plate_img, gray, binary

    def recognize_plate(self, image_path):
        """
        增强的车牌识别，多种方法相结合
        """
        try:
            print(f"开始处理图片: {image_path}")
            # 预处理图像
            binary, original, timestamp = self.preprocess_image(image_path)
            if binary is None:
                print("图片预处理失败")
                return None, "无法处理图片"

            print("图片预处理成功")
            
            # 多尝试几种识别方法，提高成功率
            result = self.try_multiple_recognition_methods(original, binary, timestamp)
            if result:
                return result, "识别成功"
                
            print("未能识别出有效车牌")
            return None, "未能识别车牌号"
            
        except Exception as e:
            print(f"识别过程错误: {str(e)}")
            return None, f"识别过程出错: {str(e)}"
    
    def try_multiple_recognition_methods(self, original, binary, timestamp):
        """
        尝试多种识别方法，按照成功可能性顺序
        """
        # 方法1: 直接识别原始图像
        result = self.try_direct_recognition(original, timestamp)
        if result:
            return result
            
        # 方法2: 定位车牌区域并识别
        result = self.try_plate_localization(original, binary, timestamp)
        if result:
            return result
            
        # 方法3: 对原始图像进行多尺度预处理后识别
        result = self.try_multiscale_recognition(original, timestamp)
        if result:
            return result
            
        # 方法4: 尝试降低验证标准
        result = self.try_lowered_validation(original, timestamp)
        if result:
            return result
            
        return None
    
    def try_direct_recognition(self, original, timestamp):
        """
        直接识别原始图像
        """
        try:
            print("方法1: 尝试直接识别原始图像")
            results = self._reader.readtext(original, 
                                         batch_size=1,
                                         paragraph=False,
                                         detail=1,
                                         allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼')
            
            for (bbox, text, confidence) in results:
                text = ''.join(text.split())
                print(f"原始图像识别文本: {text}, 置信度: {confidence}")
                if self.validate_plate_number(text) and confidence > 0.3:
                    print(f"在原始图像中找到有效车牌: {text}, 置信度: {confidence}")
                    return text
        except Exception as e:
            print(f"直接识别失败: {str(e)}")
        return None
    
    def try_plate_localization(self, original, binary, timestamp):
        """
        定位车牌区域并识别
        """
        try:
            print("方法2: 尝试定位车牌区域并识别")
            possible_plates = self.find_plate_contours(binary, timestamp)
            print(f"找到 {len(possible_plates)} 个可能的车牌区域")
            
            if not possible_plates:
                print("未检测到车牌区域")
                return None

            # 对每个可能的车牌区域进行处理
            for i, plate_points in enumerate(possible_plates):
                try:
                    # 透视变换矫正车牌
                    plate_img = self.perspective_transform(original, plate_points, timestamp, i)
                    
                    # 增强车牌图像
                    plate_img, gray, binary = self.enhance_plate_image(plate_img, timestamp, i)
                    
                    # 尝试在增强后的三种图像上识别
                    for j, img in enumerate([plate_img, gray, binary]):
                        try:
                            results = self._reader.readtext(img,
                                                         batch_size=1,
                                                         paragraph=False,
                                                         detail=1,
                                                         allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼')
                            
                            for (bbox, text, confidence) in results:
                                text = ''.join(text.split())
                                print(f"区域{i}图像{j}识别文本: {text}, 置信度: {confidence}")
                                if self.validate_plate_number(text) and confidence > 0.3:
                                    print(f"找到有效车牌: {text}, 置信度: {confidence}")
                                    return text
                        except Exception as e:
                            print(f"识别区域{i}图像{j}时出错: {str(e)}")
                            continue
                            
                except Exception as e:
                    print(f"处理第 {i+1} 个区域时出错: {str(e)}")
                    continue
        except Exception as e:
            print(f"车牌定位识别失败: {str(e)}")
        return None
    
    def try_multiscale_recognition(self, original, timestamp):
        """
        对原始图像进行多尺度预处理后识别
        """
        try:
            print("方法3: 尝试多尺度识别")
            # 多尺度处理
            scales = [0.5, 1.0, 1.5, 2.0]
            
            for i, scale in enumerate(scales):
                scaled = cv2.resize(original, None, fx=scale, fy=scale)
                
                if self.debug:
                    cv2.imwrite(f"{self.debug_dir}/{timestamp}_14_scaled_{i}.jpg", scaled)
                
                # 在缩放后的图像上直接识别
                try:
                    results = self._reader.readtext(scaled,
                                                 batch_size=1,
                                                 paragraph=False,
                                                 detail=1,
                                                 allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼')
                    
                    for (bbox, text, confidence) in results:
                        text = ''.join(text.split())
                        print(f"缩放{scale}倍识别文本: {text}, 置信度: {confidence}")
                        if self.validate_plate_number(text) and confidence > 0.3:
                            print(f"找到有效车牌: {text}, 置信度: {confidence}")
                            return text
                except Exception as e:
                    print(f"缩放{scale}倍识别失败: {str(e)}")
                    continue
        except Exception as e:
            print(f"多尺度识别失败: {str(e)}")
        return None
    
    def try_lowered_validation(self, original, timestamp):
        """
        尝试降低验证标准
        """
        try:
            print("方法4: 尝试降低验证标准")
            results = self._reader.readtext(original, 
                                         batch_size=1,
                                         paragraph=False,
                                         detail=1,
                                         allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼')
            
            # 收集所有可能是车牌的文本
            possible_plates = []
            for (bbox, text, confidence) in results:
                text = ''.join(text.split())
                print(f"潜在车牌文本: {text}, 置信度: {confidence}")
                
                # 宽松验证：只要有省份字符开头，且长度合适就考虑
                provinces = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
                if len(text) >= 5 and text[0] in provinces:
                    possible_plates.append((text, confidence))
            
            # 按置信度排序
            possible_plates.sort(key=lambda x: x[1], reverse=True)
            
            if possible_plates:
                best_plate = possible_plates[0][0]
                print(f"使用宽松验证找到车牌: {best_plate}")
                return best_plate
        except Exception as e:
            print(f"降低标准识别失败: {str(e)}")
        return None

    def validate_plate_number(self, text):
        """验证车牌号格式，采用灵活的验证规则"""
        # 移除空格和特殊字符
        text = ''.join(text.split())
        
        # 中国车牌格式验证
        provinces = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
        
        # 基本检查
        if len(text) < 6:  # 最短车牌长度
            return False
        
        if text[0] not in provinces:
            return False
        
        # 检查剩余字符是否为字母或数字
        for char in text[1:]:
            if not (char.isalnum()):
                return False
        
        # 新能源车牌检查
        if len(text) == 8:
            # 新能源车牌末尾必须是数字
            if not text[-1].isdigit():
                return False
        
        # 常规车牌检查
        elif len(text) == 7:
            # 第二位通常是字母
            if not text[1].isalpha():
                # 允许少数例外，如武警车牌
                pass
        
        return True 