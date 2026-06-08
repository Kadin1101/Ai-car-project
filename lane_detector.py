# -*- coding: utf-8 -*-

import cv2
import numpy as np

class LaneDetector:
    def __init__(self, width=320, height=240, 
                 hsv_lower=(0, 0, 200), hsv_upper=(180, 40, 255)):
        self.width  = width
        self.height = height

        # 擴大白光寬容度：降低亮度要求 (200 -> 150)，提高飽和度容忍 (40 -> 80)
        # 以應付相機自動曝光或自動白平衡造成的輕微色偏與變暗
        # 擴大白光寬容度：降低亮度要求以捕捉偏暗的白膠帶 (170 -> 130)
        # 並提高飽和度容忍 (42 -> 60)
        self.hsv_lower = np.array((0, 0, 130), dtype=np.uint8)
        self.hsv_upper = np.array((180, 60, 255), dtype=np.uint8)

        # 定義透視轉換的來源點 (原圖上的梯形 ROI)
        # 注意：這裡使用百分比或依比例推算比較安全。
        # 假設鏡頭看到地平線在畫面中上方，這裡擷取下半部
        src = np.float32([
            [int(self.width * 0), int(self.height * 0.55)], # 左上
            [int(self.width * 1), int(self.height * 0.55)], # 右上
            [self.width,            self.height],             # 右下
            [0,                     self.height]              # 左下
        ])

        # 定義透視轉換的目的地點 (鳥瞰圖的矩形)
        dst = np.float32([
            [0,          0],
            [self.width, 0],
            [self.width, self.height],
            [0,          self.height]
        ])

        # 計算透視轉換矩陣 (由前視圖 -> 鳥瞰圖)
        self.M = cv2.getPerspectiveTransform(src, dst)
        # 計算反向透視轉換矩陣 (由鳥瞰圖 -> 前視圖)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

        self.kernel = np.ones((5, 5), np.uint8)
        
        # 單線的「期望目標位置 (Target X)」，在鳥瞰圖空間中
        # 可依實際鳥瞰圖內車道線的位置進行微調
        self.target_left_x  = int(self.width * 0.1)
        self.target_right_x = int(self.width * 0.9)

    def _find_lane_points_tracked(self, slice_img, y_offset, expected_l_x, expected_r_x):
        # 尋找整個橫向切塊內的所有輪廓
        contours, _ = cv2.findContours(slice_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        l_pt = None
        r_pt = None
        min_l_dist = float('inf')
        min_r_dist = float('inf')
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < 15:
                continue
                
            rect = cv2.minAreaRect(c)
            thickness = min(rect[1][0], rect[1][1])
            
            # 膠帶在 320x240 畫面中大約只有 10~18 像素寬
            if thickness > 22:
                continue
                
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"]) + y_offset
                
                # 根據 X 座標分類為左線或右線，並挑選「距離預期位置最近」的
                if cx < self.width // 2:
                    dist = abs(cx - expected_l_x)
                    if dist < min_l_dist:
                        # 加入最大橫向偏移限制，防止突然跳去抓極遠處的斑馬線
                        if dist < 120:
                            min_l_dist = dist
                            l_pt = (cx, cy)
                else:
                    dist = abs(cx - expected_r_x)
                    if dist < min_r_dist:
                        if dist < 120:
                            min_r_dist = dist
                            r_pt = (cx, cy)
                        
        return l_pt, r_pt

    def _transform_point(self, pt, matrix):
        """將單一座標點套用矩陣轉換"""
        if pt is None: return None
        pt_np = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt_np, matrix)
        return int(mapped[0][0][0]), int(mapped[0][0][1])

    def process(self, frame):
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        # 1. 影像預處理與自適應二值化 (Adaptive Threshold)
        # 放棄極度不穩定的 HSV 色彩過濾，改用單純的「亮度對比」來抓白線
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # (A) 自適應二值化：區塊設為 41x41。只要像素比周圍 41x41 區域亮 15，就當作白線。
        # 這個演算法會把「真正細長的白膠帶」完美抓出來，同時把「大面積均勻的反光」中心挖成黑色！
        adapt_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, -15)
        
        # (B) 絕對亮度過濾：為了避免抓到暗處的木紋邊緣，設定像素亮度至少要 > 110
        _, basic_thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        
        # 兩者交集：夠亮，且比周圍還要亮
        thresh = cv2.bitwise_and(adapt_thresh, basic_thresh)
        
        # 使用閉運算 (CLOSE) 將散落的小白點（斷裂的膠帶）無縫連接起來
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)

        # 2. 透視轉換
        bev_img = cv2.warpPerspective(thresh, self.M, (self.width, self.height), flags=cv2.INTER_LINEAR)

        # 3. 4塊分割尋找質心 (由下往上動態追蹤)
        mid_x = self.width // 2
        num_slices = 4
        slice_h = self.height // num_slices
        
        left_pts = []
        right_pts = []
        all_pts = []
        
        slices_with_two = 0  # 記錄有幾個切塊「同時包含左右兩個點」
        
        # 初始的預期位置 (預設為最底層理想的車道線位置)
        expected_l_x = self.target_left_x
        expected_r_x = self.target_right_x
        
        # 由畫面最下方 (i=3) 往最上方 (i=0) 掃描
        for i in range(num_slices - 1, -1, -1):
            y1 = i * slice_h
            y2 = (i + 1) * slice_h
            slice_img = bev_img[y1:y2, :]
            
            l_center, r_center = self._find_lane_points_tracked(slice_img, y1, expected_l_x, expected_r_x)
            
            if l_center:
                left_pts.append(l_center)
                all_pts.append(l_center)
                # 更新下一層的預期位置為「這一層找到的 X 座標」
                expected_l_x = l_center[0]
                
            if r_center:
                right_pts.append(r_center)
                all_pts.append(r_center)
                # 更新下一層的預期位置
                expected_r_x = r_center[0]
                
            if l_center and r_center:
                slices_with_two += 1

        error = None
        display_frame = frame.copy()

        # 4. 直線擬合與作圖函數
        def fit_and_draw(pts, color, label):
            if len(pts) < 2:
                return None, None
            
            xs = np.array([p[0] for p in pts])
            ys = np.array([self.height - p[1] for p in pts]) # y_real: 0~240
            
            if max(ys) - min(ys) < 5:
                return None, None
                
            m, c = np.polyfit(ys, xs, 1)
            
            pt_bottom_bev = (int(c), self.height)
            pt_top_bev    = (int(m * self.height + c), 0)
            
            p1 = self._transform_point(pt_bottom_bev, self.Minv)
            p2 = self._transform_point(pt_top_bev, self.Minv)
            
            if p1 and p2:
                cv2.line(display_frame, p1, p2, color, 3)
                text_y = max(20, p1[1] - 10) if p1[1] > 20 else p2[1] + 20
                cv2.putText(display_frame, label, (p1[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
            for pt in pts:
                orig_pt = self._transform_point(pt, self.Minv)
                if orig_pt:
                    cv2.circle(display_frame, orig_pt, 5, color, -1) 
                    
            return m, c

        # 5. 狀態判定與轉向邏輯
        total_pts = len(all_pts)
        m_L, c_L = None, None
        m_R, c_R = None, None

        if total_pts <= 2:
            # [模式 A] 沒線 (點數太少)
            cv2.putText(display_frame, "Lost Lane (pts<=2)", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
        elif slices_with_two == 4:
            # [模式 B] 完美的兩條線
            m_L, c_L = fit_and_draw(left_pts, (255, 0, 0), "L")
            m_R, c_R = fit_and_draw(right_pts, (0, 0, 255), "R")
            
            if m_L is not None and m_R is not None:
                # 放棄容易受鳥瞰圖變形影響的斜率判斷 (slope_diff)
                # 改為使用「兩條線的底部中點」來鎖定畫面正中央
                lane_center = (c_L + c_R) / 2
                target_center = self.width / 2
                error = lane_center - target_center
                cv2.putText(display_frame, "Track CENTER (2 Lines)", (2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
        elif slices_with_two == 0:
            # [模式 C] 絕對單線 (沒有任何切塊包含 2 個點，代表這完全是一條跨界的線)
            # 把全部的點合併成一條線擬合
            m, c = fit_and_draw(all_pts, (0, 255, 0), "Single")
            if m is not None and c is not None:
                if c < self.width // 2:
                    error = c - self.target_left_x
                    cv2.putText(display_frame, "Track LEFT (Merged)", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    m_L = m
                else:
                    error = c - self.target_right_x
                    cv2.putText(display_frame, "Track RIGHT (Merged)", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    m_R = m
            else:
                 cv2.putText(display_frame, "Lost Lane (Fit failed)", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                 
        else:
            # [模式 D] 殘缺的兩條線 (1 <= slices_with_two <= 3)
            # 不能混在一起擬合，所以我們挑選「點數較多」的那條線作為主導線
            if len(right_pts) >= len(left_pts):
                m_R, c_R = fit_and_draw(right_pts, (0, 0, 255), "R (Dom)")
                if c_R is not None:
                    error = c_R - self.target_right_x
                    cv2.putText(display_frame, "Track RIGHT (Dom)", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                m_L, c_L = fit_and_draw(left_pts, (255, 0, 0), "L (Dom)")
                if c_L is not None:
                    error = c_L - self.target_left_x
                    cv2.putText(display_frame, "Track LEFT (Dom)", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # 顯示當前誤差值與斜率資訊
        if error is not None:
            cv2.putText(display_frame, f"E:{int(error)}", (2, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        info_text = ""
        if m_L is not None: info_text += f"mL:{m_L:.2f} "
        if m_R is not None: info_text += f"mR:{m_R:.2f}"
        if info_text:
            cv2.putText(display_frame, info_text, (self.width - 150, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        return display_frame, error, bev_img
