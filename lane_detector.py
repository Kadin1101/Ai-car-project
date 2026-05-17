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
        self.hsv_lower = np.array((0, 0, 170), dtype=np.uint8)
        self.hsv_upper = np.array((180, 42, 255), dtype=np.uint8)

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

    def _get_mass_center(self, img_half, offset_x):
        M = cv2.moments(img_half)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"]) + offset_x
            cy = int(M["m01"] / M["m00"])
            return cx, cy
        return None

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

        # 1. 色彩過濾
        hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)

        # 2. 透視轉換
        bev_img = cv2.warpPerspective(thresh, self.M, (self.width, self.height), flags=cv2.INTER_LINEAR)

        # 3. 4塊分割尋找質心
        mid_x = self.width // 2
        num_slices = 4
        slice_h = self.height // num_slices
        
        left_pts = []
        right_pts = []
        all_pts = []
        
        slices_with_two = 0  # 記錄有幾個切塊「同時包含左右兩個點」
        
        for i in range(num_slices):
            y_end   = self.height - i * slice_h
            y_start = self.height - (i + 1) * slice_h
            
            left_half  = bev_img[y_start:y_end, :mid_x]
            right_half = bev_img[y_start:y_end, mid_x:]
            
            lc = self._get_mass_center(left_half, 0)
            rc = self._get_mass_center(right_half, mid_x)
            
            y_center = y_start + slice_h // 2
            
            if lc and rc:
                slices_with_two += 1
                
            if lc: 
                pt = (lc[0], y_center)
                left_pts.append(pt)
                all_pts.append(pt)
            
            if rc: 
                pt = (rc[0], y_center)
                right_pts.append(pt)
                all_pts.append(pt)

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
            # [模式 B] 完美的兩條線 (所有切塊都有 2 個點)
            m_L, c_L = fit_and_draw(left_pts, (255, 0, 0), "L")
            m_R, c_R = fit_and_draw(right_pts, (0, 0, 255), "R")
            
            if m_L is not None and m_R is not None:
                slope_diff = m_L + m_R
                SLOPE_THRESHOLD = 0.5
                if abs(slope_diff) > SLOPE_THRESHOLD:
                    error = slope_diff * 60
                    cv2.putText(display_frame, f"TURN (Slope:{slope_diff:.2f})", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                else:
                    error = c_R - self.target_right_x
                    cv2.putText(display_frame, "Track RIGHT (2 Lines)", (2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
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