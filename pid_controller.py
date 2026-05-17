# pid_controller.py

# -*- coding: utf-8 -*-

import time



class PIDController:



    def __init__(self, kp=0.35, ki=0.0, kd=0.18):

        self.kp = kp

        self.ki = ki

        self.kd = kd

        

        self.prev_error = 0

        self.integral = 0

        self.prev_time = time.time()

        

    def compute(self, error, deadband=15):

        # 加入死區 (Deadband) 機制：調整直線前進的容忍區間
        # 如果誤差小於 deadband，就視為 0，避免在直線上不斷微調導致車頭左右扭動
        if abs(error) < deadband:
            error = 0

        current_time = time.time()
        dt = current_time - self.prev_time

        

        # 避免除以零的狀況

        if dt <= 0.0:

            dt = 0.01



        # 1. 比例項 (P): 根據當前誤差立即修正

        p_term = self.kp * error

        

        # 2. 積分項 (I): 累積過去的誤差，解決穩態誤差

        self.integral += error * dt

        # 限制積分項的大小 (Anti-windup)，防止偏差累積過大導致失控

        self.integral = max(min(self.integral, 1000), -1000)

        i_term = self.ki * self.integral

        

        # 3. 微分項 (D): 預測未來的誤差變化，減少震盪

        derivative = (error - self.prev_error) / dt

        d_term = self.kd * derivative

        

        # 計算總輸出

        steering = p_term + i_term + d_term

        

        # 更新狀態供下次計算使用

        self.prev_error = error

        self.prev_time = current_time

        

        return steering



    def update_params(self, kp, ki, kd):

        self.kp = float(kp)

        self.ki = float(ki)

        self.kd = float(kd)

        # 參數改變時重置積分，避免歷史數據干擾
        
        self.integral = 0