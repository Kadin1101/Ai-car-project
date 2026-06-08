# -*- coding: utf-8 -*-

import math
import time
import threading
import smbus
from gpiozero import LED

FORWARD = 'forward'
BACKWARD = 'backward'

class PCA9685:
    __MODE1 = 0x00
    __PRESCALE = 0xFE
    __LED0_ON_L = 0x06

    def __init__(self, address=0x40, debug=False):
        self.bus = smbus.SMBus(1)
        self.address = address
        self.debug = debug
        self.lock = threading.Lock()
        self.write(self.__MODE1, 0x00)

    def write(self, reg, value):
        with self.lock:
            self.bus.write_byte_data(self.address, reg, value)

    def read(self, reg):
        with self.lock:
            return self.bus.read_byte_data(self.address, reg)

    def setPWMFreq(self, freq):
        prescaleval = 25000000.0 / 4096.0 / float(freq) - 1.0
        prescale = math.floor(prescaleval + 0.5)
        oldmode = self.read(self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10
        self.write(self.__MODE1, newmode)
        self.write(self.__PRESCALE, int(prescale))
        self.write(self.__MODE1, oldmode)
        time.sleep(0.005)
        self.write(self.__MODE1, oldmode | 0x80)

    def setPWM(self, channel, on, off):
        self.write(self.__LED0_ON_L + 4 * channel, on & 0xFF)
        self.write(self.__LED0_ON_L + 4 * channel + 1, on >> 8)
        self.write(self.__LED0_ON_L + 4 * channel + 2, off & 0xFF)
        self.write(self.__LED0_ON_L + 4 * channel + 3, off >> 8)

    def setDutycycle(self, channel, percent):
        percent = max(0, min(100, int(percent)))
        pulse = int(percent * 4095 / 100.0)
        self.setPWM(channel, 0, pulse)

    def setLevel(self, channel, value):
        self.setPWM(channel, 0, 4095 if value else 0)


class MotorController:
    # ── 硬體校準參數 ──
    # 如果車子走直線時偏向一邊，可調低較快那一側的 Scale 值 (0.0 ~ 1.0)
    LEFT_MOTOR_SCALE = 1.00
    RIGHT_MOTOR_SCALE = 1.00
    
    # 如果特定一側馬達接線相反，可將 False 改為 True
    LEFT_MOTOR_INVERT = False
    RIGHT_MOTOR_INVERT = False

    # 轉向設定 (可用 'arc' 圓弧過彎 或 'spin' 原地打轉)
    TURN_STYLE = "arc"
    TURN_PWM = 58
    TURN_INNER_RATIO = 0.25

    def __init__(self):
        # 腳位定義保持與原硬體一致
        self.PWMA, self.AIN1, self.AIN2 = 0, 2, 1     # 左前輪
        self.PWMB, self.BIN1, self.BIN2 = 5, 3, 4     # 右前輪
        self.PWMC, self.CIN1, self.CIN2 = 6, 8, 7     # 左後輪
        self.PWMD, self.DIN1, self.DIN2 = 11, 25, 24  # 右後輪

        # 初始化 I2C 與 GPIO
        self.pwm = PCA9685(0x40, debug=False)
        self.pwm.setPWMFreq(50)
        self.motorD1 = LED(self.DIN1)
        self.motorD2 = LED(self.DIN2)
        
        # 狀態追蹤
        self.is_stopped = True  # 紀錄車輛目前是否處於完全靜止狀態
        self.last_left_speed = 0
        self.last_right_speed = 0
        self.drive_lock = threading.Lock()
        self.full_stop()

    def _validate_speed(self, speed):
        return max(0, min(100, int(speed)))

    # ==========================================
    # 底層新版校準與映射邏輯
    # ==========================================
    
    @staticmethod
    def _calibrate_side(speed, invert_flag, scale):
        calibrated = int(round(speed * scale))
        if invert_flag:
            calibrated *= -1
        return max(-100, min(100, calibrated))

    def _set_motor(self, motor_idx, signed_speed):
        speed = abs(int(signed_speed))
        # 硬體防護：過濾掉低於靜摩擦力的 PWM 值，避免馬達堵轉導致大電流當機
        # (已恢復為 35，因為低於 35 物理上推不動，會導致無限堵轉當機)
        if 0 < speed < 35:
            speed = 0
            
        direction = "forward" if signed_speed >= 0 else "backward"

        if motor_idx == 0:   # A (左前)
            self.pwm.setDutycycle(self.PWMA, speed)
            self.pwm.setLevel(self.AIN1, 0 if direction == "forward" else 1)
            self.pwm.setLevel(self.AIN2, 1 if direction == "forward" else 0)
        elif motor_idx == 1: # B (右前)
            self.pwm.setDutycycle(self.PWMB, speed)
            self.pwm.setLevel(self.BIN1, 1 if direction == "forward" else 0)
            self.pwm.setLevel(self.BIN2, 0 if direction == "forward" else 1)
        elif motor_idx == 2: # C (左後)
            self.pwm.setDutycycle(self.PWMC, speed)
            self.pwm.setLevel(self.CIN1, 1 if direction == "forward" else 0)
            self.pwm.setLevel(self.CIN2, 0 if direction == "forward" else 1)
        elif motor_idx == 3: # D (右後)
            self.pwm.setDutycycle(self.PWMD, speed)
            if direction == "forward":
                self.motorD1.off()
                self.motorD2.on()
            else:
                self.motorD1.on()
                self.motorD2.off()

    def _apply_drive(self, left_speed, right_speed):
        """將左右轉速經過校準後，統一輸出至四顆馬達"""
        
        with self.drive_lock:
            # Slew Rate Limiter (防暴衝/防瞬間反轉電流過載)
            # 全域硬體防護：無論是手動還是自動，都限制單次呼叫的電壓變化量
            max_change = 25
            if left_speed - self.last_left_speed > max_change:
                left_speed = self.last_left_speed + max_change
            elif left_speed - self.last_left_speed < -max_change:
                left_speed = self.last_left_speed - max_change
                
            if right_speed - self.last_right_speed > max_change:
                right_speed = self.last_right_speed + max_change
            elif right_speed - self.last_right_speed < -max_change:
                right_speed = self.last_right_speed - max_change

            calibrated_left = self._calibrate_side(left_speed, self.LEFT_MOTOR_INVERT, self.LEFT_MOTOR_SCALE)
            calibrated_right = self._calibrate_side(right_speed, self.RIGHT_MOTOR_INVERT, self.RIGHT_MOTOR_SCALE)
            
            # 左側 (A, C)
            self._set_motor(0, calibrated_left)
            self._set_motor(2, calibrated_left)
            # 右側 (B, D)
            self._set_motor(1, calibrated_right)
            self._set_motor(3, calibrated_right)
            
            self.last_left_speed = left_speed
            self.last_right_speed = right_speed

    def _turn(self, direction, requested_speed):
        turn_pwm = max(self.TURN_PWM, int(requested_speed))
        inner = int(round(turn_pwm * self.TURN_INNER_RATIO))
        if self.TURN_STYLE == "spin":
            if direction == "left":
                self._apply_drive(-turn_pwm, turn_pwm)
            else:
                self._apply_drive(turn_pwm, -turn_pwm)
            return

        # Arc mode
        if direction == "left":
            self._apply_drive(inner, turn_pwm)
        else:
            self._apply_drive(turn_pwm, inner)

    # ==========================================
    # 車體運動基礎指令
    # ==========================================
    
    def move_forward(self, speed=50):
        self._apply_drive(speed, speed)

    def move_backward(self, speed=50):
        self._apply_drive(-speed, -speed)

    def move_left(self, speed=50):
        # 麥克納姆輪平移 (如果有支援的話，原本的邏輯保留)
        # 左平移：左前退、右前進、左後進、右後退
        speed = self._validate_speed(speed)
        self._set_motor(0, -speed)
        self._set_motor(1, speed)
        self._set_motor(2, speed)
        self._set_motor(3, -speed)

    def move_right(self, speed=50):
        # 右平移：左前進、右前退、左後退、右後進
        speed = self._validate_speed(speed)
        self._set_motor(0, speed)
        self._set_motor(1, -speed)
        self._set_motor(2, -speed)
        self._set_motor(3, speed)

    def stop(self):
        with self.drive_lock:
            for pwm_channel in [self.PWMA, self.PWMB, self.PWMC, self.PWMD]:
                self.pwm.setDutycycle(pwm_channel, 0)
            self.motorD1.off()
            self.motorD2.off()
            self.is_stopped = True # 標記為靜止，下次啟動就會觸發緩起步
            self.last_left_speed = 0
            self.last_right_speed = 0

    def full_stop(self):
        self.stop()

    def direct(self, action, speed):
        if speed == 0 or action == 'stop':
            self.full_stop()
            return
    
        if action == 'forward': self.move_forward(speed)
        elif action == 'backward': self.move_backward(speed)
        elif action == 'left': self._turn("left", speed)   # 改用平滑過彎邏輯
        elif action == 'right': self._turn("right", speed) # 改用平滑過彎邏輯
        elif action == 'moveLeft': self.move_left(speed)
        elif action == 'moveRight': self.move_right(speed)

    # ==========================================
    # 支援緩啟動的上層封裝 (Web API 與自駕呼叫此處)
    # ==========================================

    def soft_move(self, action, speed):
        speed = self._validate_speed(speed)
        self.direct(action, speed)
        self.is_stopped = False

    def steer(self, base_speed, steering_adjustment):
        # 將誤差加入並將兩側速度計算出來
        left_speed = base_speed + steering_adjustment
        right_speed = base_speed - steering_adjustment
        
        self._apply_drive(left_speed, right_speed)
        self.is_stopped = (left_speed == 0 and right_speed == 0)

    # ==========================================
    # 攝影機雲台/舵機控制與清理
    # ==========================================

    def set_camera_angle(self, channel, angle):
        angle = max(0, min(180, angle)) 
        pulse_width_us = (angle * 11) + 500
        duty_cycle = int(4096 * pulse_width_us / 20000)
        self.pwm.setPWM(channel, 0, duty_cycle)
        
    def cleanup(self):
        self.full_stop()
        self.pwm.setPWM(9, 0, 0)
        self.pwm.setPWM(10, 0, 0)
        self.motorD1.close()
        self.motorD2.close()
