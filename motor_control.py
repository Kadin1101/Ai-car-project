# -*- coding: utf-8 -*-

import math
import time
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
        self.write(self.__MODE1, 0x00)

    def write(self, reg, value):
        self.bus.write_byte_data(self.address, reg, value)



    def read(self, reg):
        return self.bus.read_byte_data(self.address, reg)



    def setPWMFreq(self, freq):
        prescaleval = 25000000.0 / 4096.0 / float(freq) - 1.0
        prescale = math.floor(prescaleval + 0.5)
        oldmode = self.read(self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10
        self.write(self.__MODE1, newmode)
        self.write(self.__PRESCALE, int(prescale))
        self.write(self.__MODE1, oldmode)
        self.write(self.__MODE1, oldmode | 0x80)



    def setPWM(self, channel, on, off):
        self.write(self.__LED0_ON_L + 4 * channel, on & 0xFF)
        self.write(self.__LED0_ON_L + 4 * channel + 1, on >> 8)
        self.write(self.__LED0_ON_L + 4 * channel + 2, off & 0xFF)
        self.write(self.__LED0_ON_L + 4 * channel + 3, off >> 8)



    def setDutycycle(self, channel, percent):
        pulse = int(percent * 4096 / 100)
        self.setPWM(channel, 0, pulse)



    def setLevel(self, channel, value):
        self.setPWM(channel, 0, 4095 if value else 0)





class MotorController:
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
        

        # --- 新增：狀態追蹤 ---
        self.is_stopped = True  # 紀錄車輛目前是否處於完全靜止狀態


    def _validate_speed(self, speed):
        return max(0, min(100, speed))



    def _set_motor_direction(self, in1, in2, forward):
        self.pwm.setLevel(in1, int(forward))
        self.pwm.setLevel(in2, int(not forward))



    def run_single_motor(self, motor, direction, speed):
        speed = self._validate_speed(speed)
        forward = (direction == FORWARD)



        if motor == 0:
            self.pwm.setDutycycle(self.PWMA, speed)
            self._set_motor_direction(self.AIN1, self.AIN2, not forward)

        elif motor == 1:

            self.pwm.setDutycycle(self.PWMB, speed)

            self._set_motor_direction(self.BIN1, self.BIN2, forward)

        elif motor == 2:

            self.pwm.setDutycycle(self.PWMC, speed)

            self._set_motor_direction(self.CIN1, self.CIN2, forward)

        elif motor == 3:

            self.pwm.setDutycycle(self.PWMD, speed)

            if forward:

                self.motorD1.off()

                self.motorD2.on()

            else:

                self.motorD1.on()

                self.motorD2.off()



    # ==========================================

    # 車體運動基礎指令 (底層不含緩啟動)

    # ==========================================

    def move_forward(self, speed=50):

        for m in range(4): self.run_single_motor(m, FORWARD, speed)



    def move_backward(self, speed=50):

        for m in range(4): self.run_single_motor(m, BACKWARD, speed)



    def move_left(self, speed=50):

        self.run_single_motor(0, BACKWARD, speed)

        self.run_single_motor(1, FORWARD, speed)

        self.run_single_motor(2, FORWARD, speed)

        self.run_single_motor(3, BACKWARD, speed)



    def move_right(self, speed=50):

        self.run_single_motor(0, FORWARD, speed)

        self.run_single_motor(1, BACKWARD, speed)

        self.run_single_motor(2, BACKWARD, speed)

        self.run_single_motor(3, FORWARD, speed)



    def turn_left(self, speed=50):

        self.run_single_motor(0, BACKWARD, speed)

        self.run_single_motor(1, FORWARD, speed)

        self.run_single_motor(2, BACKWARD, speed)

        self.run_single_motor(3, FORWARD, speed)



    def turn_right(self, speed=50):

        self.run_single_motor(0, FORWARD, speed)

        self.run_single_motor(1, BACKWARD, speed)

        self.run_single_motor(2, FORWARD, speed)

        self.run_single_motor(3, BACKWARD, speed)



    def stop(self):

        for pwm_channel in [self.PWMA, self.PWMB, self.PWMC, self.PWMD]:

            self.pwm.setDutycycle(pwm_channel, 0)

        self.motorD1.off()

        self.motorD2.off()

        self.is_stopped = True # 標記為靜止，下次啟動就會觸發緩起步



    def full_stop(self):

        self.stop()



    def direct(self, action, speed):

            if speed == 0 or action == 'stop':

                self.full_stop()

                return

        

            if action == 'forward': self.move_forward(speed)

            elif action == 'backward': self.move_backward(speed)

            elif action == 'left': self.turn_left(speed)

            elif action == 'right': self.turn_right(speed)

            elif action == 'moveLeft': self.move_left(speed)

            elif action == 'moveRight': self.move_right(speed)

    

    # ==========================================

    # 支援緩啟動的上層封裝 (Web API 與自駕呼叫此處)

    # ==========================================

    def soft_move(self, action, speed):



        speed = self._validate_speed(speed)

        

        # 如果是從靜止狀態開始啟動，執行 0.1 秒的快速緩啟動保護

        if self.is_stopped and action != 'stop' and speed > 0:
            steps = 10
            for i in range(1, steps + 1):
                cur_speed = int(speed * (i / steps))
                self.direct(action, cur_speed)
                time.sleep(0.05)

            self.is_stopped = False

        else:

            self.direct(action, speed)



    def drive_steer(self, left_speed, right_speed):

        left_speed = self._validate_speed(left_speed)

        right_speed = self._validate_speed(right_speed)



        if left_speed == 0 and right_speed == 0:

            self.full_stop()

            return



        self.run_single_motor(0, FORWARD, left_speed)

        self.run_single_motor(2, FORWARD, left_speed)

        self.run_single_motor(1, FORWARD, right_speed)

        self.run_single_motor(3, FORWARD, right_speed)



    def steer(self, base_speed, steering_adjustment):

        left_speed = base_speed + steering_adjustment

        right_speed = base_speed - steering_adjustment

        

        if self.is_stopped and base_speed > 0:
            # 偵測到從紅燈或丟線狀態重新起步，自動緩加速防當機
            steps = 10
            for i in range(1, steps + 1):
                cur_l = int(left_speed * (i / steps))
                cur_r = int(right_speed * (i / steps))
                self.drive_steer(cur_l, cur_r)
                time.sleep(0.05)

            self.is_stopped = False

        else:

            # 已經在行駛中，直接無縫調整轉向

            self.drive_steer(left_speed, right_speed)



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