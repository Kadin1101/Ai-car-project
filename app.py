# -*- coding: utf-8 -*-
"""
app.py — AI 自走車後端 (包含黑白影像除錯功能)
"""
import cv2
import time
import threading
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO

# ★ 修正了 Import 錯誤，只載入需要的 MotorController
from motor_control import MotorController
from pid_controller import PIDController
from lane_detector import LaneDetector

app = Flask(__name__)

# ★ 關閉 Flask (Werkzeug) 的預設 HTTP 請求日誌，讓終端機不再被洗版
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ── 鎖 ────────────────────────────────────────────────────────────────
i2c_lock    = threading.Lock()
jpeg_lock   = threading.Lock()
mask_lock   = threading.Lock()  # ★ 保護黑白除錯影像的鎖
frame_lock  = threading.Lock()
status_lock = threading.Lock()
manual_lock = threading.Lock()

# ── 共享緩衝 ──────────────────────────────────────────────────────────
output_jpeg       = None   # 主畫面
output_mask_jpeg  = None   # ★ 黑白除錯畫面
yolo_input_frame  = None   
last_manual_time  = 0.0    
current_manual_action = 'stop'  # ★ 手動遙控狀態指令

sys_status = {
    "auto_mode":    False,
    "base_speed":   20,
    "error":        0,
    "steering":     0.0,
    "yolo_objects": [],
    "red_light":    False,
}

# ── 硬體初始化 ────────────────────────────────────────────────────────
try:
    motor = MotorController()
    print("✅ 馬達差速控制器初始化成功")
except Exception as e:
    print(f"❌ 硬體初始化失敗: {e}")
    motor = None

# YOLO 模型初始化 (請確認檔案路徑與存在)
# 若沒有 weights.pt 也可以改用 yolov8n.pt 測試
try:
    yolo_model = YOLO("weights.pt")
except Exception as e:
    print(f"❌ 硬體初始化失敗: {e}")
    motor = None

pid      = PIDController(kp=0.35, ki=0.0, kd=0.18)
detector = LaneDetector(width=320, height=240)

# 1. 嘗試載入樹莓派專屬相機
try:
    from picamera2 import Picamera2
    IS_RPI = True
except ImportError:
    IS_RPI = False
    print("⚠️ 偵測到非樹莓派環境，將啟用 OpenCV Webcam 測試模式")

# 2. 建立取得相機的工廠函數
def get_camera():
    if IS_RPI:
        picam = Picamera2()
        picam.configure(picam.create_video_configuration(main={'format': 'RGB888', 'size': (320, 240)}))
        picam.start()
        return picam
    else:
        # 在筆電上，直接開啟 USB Webcam (通常是 0)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        return cap

# 3. 擷取畫面的時候要統一格式
def capture_frame(camera):
    if IS_RPI:
        return camera.capture_array()
    else:
        ret, frame = camera.read()
        return frame if ret else None

# 初始化全局相機物件
camera = get_camera()


# ══════════════════════════════════════════════════════════════════════
# YOLO 背景執行緒
# ══════════════════════════════════════════════════════════════════════
def yolo_worker():
    global yolo_input_frame
    
    # 根據 data.yaml，0 是 green_light，1 是 red_light
    RED_LIGHT_CLASS_ID = 1

    while True:
        with frame_lock:
            frame = yolo_input_frame
            yolo_input_frame = None

        if frame is not None:
            try:
                results = yolo_model.predict(frame, imgsz=320, conf=0.2, verbose=False)
                is_red_light = False
                detected_labels = []
                detected_boxes = []

                for r in results:
                    for box in r.boxes:
                        class_id = int(box.cls[0].item())
                        # ★ 修正: 這裡改用 class_id 來取得標籤名稱
                        class_name = yolo_model.names[class_id] 
                        detected_labels.append(class_name)
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detected_boxes.append((class_name, int(x1), int(y1), int(x2), int(y2)))

                        if class_id == RED_LIGHT_CLASS_ID:
                            is_red_light = True
                
                with status_lock:
                    sys_status["red_light"]    = is_red_light
                    sys_status["yolo_objects"] = detected_labels
                    sys_status["yolo_boxes"]   = detected_boxes
            except Exception as e:
                print(f"[YOLO] 推論錯誤: {e}")

        time.sleep(0.25)

# ★ 在函數最外面啟動 YOLO 執行緒
threading.Thread(target=yolo_worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════
# 主捕獲 + 自動駕駛執行緒 (防崩潰除錯版)
# ══════════════════════════════════════════════════════════════════════
def capture_worker():
    global output_jpeg, output_mask_jpeg, yolo_input_frame, last_manual_time
    last_valid_steering = 0
    last_valid_time     = 0
    last_print_state    = "unknown"  # ★ 用來記錄前一次列印的狀態，避免重複洗版

    while True:
        try:
            # 1. 抓取畫面
            try:
                frame = capture_frame(camera)
            except Exception as e:
                print(f"[Camera] 擷取失敗: {e}")
                time.sleep(0.05)
                continue
            
            if frame is None:
                continue

            frame = cv2.flip(frame, -1)

            # 2. 車道偵測
            display_frame, error, mask_img = detector.process(frame)

            # 3. 更新狀態
            with status_lock:
                sys_status["error"] = error if error is not None else 0
                red_light  = sys_status["red_light"]
                auto_mode  = sys_status["auto_mode"]
                base_speed = sys_status["base_speed"]

            with frame_lock:
                yolo_input_frame = display_frame.copy()

            if red_light:
                cv2.putText(display_frame, "RED LIGHT",
                            (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # 4. 自動駕駛控制
            if auto_mode and motor:
                try:
                    with i2c_lock:
                        if red_light:
                            motor.full_stop()
                            if last_print_state != "red_light_stop":
                                print("\n🛑 偵測到紅燈，車輛停止。")
                                last_print_state = "red_light_stop"
                        elif error is not None:
                            steering = pid.compute(error)
                            motor.steer(base_speed, steering)
                            last_valid_steering = steering
                            last_valid_time     = time.time()
                            if last_print_state != "auto_moving":
                                print("\n🟢 開始沿車道行駛。")
                                last_print_state = "auto_moving"
                        else:
                            elapsed = time.time() - last_valid_time
                            if elapsed < 0.4:
                                motor.steer(base_speed, last_valid_steering)
                                cv2.putText(display_frame, "Memory",
                                            (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)
                            else:
                                motor.full_stop()
                                cv2.putText(display_frame, "Lost Lane",
                                            (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                                if last_print_state != "lost_lane_stop":
                                    print("\n⚠️ 失去車道線，車輛停止。")
                                    last_print_state = "lost_lane_stop"

                    if error is not None:
                        with status_lock:
                            sys_status["steering"] = round(steering, 2)

                except OSError as e:
                    print(f"[I2C] 寫入失敗: {e}")

            # 5. 壓縮與推流
            with status_lock:
                boxes = sys_status.get("yolo_boxes", [])
            
            # 將單通道黑白遮罩轉為彩色，才能畫彩色框
            mask_img_bgr = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
            
            # 在原圖與 DEBUG 圖上繪製 YOLO 框
            for (c_name, x1, y1, x2, y2) in boxes:
                color = (0, 0, 255) if c_name == 'red_light' else (0, 255, 0)
                # 畫在即時畫面
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, c_name, (x1, max(10, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                # 畫在 DEBUG 畫面
                cv2.rectangle(mask_img_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(mask_img_bgr, c_name, (x1, max(10, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            ok, buf = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with jpeg_lock:
                    output_jpeg = buf.tobytes()

            ok_m, buf_m = cv2.imencode('.jpg', mask_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok_m:
                with mask_lock:
                    output_mask_jpeg = buf_m.tobytes()

            time.sleep(0.25)

        except Exception as e:
            import traceback
            print(f"\n❌ [影像執行緒發生錯誤] 說明: {e}")
            traceback.print_exc()
            time.sleep(1)

# ★ 在函數最外面啟動相機執行緒
threading.Thread(target=capture_worker, daemon=True).start()

# ══════════════════════════════════════════════════════════════════════
# 馬達手動控制執行緒 (高頻率 20Hz，專心處理硬體指令)
# ══════════════════════════════════════════════════════════════════════
def motor_worker():
    global current_manual_action
    last_print_state_manual = "unknown"
    
    while True:
        try:
            with status_lock:
                auto_mode = sys_status["auto_mode"]
                base_speed = sys_status["base_speed"]
            
            if not motor or auto_mode:
                time.sleep(0.05)
                continue

            # ── 手動模式 ──
            # 檢查心跳是否超時 (大於 0.3 秒沒收到前端請求，強制煞停)
            if time.time() - last_manual_time > 0.3:
                if not motor.is_stopped:
                    try:
                        with i2c_lock:
                            motor.full_stop()
                            current_manual_action = 'stop'
                    except OSError:
                        pass
                
                if last_print_state_manual != "manual_stop":
                    print("\n✋ 手動模式待命 (車輛停止)。")
                    last_print_state_manual = "manual_stop"
            else:
                # 依據最新狀態指令動作
                try:
                    with i2c_lock:
                        if current_manual_action == 'stop':
                            if not motor.is_stopped:
                                motor.full_stop()
                        else:
                            motor.soft_move(current_manual_action, base_speed)
                except OSError:
                    pass
                
                if last_print_state_manual != "manual_moving":
                    print(f"\n🕹️ 手動駕駛中 ({current_manual_action})...")
                    last_print_state_manual = "manual_moving"

            time.sleep(0.05)

        except Exception as e:
            time.sleep(0.5)

threading.Thread(target=motor_worker, daemon=True).start()

# ══════════════════════════════════════════════════════════════════════
# Flask 串流函式
# ══════════════════════════════════════════════════════════════════════
def generate_frames():
    while True:
        with jpeg_lock:
            jpg = output_jpeg
        if jpg is None:
            time.sleep(0.01)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        time.sleep(0.05)

def generate_mask_frames():
    """★ 純讀取函式：推送黑白除錯 JPEG 給客戶端"""
    while True:
        with mask_lock:
            jpg = output_mask_jpeg
        if jpg is None:
            time.sleep(0.01)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        time.sleep(0.05)


# ══════════════════════════════════════════════════════════════════════
# Web API 路由
# ══════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_debug')
def video_feed_debug():
    """★ 黑白除錯影像串流路由"""
    return Response(generate_mask_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    with status_lock:
        return jsonify(dict(sys_status))

@app.route('/set_mode', methods=['POST'])
def set_mode():
    mode = request.form.get('mode')
    is_auto = (mode == 'auto')
    with status_lock:
        sys_status["auto_mode"] = is_auto
    if not is_auto and motor:
        try:
            with i2c_lock:
                motor.full_stop()
        except OSError as e:
            print(f"[I2C] 切換模式煞車失敗: {e}")
    return 'OK'

@app.route('/set_speed', methods=['POST'])
def set_speed():
    speed = request.form.get('speed', type=int)
    if speed is not None:
        with status_lock:
            sys_status["base_speed"] = max(0, min(80, speed))
    return 'OK'

@app.route('/set_pid', methods=['POST'])
def set_pid():
    kp = request.form.get('kp', type=float)
    ki = request.form.get('ki', type=float)
    kd = request.form.get('kd', type=float)
    if None not in (kp, ki, kd):
        pid.update_params(kp, ki, kd)
    return 'OK'

@app.route('/control', methods=['POST'])
def control():
    global last_manual_time, current_manual_action
    
    with status_lock:
        if sys_status["auto_mode"]:
            return 'Ignored'

    action = request.form.get('action')
    if action not in ('stop', 'forward', 'backward', 'left', 'right', 'moveLeft', 'moveRight'):
        return 'Invalid'

    last_manual_time = time.time()
    current_manual_action = action

    return 'OK'

if __name__ == '__main__':
    try:
        print("✅ 伺服器已啟動，按 Ctrl+C 安全關閉")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 收到終止訊號，準備安全關閉系統...")
    finally:
        # 確保車子完全停止，並安全釋放相機資源
        if motor:
            try:
                motor.full_stop()
            except:
                pass
        try:
            if IS_RPI:
                camera.stop()
            else:
                camera.release()
        except:
            pass
        print("✅ 系統已安全關閉")