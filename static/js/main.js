// main.js — AI Car Control Panel

// ── 1. 定時輪詢車輛狀態（每 300ms）──────────────────────────────────
setInterval(fetchStatus, 300);

function fetchStatus() {
    fetch('/get_status')
        .then(r => r.json())
        .then(data => {
            document.getElementById('ai-mode-text').innerText =
                data.auto_mode ? '自動循線中' : '手動模式';
            document.getElementById('lane-error-text').innerText =
                data.error + ' px';
            document.getElementById('steering-text').innerText =
                parseFloat(data.steering).toFixed(2);

            const redEl = document.getElementById('red-light-text');
            if (data.red_light) {
                redEl.innerText = '⚠ DETECTED（停車）';
                redEl.className = 'danger';
            } else {
                redEl.innerText = 'CLEAR';
                redEl.className = 'safe';
            }

            const objText = data.yolo_objects.length > 0
                ? data.yolo_objects.join(', ') : 'No objects';
            document.getElementById('yolo-objs-text').innerText = objText;
            document.getElementById('speed-display-text').innerText = data.base_speed;

            const slider = document.getElementById('speed-slider');
            if (parseInt(slider.value) !== data.base_speed) {
                slider.value = data.base_speed;
                document.getElementById('speed-label').innerText = data.base_speed;
            }
        })
        .catch(err => console.error('狀態讀取失敗:', err));
}

// ── 2. 模式切換 ──────────────────────────────────────────────────────
function setMode(mode) {
    const fd = new URLSearchParams();
    fd.append('mode', mode);
    fetch('/set_mode', { method: 'POST', body: fd })
        .then(() => {
            document.getElementById('btn-manual').classList.toggle('active', mode === 'manual');
            document.getElementById('btn-auto').classList.toggle('active', mode === 'auto');
            // 切回手動時確保停車
            if (mode === 'manual') sendCmd('stop');
        });
}

// ── 3. 速度滑桿 ──────────────────────────────────────────────────────
function onSpeedInput(val) {
    isDragging = true; // 標記為正在拖曳
    document.getElementById('speed-label').innerText = val;
    document.getElementById('speed-display-text').innerText = val;
}

function applySpeed(val) {
    isDragging = false; // 放開拉桿，結束拖曳狀態
    const fd = new URLSearchParams();
    fd.append('speed', val);
    fetch('/set_speed', { method: 'POST', body: fd });
}

// ── 4. PID 套用 ──────────────────────────────────────────────────────
function applyPID() {
    const fd = new URLSearchParams();
    fd.append('kp', document.getElementById('input-kp').value);
    fd.append('ki', document.getElementById('input-ki').value);
    fd.append('kd', document.getElementById('input-kd').value);
    fetch('/set_pid', { method: 'POST', body: fd })
        .then(() => {
            const alertBox = document.getElementById('pid-alert');
            alertBox.classList.remove('hidden');
            setTimeout(() => alertBox.classList.add('hidden'), 2000);
        });
}

// ── 5. 手動控制（心跳模式）──────────────────────────────────────────
// 按壓時每 100ms 送一次指令；放開立即送 stop 並清除計時器
let cmdInterval = null;

function sendCmd(action) {
    const fd = new URLSearchParams();
    fd.append('action', action);
    fetch('/control', { method: 'POST', body: fd });
}

function startCmd(action) {
    // 立即送第一次，再每 100ms 重送（心跳）
    sendCmd(action);
    if (cmdInterval) clearInterval(cmdInterval);
    cmdInterval = setInterval(() => sendCmd(action), 100);
}

function stopCmd() {
    if (cmdInterval) {
        clearInterval(cmdInterval);
        cmdInterval = null;
    }
    sendCmd('stop');
}

function updateStatus() {
    fetch('/get_status')
        .then(response => response.json())
        .then(data => {
            // 1. 更新 AI 模式
            document.getElementById('ai-mode-text').innerText = data.auto_mode ? "自動駕駛" : "手動模式";

            // 2. 更新車道偏移與 PID (從後端取得，假設後端有傳)
            document.getElementById('lane-error-text').innerText = data.error !== null ? data.error : "0";
            document.getElementById('steering-text').innerText = data.steering !== null ? data.steering : "0.00";

            // 3. 更新目前車速顯示
            document.getElementById('speed-display-text').innerText = data.base_speed;

            // 4. 更新紅綠燈狀態
            const redLightEl = document.getElementById('red-light-text');
            if (data.red_light) {
                redLightEl.innerText = "🛑 RED LIGHT";
                redLightEl.style.color = "#ff4444"; // 改為紅色
            } else {
                redLightEl.innerText = "CLEAR";
                redLightEl.style.color = "#00C851"; // 改為綠色
            }

            // 5. 更新 YOLO 辨識物件
            const objText = data.yolo_objects && data.yolo_objects.length > 0
                ? data.yolo_objects.join(', ')
                : "No objects";
            document.getElementById('yolo-objs-text').innerText = objText;
        })
        .catch(err => console.error("抓取狀態失敗:", err));
}

// 每 200 毫秒更新一次網頁資訊
setInterval(updateStatus, 200);