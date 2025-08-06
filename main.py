# https://github.com/xx236/25-E-MAIXCAM-
import re
from maix import camera, display, image, nn, app,uart,time
import struct,math

device="/dev/serial0"
serial = uart.UART(device, 115200)

# 卡尔曼滤波器实现


class TargetTracker:
    def __init__(self):
        # 状态向量 [x, y, vx, vy] - 位置和速度
        self.state = [0, 0, 0, 0]
        
        # 协方差矩阵 (简化版)
        self.P = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
        
        # 状态转移矩阵 (匀速模型)
        self.F = [[1, 0, 0.1, 0],  # dt=0.1s
                  [0, 1, 0, 0.1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
        
        # 过程噪声 (根据目标机动性调整)
        self.Q = 0.01
        
        # 测量噪声 (根据检测稳定性调整)
        self.R = 0.1
        
        # 移动平均窗口
        self.ma_window = []
        self.window_size = 5
        
        # 计时器
        self.last_update = time.ticks_ms()
        self.dt = 0.1  # 默认时间间隔
        
    def predict(self):
        """预测目标位置"""
        # 更新时间间隔
        current_time = time.ticks_ms()
        self.dt = time.ticks_diff(current_time, self.last_update) / 1000.0
        self.last_update = current_time
        
        # 更新状态转移矩阵中的时间项
        self.F[0][2] = self.dt
        self.F[1][3] = self.dt
        
        # 状态预测: state = F * state
        x, y, vx, vy = self.state
        self.state = [
            x + vx * self.dt,
            y + vy * self.dt,
            vx,
            vy
        ]
        
        # 简化协方差预测 (避免矩阵运算)
        for i in range(4):
            self.P[i][i] += self.Q
    
    def update(self, x_meas, y_meas):
        """用新测量值更新滤波器"""
        # 移动平均滤波先处理测量值
        self.ma_window.append((x_meas, y_meas))
        if len(self.ma_window) > self.window_size:
            self.ma_window.pop(0)
        
        # 计算移动平均
        x_sum = sum(p[0] for p in self.ma_window)
        y_sum = sum(p[1] for p in self.ma_window)
        x_ma = x_sum / len(self.ma_window)
        y_ma = y_sum / len(self.ma_window)
        
        # 预测当前状态
        self.predict()
        
        # 计算卡尔曼增益 (简化版)
        S = self.P[0][0] + self.R  # 位置x的协方差
        Kx = self.P[0][0] / S
        Ky = self.P[1][1] / (self.P[1][1] + self.R)
        
        # 更新状态估计
        x, y, vx, vy = self.state
        self.state = [
            x + Kx * (x_ma - x),
            y + Ky * (y_ma - y),
            vx,
            vy
        ]
        
        # 更新协方差 (简化)
        self.P[0][0] *= (1 - Kx)
        self.P[1][1] *= (1 - Ky)
        
        return self.state[0], self.state[1], self.state[2], self.state[3]
    
    def get_position(self):
        """获取当前位置和速度"""
        return self.state[0], self.state[1], self.state[2], self.state[3]
    
    def get_predicted_position(self, lookahead=0.1):
        """预测未来位置（用于画圆功能）"""
        x, y, vx, vy = self.state
        return x + vx * lookahead, y + vy * lookahead
tracker = TargetTracker()
def send_data_packet(x, y):
    packet = struct.pack("<bbHH",
                       0x7A, 0x70,
                       int(x),
                       int(y))
    serial.write(packet)
    print(f"Sent: ({x}, {y})")


detector = nn.YOLOv5(model="/root/models/model_rect.mud")
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
dis = display.Display()
print("www")
print(detector.input_width(),detector.input_height(), detector.input_format())

while not app.need_exit():
    img = cam.read()#.lens_corr(strength=1.1)
    objs = detector.detect(img, conf_th = 0.75, iou_th = 0.45)
    for obj in objs:
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_RED)
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y, msg, color = image.COLOR_RED)
        x, y, vx, vy = tracker.update((obj.x+obj.w/2), (obj.y+obj.h/2))
        img.draw_cross(int(x),int(y),image.COLOR_GREEN,3,2)
        send_data_packet(x, y)   

    dis.show(img)
