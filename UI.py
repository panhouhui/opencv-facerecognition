import datetime
import cv2
import tkinter as tk
from tkinter import Label, Button, Entry, messagebox, END
import time
import dlib
import face_recognition
from PIL import Image, ImageTk
import threading
import os
import re
import numpy as np

""" 注意：使用该项目：有几个位置需要修改：模型的文件路径需要修改你自己的，需要根据实际情况设置对应的阈值，下载项目的路径，不要出现中文，要不然会出好奇怪的错误  """

# 加载 dlib 面部标志预测模型
predictor_path = r"shape_predictor_68_face_landmarks.dat"  # 请替换为你本地的模型路径

predictor = dlib.shape_predictor(predictor_path)

class FaceRecognitionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("人脸识别系统")
        self.root.geometry("600x750")

        # 摄像头显示框
        self.video_label = Label(root)
        self.video_label.pack(pady=10)

        # 状态显示框
        self.status_label = Label(root, text="状态：等待操作", font=("Arial", 14), fg="blue")
        self.status_label.pack(pady=10)

        # 录入人脸按钮
        self.btn_register = Button(root, text="录入人脸", command=self.register_face, font=("Arial", 12))
        self.btn_register.pack(pady=5)

        # 学号输入框
        self.student_id_label = Label(root, text="学号:", font=("Arial", 12))
        self.student_id_label.pack()
        self.student_id_entry = Entry(root, font=("Arial", 12))
        self.student_id_entry.pack(pady=5)

        # 姓名输入框
        self.name_label = Label(root, text="姓名:", font=("Arial", 12))
        self.name_label.pack()
        self.name_entry = Entry(root, font=("Arial", 12))
        self.name_entry.pack(pady=5)

        # 其他按钮
        self.btn_recognize = Button(root, text="人脸识别签到", command=self.recognize_face, font=("Arial", 12))
        self.btn_recognize.pack(pady=5)

        self.btn_view_logs = Button(root, text="查看签到信息", command=self.view_logs, font=("Arial", 12))
        self.btn_view_logs.pack(pady=5)

        # 退出按钮
        self.btn_exit = Button(root, text="退出系统", command=self.exit_system, font=("Arial", 12), bg="red", fg="white")
        self.btn_exit.pack(pady=5)

        # 创建保存人脸数据的文件夹
        self.face_dir = "faces"
        if not os.path.exists(self.face_dir):
            os.makedirs(self.face_dir)

        # OpenCV 人脸检测
        self.face_cascade = cv2.CascadeClassifier(r"otherDocuments\haarcascade_frontalface_default.xml")

        # 开启摄像头线程
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.update_frame()

    def update_frame(self):
        """ 实时更新摄像头画面 """
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (400, 400))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            self.root.after(10, self.update_frame)

    def register_face(self):
        """ 录入人脸 """
        student_id = self.student_id_entry.get().strip()   #获取输入框中的学号

        name = self.name_entry.get().strip()    #获取输入框中的姓名

        if not student_id or not name:

            messagebox.showerror("错误", "学号和姓名不能为空！")

            return

        # 学号必须是纯数字
        if not student_id.isdigit():

            messagebox.showerror("错误", "学号必须是数字！")

            return

        # 姓名只能包含字母和少量特殊字符（如空格）
        if not re.match(r"^[a-zA-Z\u4e00-\u9fa5 ]+$", name):

            messagebox.showerror("错误", "姓名只能包含汉字或字母！")

            return

        # 检查学号和姓名是否已存在
        filename = f"{self.face_dir}/{student_id}_{name}.jpg"

        # 检查学号或姓名是否已存在
        existing_files = os.listdir(self.face_dir)  # 获取目录下的所有文件

        for file in existing_files:

            # 提取文件名中的学号和姓名部分
            existing_name, existing_student_id = file.replace(".jpg", "").split("_")

            if existing_name == name or existing_student_id == student_id:

                messagebox.showwarning("警告", f"该用户已录入！")

                return

        self.status_label.config(text=f"状态：正在录入 {student_id}-{name}", fg="red")

        # 启动线程进行拍照和录入
        threading.Thread(target=self.capture_face, args=(student_id, name, filename)).start()

    def capture_face(self, student_id, name, filename):
        """ 拍照并检测人脸 """
        ret, frame = self.cap.read()

        if not ret or frame is None:

            messagebox.showerror("错误", "无法捕获人脸图像！")

            return

        # 转换为灰度图以检测人脸
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用级联分类器检测人脸
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) == 0:

            messagebox.showerror("错误", "未检测到人脸，请调整姿势重试！")

            return

        # 确保文件夹存在
        if not os.path.exists(self.face_dir):

            os.makedirs(self.face_dir)

        # 处理文件名，避免中文或特殊字符导致路径错误
        safe_name = re.sub(r'[^\w\s]', '', name)

        file_path = os.path.abspath(os.path.join(self.face_dir, f"{safe_name}_{student_id}.jpg"))

        # 先进行人脸比对检查，避免重复录入
        if self.is_duplicate_face(frame):

            messagebox.showerror("错误", "检测到相同的人脸，无法重复录入！")

            return

        for (x, y, w, h) in faces:
            face_rectangle = dlib.rectangle(x, y, x + w, y + h)
            landmarks = predictor(gray, face_rectangle)

            # 在图像上标记面部特征点
            for n in range(68):

                x_landmark = landmarks.part(n).x

                y_landmark = landmarks.part(n).y

                cv2.circle(frame, (x_landmark, y_landmark), 1, (0, 255, 0), -1)

        # 保存图片
        try:

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 转换为RGB

            npy_path = r'opencv人脸识别/npy'

            self.save_image_as_npy(image, student_id, name)

            image.save(file_path)

            messagebox.showinfo("成功", f"人脸已保存：{file_path}")

            self.update_status(f"状态：{name}-{student_id} 录入完成", "blue")

        except Exception as e:

            print(f"保存图片时发生错误: {e}")  # 输出异常信息

            messagebox.showerror("错误", f"保存人脸图像失败：{e}")

    def save_image_as_npy(self, rgb_image, student_id, name):
        """ 提取人脸特征并保存为 .npy 文件 """
        try:
            npy_dir = "npy"  # 存储 npy 文件的文件夹
            os.makedirs(npy_dir, exist_ok=True)  # 确保文件夹存在

            filename = f"{name}_{student_id}.npy"  # 生成文件名
            output_path = os.path.join(npy_dir, filename).replace("\\", "/")

            # **1. 确保 rgb_image 是 NumPy 数组**
            if not isinstance(rgb_image, np.ndarray):
                rgb_image = np.array(rgb_image)  # 转换为 NumPy 数组

            # **2. 检测人脸位置**
            face_locations = face_recognition.face_locations(rgb_image)

            if len(face_locations) == 0:
                print("未检测到人脸，无法保存特征！")
                return False

            # **3. 提取人脸特征**
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

            if len(face_encodings) == 0:
                print("无法提取人脸特征！")
                return False

            # **4. 只保存 128 维特征向量**
            np.save(output_path, face_encodings[0])

            print(f"已保存 {name} 的人脸特征到 {output_path}")

            return True

        except Exception as e:
            print(f"保存人脸特征时发生错误: {e}")
            return False

    def is_duplicate_face(self, file_path):
        """ 检查是否已有相同的人脸数据（简化为文件路径检查） """
        # 比如使用 `face_recognition` 库来比较人脸特征

        # 检查文件是否存在
        if os.path.exists(file_path):

            return True

        return False

    def update_status(self, status_text, color):
        """ 在主线程中更新状态 """
        self.status_label.config(text=status_text, fg=color)

    def recognize_face(self):
        """ 模拟人脸识别签到 """
        self.face_encodings = {}  # 用于存储人脸特征

        self.face_dir = "npy"  # 存储 npy 文件的文件夹

        self.cap = cv2.VideoCapture(0)  # 摄像头实例

        # 启动线程进行人脸识别
        threading.Thread(target=self.simulate_process, args=("人脸识别签到",)).start()

        # 捕获摄像头画面
        ret, frame = self.cap.read()

        if not ret:
            messagebox.showerror("错误", "无法捕获摄像头画面！")
            return

        # 使用 face_recognition 提取当前图像中的人脸特征
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 检测当前帧中的人脸位置
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 0:
            messagebox.showerror("错误", "未检测到人脸，请调整姿势重试！")
            return

        # 提取人脸特征（注意：face_encodings 会返回128维特征向量）
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_encodings) == 0:

            messagebox.showerror("错误", "未检测到人脸特征！")

            return

        # 加载已录入的人脸特征
        self.load_face_encodings()

        # 比对当前图像中的人脸特征与已录入的人脸特征
        for (name,student_id), known_encoding in self.face_encodings.items():

            # 使用 .shape[0] 来确保特征向量是128维
            if known_encoding.shape[0] == 128 and face_encodings[0].shape[0] == 128:
                # 计算欧氏距离
                face_distance = face_recognition.face_distance([known_encoding], face_encodings[0])

                # 设置一个阈值，判断相似度是否足够高
                threshold = 0.9  # 你可以根据需要调整这个值

                if face_distance[0] < threshold:  # 如果距离小于阈值，认为是匹配的

                    self.update_status(f"状态：{name}_{student_id} 签到成功", "green")

                    self.log_sign_in(name, student_id)  # **记录签到信息**

                    return

        # 如果没有匹配到
        messagebox.showerror("错误", "未匹配到录入的人脸，请录入人脸后再进行签到！")

    def load_face_encodings(self):
        """ 从.npy文件加载人脸特征 """
        self.face_encodings = {}  # 用于存储所有学生的人脸特征

        # 获取文件夹中的所有.npy文件
        for filename in os.listdir(self.face_dir):
            if filename.endswith(".npy"):
                student_id, name = filename.split("_")  # 提取学号和姓名
                npy_path = os.path.join(self.face_dir, filename)

                # 从 .npy 文件加载人脸特征（确保是128维的向量）
                known_encoding = np.load(npy_path)  # 载入的应该是128维的特征向量

                # 存储学号、姓名与人脸特征的映射
                self.face_encodings[(student_id, name)] = known_encoding
                print(f"加载了 {name}_{student_id} 的人脸特征")

    def log_sign_in(self, name, student_id):
        """ 记录签到信息到 sign_in_log.txt """
        log_file = "sign_in_log.txt"  # 签到日志文件

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_entry = f"{current_time} - 学号: {student_id}, 姓名: {name}\n"

        try:
            with open(log_file, "a", encoding="utf-8") as file:

                file.write(log_entry)

            print(f"签到记录成功: {log_entry.strip()}")

        except Exception as e:

            print(f"记录签到时发生错误: {e}")

    def view_logs(self):
        """ 查看签到信息（可扩展） """
        self.status_label.config(text="状态：查看签到信息", fg="black")

        """ 打开一个窗口显示签到记录 """
        log_file = "sign_in_log.txt"  # 签到日志文件

        # 创建新窗口
        log_window = tk.Toplevel(self.root)
        log_window.title("签到记录")
        log_window.geometry("500x400")

        # 创建文本框（Text Widget）用于显示签到内容
        text_area = tk.Text(log_window, wrap="word", font=("Arial", 12))
        text_area.pack(expand=True, fill="both", padx=10, pady=10)

        # 添加滚动条
        scrollbar = tk.Scrollbar(log_window)
        scrollbar.pack(side="right", fill="y")
        scrollbar.config(command=text_area.yview)
        text_area.config(yscrollcommand=scrollbar.set)

        try:
            # 读取签到记录
            with open(log_file, "r", encoding="utf-8") as file:

                logs = file.read()

        except FileNotFoundError:

            logs = "暂无签到记录"

        # 插入日志内容，并确保文本框可见
        text_area.config(state="normal")
        text_area.insert(END, logs if logs.strip() else "暂无签到记录")
        text_area.config(state="disabled")  # 设为只读

    def simulate_process(self, process_name):
        """ 模拟人脸处理过程 """
        time.sleep(3)
        self.update_status(f"状态：{process_name}完成", "blue")

    def close_app(self):
        """ 关闭应用时释放摄像头 """
        self.running = False
        self.cap.release()
        self.root.quit()

    def exit_system(self):
        """ 退出系统 """
        self.running = False
        self.cap.release()  # 释放摄像头
        self.root.quit()  # 退出 Tkinter 应用

if __name__ == "__main__":

    root = tk.Tk()

    app = FaceRecognitionApp(root)

    root.protocol("WM_DELETE_WINDOW", app.close_app)

    root.mainloop()
