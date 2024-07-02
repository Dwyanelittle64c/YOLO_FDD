import tkinter
from tkinter import *

from PIL import ImageTk, Image

from utils.utils import resize_image
from utils.utils_GUI import get_dir_path, get_video_path, predict_cap, predict_start, get_img_path
from yolo import YOLO


color=(240,240,240)
root = Tk()
root.title("布匹缺陷检测系统")
# root.iconbitmap("my_icon.ico")
root.geometry("1280x720")
# root.resizable(False, False)

# root.config(bg="blue")
yolo=YOLO()

fr_img = LabelFrame(root, text="image", width=1050, height=420)

image = ImageTk.PhotoImage(resize_image(Image.open("yolov7net.png"),(1050, 420), True, color))
lb_img=Label(fr_img, image=image)
lb_img.pack()
# fr_img.pack(fill="both", padx=10)
# fr_img.place(x=100,y=100)

# Button(fr_img, text="1").pack()
# Button(fr_img, text="2").pack()

frame_title = StringVar()  # 创建sts为Tk中可变的变量

fr_result = LabelFrame(root, text="info", width=1050, height=180,pady=10)

path = StringVar()  # 创建sts为Tk中可变的变量
path.set("请选择操作!")  # 初始化strs变量
et_result = Label(fr_result, width=150, height=10, textvariable=path)
et_result.pack()

fr_buttons = Frame(root, width=150, height=720)

# fr_buttons.grid(row=0,column=1)
# fr_buttons.pack(fill="both", padx=10,side='right',expand=False)
# fr_buttons.place()

bt_pred_img = Button(fr_buttons, text="选择单张图片", width=16, height=2, command=lambda: get_img_path(path,fr_result,lb_img))
bt_pred_img.pack(pady=(10,30),padx=(20,0))

bt_pred_dir = Button(fr_buttons, text="选择文件夹", width=16, height=2, command=lambda: get_dir_path(path,fr_result))
bt_pred_dir.pack(pady=30,padx=(20,0))

bt_pred_video = Button(fr_buttons, text="选择视频", width=16, height=2, command=lambda: get_video_path(path,fr_result,lb_img))
bt_pred_video.pack(pady=30,padx=(20,0))

bt_pred_cap = Button(fr_buttons, text="连接摄像头", width=16, height=2, command=lambda: predict_cap(path))
bt_pred_cap.pack(pady=30,padx=(20,0))

bt_pred_start = Button(fr_buttons, text="开始", width=16, height=2, command=lambda: predict_start(yolo=yolo,strs=path,
                                                                                                  lb_img=lb_img,bt_pred_start=bt_pred_start,
                                                                                                  ))
bt_pred_start.pack(pady=(100, 40),padx=(20,0))
# bt_pred_start.config(state='disabled')

fr_img.grid(row=0, column=0, padx=(20, 0), pady=10)
fr_buttons.grid(row=0, column=1, padx=20, rowspan=2)
fr_result.grid(row=1, column=0, padx=(20, 0))
# relief = flat, groove, raised, ridge, solid, sunken
# lab1 = Label(root, text="Hello, Tkinter!",relief='solid')
# lab2 = Label(root, text="Hello, Tkinter2!",relief='raised')
# lab1.pack()
# lab2.pack()

mainloop()
