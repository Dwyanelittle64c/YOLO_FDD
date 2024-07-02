import os
import threading
import time
import tkinter
from tkinter import filedialog

import tkinter.messagebox

import cv2

from PIL import ImageDraw, Image, ImageTk, ImageFont

from utils.utils import resize_image


color = (240, 240, 240)
cap_key = 0
video_key=0
stop=False


def get_img_path(strs, fr_result, lb_img):
    print("获取img_path")
    sfname = filedialog.askopenfilename(title='选择图片', filetypes=[('JPG','.jpg'),('PNG','.png'),('JPEG','.jpeg'),('All Files', '*')])
    type = sfname[-4:].lower()
    if type in ['.jpg', '.png', '.jpeg']:
        # print(sfname[-4:])
        print(sfname)
        strs.set(sfname)
        fr_result.config(text='img_path')
        global image
        image = Image.open(sfname)
        # print(image.size)

        image = ImageTk.PhotoImage(resize_image(image, (1050, 420), True, color))
        # lb_img.config(image=image)
        lb_img['image'] = image
        # bt_pred_img.config(text='111')
        # return sfname
    elif type == '':
        pass
    else:
        tkinter.messagebox.showerror(title="Error", message="选取的不是一张图片")


def get_dir_path(strs, fr_result):
    print("获取dir_path")
    sfname = filedialog.askdirectory(title='选择文件夹')
    print(sfname)
    if sfname != '':
        strs.set(sfname)
    # return sfname
    fr_result.config(text='dir_path')


def get_video_path(strs, fr_result, lb_img):
    # todo 读入视频路径
    print("获取video_path")
    sfname = filedialog.askopenfilename(title='选择视频', filetypes=[('MP4','.mp4'),('All Files', '*')])
    type = sfname[-4:].lower()
    if type in ['.mp4', ]:
        # print(sfname[-4:])
        # print(sfname)
        strs.set(sfname)
        fr_result.config(text='video_path')
        capture = cv2.VideoCapture(sfname)
        ref, frame = capture.read()
        if not ref:
            tkinter.messagebox.showerror(title="Error",
                                         message="未能正确读取视频")
        else:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            # 转变成Image
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(resize_image(frame, (1050, 420), True, color))
            lb_img.imgtk = frame
            lb_img.config(image=frame)

        # print(image.size)

        # bt_pred_img.config(text='111')
        # return sfname
    elif type == '':
        pass
    else:
        tkinter.messagebox.showerror(title="Error", message="选取的不是视频")


def predict_cap(strs):
    print("预测摄像头")
    capture = cv2.VideoCapture(0)
    ref, frame = capture.read()
    if not ref:
        tkinter.messagebox.showerror(title="Error",
                                     message="未能正确读取摄像头，请注意是否正确安装摄像头")
    else:
        tkinter.messagebox.showinfo(title="Info", message="成功连接摄像头")
        strs.set('Successfully connected camera!')
    capture.release()


def predict_start(strs, lb_img, yolo, bt_pred_start):
    global cap_key
    global video_key
    global stop


    print('start')

    path = strs.get()
    print(path)

    # todo 检测单张或视频
    if os.path.isfile(path):
        # todo 检测图片
        if path[-4:] in ['.jpg', '.png', '.jpeg']:
            # 必须加global，不然不显示图片
            global image
            results_list, image = predict_img(path, yolo)
            # image.show()
            # print(image.size)
            image = ImageTk.PhotoImage(resize_image(image, (1050, 420), True, color))
            lb_img.config(image=image)
            results = ''
            for i in results_list:
                results += str(i)
                results += '\n'
            if str(results)=='':
                strs.set('未检测到目标！')
            else:
                strs.set(str(results))
        # todo 检测视频
        else:
            print('检测视频')
            capture = cv2.VideoCapture(path)
            ref, frame = capture.read()
            if not ref:
                tkinter.messagebox.showerror(title="Error",
                                             message="未能正确读取视频，请注意是否正确填写视频路径")

            if ref and video_key == 0:
                bt_pred_start.config(text='停止')
                video_key = 1
                predict_video_show(capture=capture, lb_img=lb_img, yolo=yolo)

            else:
                stop=True
                video_key = 0
                bt_pred_start.config(text='开始')

                image = ImageTk.PhotoImage(resize_image(Image.open("yolov7net.png"),(1050, 420), True, color))
                lb_img.config(image=image)
                strs.set('请选择操作！')


            # predict_video(video_path=path,yolo=yolo,lb_img=lb_img)

    # todo 检测文件夹
    elif os.path.isdir(path):
        print('检测文件夹')


    # todo 调用摄像头检测
    elif path == 'Successfully connected camera!':
        print('调用摄像头检测')
        capture = cv2.VideoCapture(0)
        ref, frame = capture.read()
        if not ref:
            tkinter.messagebox.showerror(title="Error",
                                         message="未能正确读取摄像头，请注意是否正确安装摄像头")

        if cap_key == 0:
            cap_key = 1
            bt_pred_start.config(text='停止')
            predict_video_show(capture=capture, lb_img=lb_img, yolo=yolo)
        else:
            cap_key = 0
            bt_pred_start.config(text='开始')
            capture.release()
            image = ImageTk.PhotoImage(resize_image(Image.open("yolov7net.png"),(1050, 420), True, color))
            lb_img.config(image=image)
            strs.set('请选择操作！')

    else:
        print('please choose a mode.')
        pass


def predict_img(path, yolo):
    # yolo= YOLO()

    try:
        image = Image.open(path)
    except:
        print('Open Error! Try again!')
    else:
        results_list, image = yolo.detect_img_cut(image, th=20)
        # print(yolo.class_names)

        for result in results_list:
            result[0][0]=yolo.class_names[int(result[0][0])]
        print(results_list)
        return results_list, image


def predict_video_show(capture, lb_img, yolo):
    fps = 0.0
    global stop
    while capture.isOpened():
        if stop:


            stop=False
            break

        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        if not ref:
            break
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # 转变成Image
        frame = Image.fromarray(frame)
        # 进行检测
        frame = yolo.detect_image(frame)

        fps = (fps + (1. / (time.time() - t1))) / 2
        draw = ImageDraw.Draw(frame)
        # font = ImageFont.truetype(font='model_data/simhei.ttf',
        #                           size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        font = ImageFont.truetype(font='../model_data/simhei.ttf',
                                  size=40)
        draw.text((20, 20), "FPS:%.1f" % (fps), fill=(0, 255, 255), font=font)

        # frame.show()
        # image.show()
        # print(image.size)
        frame = ImageTk.PhotoImage(resize_image(frame, (1050, 420), True, color))
        lb_img.imgtk = frame
        lb_img.config(image=frame)
        lb_img.update()
        # lb_img.image=frame
        # lb_img['image'] = frame

        # print(fps)
        del draw

        # frame = cv2.putText(frame, "FPS: %.1f" % (fps), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)

    print("Video Detection Done!")
