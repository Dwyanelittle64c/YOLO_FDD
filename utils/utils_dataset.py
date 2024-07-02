import json
import random

import xml.etree.ElementTree as ET

import cv2
import numpy as np
import tqdm
from PIL import Image, ImageFont
from PIL import ImageDraw
import os
from shutil import copyfile


def delete_file(dir_path):
    dir_list=os.listdir(dir_path)
    for i in dir_list:
        os.remove(os.path.join(dir_path,i))



def show_dataset(img_path='VOCdevkit/VOCdataset/JPEGImages', anno_path='VOCdevkit/VOCdataset/annotation.json',image_size=(600,600)):
    print('show_dataset')
    class_list=[]
    img_list=os.listdir(img_path)
    random.shuffle(img_list)
    with open(anno_path, 'r') as f:
        dict = json.load(f)

    classes_num={}
    for i in dict:
        if i['defect_name'] not in classes_num:
            classes_num[i['defect_name']]=0
        else:
            classes_num[i['defect_name']]+=1
    print(classes_num)

    for img in img_list:
        image = cv2.imread(os.path.join(img_path, img))
        exsit=False
        watch=False
        for i in dict:
            if i['name']==img:
                if exsit==False:
                    print('\n',img,"",end='')
                exsit=True
                de = i.get('defect_name')
                if de in class_list and class_list!=[]:
                    watch=True
                elif class_list==[]:
                    watch=True
                bbox = i.get('bbox')
                a = bbox[0]
                b = bbox[1]
                c = bbox[2]
                d = bbox[3]
                print(de,bbox,end='')
                fontpath = "model_data/simhei.ttf"
                font = ImageFont.truetype(fontpath, 32)
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)
                # 绘制文字信息
                if b > 50:
                    draw.text((a, b - 40), de, font=font, fill=(0, 255, 0))
                else:
                    draw.text((a + 50, b), de, font=font, fill=(0, 255, 0))

                image = np.array(image)
                cv2.rectangle(image, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 10)
        if not exsit:
            print(img,'has not bbox!!')
        if watch:
            image = cv2.resize(image, image_size)
            cv2.imshow('window', image)
            cv2.waitKey()

            # delete=input('\ndel?')
            # if delete=='y':
            #     print('deletc',img)
            #     os.remove(os.path.join(img_path, img))


def get_cut_imgs_8(image):
    image=image.resize((2048,1024))
    # print(image.size)
    # image.show()
    image=np.array(image)

    # print(image.shape)
    w=image.shape[1]
    h=image.shape[0]
    imgs=[]
    a=512
    for i in range(2):
        for j in range(4):
            img_cut=image[a*i:a*(i+1),a*j:a*(j+1),:].copy()
            # print(img_cut.shape)

            img_cut = Image.fromarray(img_cut)
            # img.show()
            imgs.append(img_cut)

    return imgs





def img_cut_10(img_save_path, json_save_path=r'D:\DeepLearning\dataset\cloth_cut\train\anno/anno.json'):
    print('img_cut')
    delete_file(dir_path=img_save_path)
    print('delete_file done.')
    anno_path = r"D:\DeepLearning\dataset\Cloth\train1_datasets\guangdong1_round1_train1_20190828\Annotations\anno_train.json"
    img_path = r"D:\DeepLearning\dataset\Cloth\train1_datasets\guangdong1_round1_train1_20190828\defect_Images"
    pre_h = 1000
    pre_w = 2446
    h = 1024
    w = 2560
    num_h = 2
    num_w = 5

    drop_rate=0.3

    long_w = w // num_w
    long_h = h // num_h
    img_list = os.listdir(img_path)
    print("all:",len(img_list))
    dict_new=[]

    with open(anno_path, 'r') as f:
        dict = json.load(f)

    # img_list=['005bcbfd126fb67c1413297296.jpg','00fc63aff57def8d0947060525.jpg']

    for idx,i in enumerate(tqdm.tqdm(img_list)):
        img = cv2.imread(os.path.join(img_path, i))
        # print(os.path.join(img_path,i))
        # print(img)
        img = cv2.resize(img, (w, h))
        # img=np.array(img)
        img_part_list = []
        img_part_list.append(img[long_h * 0:long_h * 1, long_w * 0:long_w * 1, ].copy())
        img_part_list.append(img[long_h * 0:long_h * 1, long_w * 1:long_w * 2, ].copy())
        img_part_list.append(img[long_h * 0:long_h * 1, long_w * 2:long_w * 3, ].copy())
        img_part_list.append(img[long_h * 0:long_h * 1, long_w * 3:long_w * 4, ].copy())
        img_part_list.append(img[long_h * 0:long_h * 1, long_w * 4:long_w * 5, ].copy())

        img_part_list.append(img[long_h * 1:long_h * 2, long_w * 0:long_w * 1, ].copy())
        img_part_list.append(img[long_h * 1:long_h * 2, long_w * 1:long_w * 2, ].copy())
        img_part_list.append(img[long_h * 1:long_h * 2, long_w * 2:long_w * 3, ].copy())
        img_part_list.append(img[long_h * 1:long_h * 2, long_w * 3:long_w * 4, ].copy())
        img_part_list.append(img[long_h * 1:long_h * 2, long_w * 4:long_w * 5, ].copy())


        id_list=[]
        # print(i,'done.')
        for d in dict:
            if d['name'] == i:
                if d['bbox'][0]==0:
                    d['bbox'][0]=1
                if d['bbox'][1]==0:
                    d['bbox'][1]=1
                if d['bbox'][2]==pre_w:
                    d['bbox'][2]=pre_w-1
                if d['bbox'][3]==pre_h:
                    d['bbox'][3]=pre_h-1
                x1 = d['bbox'][0]
                y1 = d['bbox'][1]
                x2 = d['bbox'][2]
                y2 = d['bbox'][3]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                x1,x2,y1,y2=int(x1*w/pre_w),int(x2*w/pre_w),int(y1*h/pre_h),int(y2*h/pre_h)
                _x1,_x2,_y1,_y2=x1,x2,y1,y2
                kx1,kx2,ky1,ky2=x1//long_w,x2//long_w,y1//long_h,y2//long_h

                square_src=(x2-x1)*(y2-y1)




                if kx1==kx2 and ky1==ky2:
                    x1,x2,y1,y2=x1%long_w,x2%long_w,y1%long_h,y2%long_h
                    if ky1 == 0:
                        img_id = kx1
                    else:
                        img_id = kx1 + 5
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(img_id) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]

                    dict_new.append(box_dict)
                    id_list.append(img_id)

                else:
                    # num_part=(kx2-kx1+1)*(ky2-ky1+1)
                    # part_list=[]

                    for id_x in range(kx1,kx2+1):
                        for id_y in range(ky1,ky2+1):
                            x1,x2,y1,y2=_x1,_x2,_y1,_y2
                            # print(id_x,id_y)
                            # 左上角框
                            if id_x == kx1 and id_y==ky1:
                                x1=x1%long_w
                                y1=y1%long_h
                                if ky1==ky2: # 未跨行
                                    x2=long_w
                                    y2=y2%long_h
                                elif kx1==kx2:
                                    x2=x2%long_w
                                    y2=long_h

                            # 右下角框
                            if id_x == kx2 and id_y == ky2:
                                x2 = x2 % long_w
                                y2 = y2 % long_h
                                if ky1 == ky2:
                                    x1 = 1
                                    y1 = y1 % long_h
                                elif kx1 == kx2:
                                    x1 = x1 % long_w
                                    y1 = 1

                            # 中间框
                            if kx1<id_x<kx2:
                                x1=1
                                x2=long_w
                                if ky1==ky2:
                                    y1=y1%long_h
                                    y2=y2%long_h
                                elif ky1==id_y:
                                    y1=y1%long_h
                                    y2=long_w
                                elif ky2==id_y:
                                    y1=1
                                    y2=y2%long_h

                            # 出现横跨的情况
                            if (kx2-kx1)>=2 and ky1!=ky2:
                                # 右上角
                                if id_x==kx2 and id_y==ky1:
                                    x1=1
                                    x2=x2%long_w
                                    y1=y1%long_h
                                    y2=long_h
                                # 左下角
                                elif id_x==kx1 and id_y==ky2:
                                    x1=x1%long_w
                                    x2=long_w
                                    y1=1
                                    y2=y2%long_h





                            if id_y==0:
                                img_id=id_x
                            else:
                                img_id=id_x+5

                            box_dict = {}
                            box_dict['name']=i[0:-4]+'_'+str(img_id)+'.jpg'
                            box_dict['defect_name']=d['defect_name']
                            box_dict['bbox']=[x1,y1,x2,y2]

                            square=(x2-x1)*(y2-y1)
                            if square>square_src*drop_rate and 0<=x1<=long_w\
                                    and 0<x2<=long_w\
                                    and 0<y1<=long_h\
                                    and 0<y2<=long_h:
                                dict_new.append(box_dict)
                                id_list.append(img_id)

                    # return


                # if long_w * 0 < center_x < long_w * 1 and long_h * 0 < center_y < long_h * 1:
                #     x1 = long_w * 0+1 if x1 < long_w * 0 else x1
                #     x2 = long_w * 1-1 if x2 > long_w * 1 else x2
                #     y1 = long_h * 0+1 if y1 < long_h * 0 else y1
                #     y2 = long_h * 1-1 if y2 > long_h * 1 else y2
                #     path_id='0'
                # #     0348886eaf1a6e250914417184
                #
                # if long_w * 1 < center_x < long_w * 2 and long_h * 0 < center_y < long_h * 1:
                #     x1 = long_w * 1+1 if x1 < long_w * 1 else x1
                #     x2 = long_w * 2-1 if x2 > long_w * 2 else x2
                #     y1 = long_h * 0+1 if y1 < long_h * 0 else y1
                #     y2 = long_h * 1-1 if y2 > long_h * 1 else y2
                #     path_id='1'
                #
                # if long_w * 2 < center_x < long_w * 3 and long_h * 0 < center_y < long_h * 1:
                #     x1 = long_w * 2+1 if x1 < long_w * 2 else x1
                #     x2 = long_w * 3-1 if x2 > long_w * 3 else x2
                #     y1 = long_h * 0+1 if y1 < long_h * 0 else y1
                #     y2 = long_h * 1-1 if y2 > long_h * 1 else y2
                #     path_id='2'
                #
                # if long_w * 3 < center_x < long_w * 4 and long_h * 0 < center_y < long_h * 1:
                #     x1 = long_w * 3+1 if x1 < long_w * 3 else x1
                #     x2 = long_w * 4-1 if x2 > long_w * 4 else x2
                #     y1 = long_h * 0+1 if y1 < long_h * 0 else y1
                #     y2 = long_h * 1-1 if y2 > long_h * 1 else y2
                #     path_id='3'
                #
                # if long_w * 4 < center_x < long_w * 5 and long_h * 0 < center_y < long_h * 1:
                #     x1 = long_w * 4+1 if x1 < long_w * 4 else x1
                #     x2 = long_w * 5-1 if x2 > long_w * 5 else x2
                #     y1 = long_h * 0+1 if y1 < long_h * 0 else y1
                #     y2 = long_h * 1-1 if y2 > long_h * 1 else y2
                #     path_id='4'
                #
                # if long_w * 0 < center_x < long_w * 1 and long_h * 1 < center_y < long_h * 2:
                #     x1 = long_w * 0+1 if x1 < long_w * 0 else x1
                #     x2 = long_w * 1-1 if x2 > long_w * 1 else x2
                #     y1 = long_h * 1+1 if y1 < long_h * 1 else y1
                #     y2 = long_h * 2-1 if y2 > long_h * 2 else y2
                #     path_id='5'
                #
                # if long_w * 1 < center_x < long_w * 2 and long_h * 1 < center_y < long_h * 2:
                #     x1 = long_w * 1+1 if x1 < long_w * 1 else x1
                #     x2 = long_w * 2-1 if x2 > long_w * 2 else x2
                #     y1 = long_h * 1+1 if y1 < long_h * 1 else y1
                #     y2 = long_h * 2-1 if y2 > long_h * 2 else y2
                #     path_id='6'
                #
                # if long_w * 2 < center_x < long_w * 3 and long_h * 1 < center_y < long_h * 2:
                #     x1 = long_w * 2+1 if x1 < long_w * 2 else x1
                #     x2 = long_w * 3-1 if x2 > long_w * 3 else x2
                #     y1 = long_h * 1+1 if y1 < long_h * 1 else y1
                #     y2 = long_h * 2-1 if y2 > long_h * 2 else y2
                #     path_id='7'
                #
                # if long_w * 3 < center_x < long_w * 4 and long_h * 1 < center_y < long_h * 2:
                #     x1 = long_w * 3+1 if x1 < long_w * 3 else x1
                #     x2 = long_w * 4-1 if x2 > long_w * 4 else x2
                #     y1 = long_h * 1+1 if y1 < long_h * 1 else y1
                #     y2 = long_h * 2-1 if y2 > long_h * 2 else y2
                #     path_id='8'
                #
                # if long_w * 4 < center_x < long_w * 5 and long_h * 1 < center_y < long_h * 2:
                #     x1 = long_w * 4+1 if x1 < long_w * 4 else x1
                #     x2 = long_w * 5-1 if x2 > long_w * 5 else x2
                #     y1 = long_h * 1+1 if y1 < long_h * 1 else y1
                #     y2 = long_h * 2-1 if y2 > long_h * 2 else y2
                #     path_id='9'
                #
                # x1, x2, y1, y2 = x1 % long_w, x2 % long_w, y1 % long_h, y2 % long_h




        for id, image in enumerate(img_part_list):
            path = os.path.join(img_save_path, i[0:-4] + '_' + str(id) + '.jpg')
            if id in id_list:
                cv2.imwrite(path, image)
        # return
    with open(json_save_path, 'w') as f:
        json.dump(dict_new, f)

        # break

def img_cut_2(img_save_path, json_save_path=r'D:\DeepLearning\dataset\cloth_cut\train\anno/anno.json'):
    print('img_cut')
    delete_file(dir_path=img_save_path)
    print('delete_file done.')
    anno_path = r"D:\DeepLearning\dataset\Cloth\train1_datasets\guangdong1_round1_train1_20190828\Annotations\anno_train.json"
    img_path = r"D:\DeepLearning\dataset\Cloth\train1_datasets\guangdong1_round1_train1_20190828\defect_Images"
    pre_h = 1000
    pre_w = 2446
    h = 1024
    w = 2048
    num_h = 1
    num_w = 2

    drop_rate=0.5

    long_w = w // num_w     # 1024
    long_h = h // num_h     # 1024
    img_list = os.listdir(img_path)
    print("all:",len(img_list))
    dict_new=[]

    with open(anno_path, 'r') as f:
        dict = json.load(f)

    # img_list=['005bcbfd126fb67c1413297296.jpg','00fc63aff57def8d0947060525.jpg']

    for idx,i in enumerate(tqdm.tqdm(img_list)):
        img = cv2.imread(os.path.join(img_path, i))
        # print(os.path.join(img_path,i))
        # print(img)
        img = cv2.resize(img, (w, h))
        # img=np.array(img)
        img_part_list = []
        img_part_list.append(img[0:1024,0:1024].copy())
        img_part_list.append(img[0:1024,1024:2048].copy())
        # print(img_part_list[0].shape)
        # print(img_part_list[1].shape)




        id_list=[]
        # print(i,'done.')
        for d in dict:
            if d['name'] == i:
                x1 = d['bbox'][0]
                y1 = d['bbox'][1]
                x2 = d['bbox'][2]
                y2 = d['bbox'][3]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                x1,x2,y1,y2=int(x1*w/pre_w),int(x2*w/pre_w),int(y1*h/pre_h),int(y2*h/pre_h)

                # 防止出现越界访问
                x2 = 2047 if x2==2048 else x2
                y2 = 1023 if y2==1024 else y2

                _x1,_x2,_y1,_y2=x1,x2,y1,y2

                square_src=(x2-x1)*(y2-y1)

                # 都在左侧
                if _x1<1024 and _x2<1024:
                    box_dict = {}

                    box_dict['name'] = i[0:-4] + '_' + str(0) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2-x1)*(y2-y1) > square_src*drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(0)

                # 两侧
                elif _x1<1024 and _x2>1024:
                    box_dict = {}
                    x1=_x1
                    x2=1023
                    box_dict['name'] = i[0:-4] + '_' + str(0) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]

                    if (x2-x1)*(y2-y1) > square_src*drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(0)

                    box_dict = {}


                    x2=_x2%1024
                    x1=0
                    box_dict['name'] = i[0:-4] + '_' + str(1) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]

                    if (x2-x1)*(y2-y1) > square_src*drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(1)
                elif _x1>=1024 and _x2>1024:
                    box_dict = {}

                    x1=_x1%1024
                    x2=_x2%1024
                    box_dict['name'] = i[0:-4] + '_' + str(1) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]

                    if (x2-x1)*(y2-y1) > square_src*drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(1)




        for id, image in enumerate(img_part_list):
            path = os.path.join(img_save_path, i[0:-4] + '_' + str(id) + '.jpg')
            # print(path)
            if id in id_list:
                cv2.imwrite(path, image)
        # return
    with open(json_save_path, 'w') as f:
        json.dump(dict_new, f)
def img_cut_4(img_save_path, json_save_path=r'D:\DeepLearning\dataset\cloth_cut\train\anno/anno.json'):
    print('img_cut_4')
    delete_file(dir_path=img_save_path)
    print('delete_file done.')
    anno_path = r"D:\DeepLearning\dataset\cloth_new\train\anno\anno.json"
    img_path = r"D:\DeepLearning\dataset\cloth_new\train\image"
    pre_h = 1024
    pre_w = 1024
    h = 1024
    w = 1024
    num_h = 2
    num_w = 2
    long_w=pre_w//num_w
    long_h=pre_h//num_h

    drop_rate=0.35

    img_list = os.listdir(img_path)
    print("all:",len(img_list))
    dict_new=[]

    with open(anno_path, 'r') as f:
        dict = json.load(f)
    random.shuffle(img_list)
    # img_list=['005bcbfd126fb67c1413297296.jpg','00fc63aff57def8d0947060525.jpg']

    for idx,i in enumerate(tqdm.tqdm(img_list)):
        img = cv2.imread(os.path.join(img_path, i))
        # print(os.path.join(img_path,i))
        # print(img)
        # img=np.array(img)
        img_part_list = []
        img_part_list.append(img[0:512,0:512].copy())
        img_part_list.append(img[0:512,512:1024].copy())
        img_part_list.append(img[512:1024,0:512].copy())
        img_part_list.append(img[512:1024,512:1024].copy())
        # print(img_part_list[0].shape)
        # print(img_part_list[1].shape)


        id_list=[]
        # print(i,'done.')
        for d in dict:
            if d['name'] == i:


                x1 = d['bbox'][0]
                y1 = d['bbox'][1]
                x2 = d['bbox'][2]
                y2 = d['bbox'][3]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # 防止出现越界访问
                x2 = 2047 if x2==2048 else x2
                y2 = 1023 if y2==1024 else y2

                _x1,_x2,_y1,_y2=x1,x2,y1,y2
                # print([x1, y1, x2, y2])
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)



                square_src=(x2-x1)*(y2-y1)

                # 左上角
                if _x1<long_w and _x2<long_w and _y1<long_h and _y2<long_h:
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(0) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2 - x1) * (y2 - y1) > square_src * drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(0)

                # 右上角
                elif _x1>long_w and _x2>long_w and _y1<long_h and _y2<long_w:
                    x1=_x1%long_w
                    x2=_x2%long_w
                    y1=_y1
                    y2=_y2
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(1) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2 - x1) * (y2 - y1) > square_src * drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(1)



                elif _x1<long_w and _x2<long_w and _y1>long_h and _y2>long_h:
                    y1=_y1%long_h
                    y2=_y2%long_h
                    x1=_x1
                    x2=_x2
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(2) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2 - x1) * (y2 - y1) > square_src * drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(2)


                elif _x1>long_w and _x2>long_w and _y1>long_h and _y2>long_h:
                    y1 = _y1 % long_h
                    y2 = _y2 % long_h
                    x1 = _x1 % long_w
                    x2 = _x2 % long_w
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(3) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2 - x1) * (y2 - y1) > square_src * drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(3)

                # 横跨
                elif (_x1<long_w and _x2>long_w and _y1<long_h and _y2<long_h):
                    y1 = _y1%long_h
                    y2 = _y2%long_h
                    x1 = _x1
                    x2 = long_w-1
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(0) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2 - x1) * (y2 - y1) > square_src * drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(0)

                    y1 = _y1 % long_h
                    y2 = _y2 % long_h
                    x1 = 1
                    x2 = _x2%long_w
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(1) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2 - x1) * (y2 - y1) > square_src * drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(1)
                elif  (x1<long_w and _x2>long_w and _y1>long_h and _y2>long_h):
                    y1 = _y1%long_h
                    y2 = _y2%long_h
                    x1 = _x1
                    x2 = long_w-1
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(2) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2 - x1) * (y2 - y1) > square_src * drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(2)

                    y1 = _y1 % long_h
                    y2 = _y2 % long_h
                    x1 = 1
                    x2 = _x2%long_w
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(3) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2 - x1) * (y2 - y1) > square_src * drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(3)

                # 竖跨
                elif _x1<long_w and _x2<long_w and _y1<long_h and _y2>long_h:
                    y1 = _y1 % long_h
                    y2 = long_h-1
                    x1 = _x1 % long_w
                    x2 = _x2 % long_w
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(0) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2 - x1) * (y2 - y1) > square_src * drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(0)

                    y1 = 1
                    y2 = _y2%long_h
                    x1 = _x1 % long_w
                    x2 = _x2 % long_w
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(2) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2 - x1) * (y2 - y1) > square_src * drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(2)

                elif _x1>long_w and _x2>long_w and _y1<long_h and _y2>long_h:
                    y1 = _y1 % long_h
                    y2 = long_h-1
                    x1 = _x1 % long_w
                    x2 = _x2 % long_w
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(1) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2 - x1) * (y2 - y1) > square_src * drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(1)

                    y1 = 1
                    y2 = _y2%long_h
                    x1 = _x1 % long_w
                    x2 = _x2 % long_w
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(3) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2 - x1) * (y2 - y1) > square_src * drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(3)


                # # 四周型
                elif _x1<long_w and _y1<long_h and _x2>long_w and _y2>long_h:
                    y1 = _y1 % long_h
                    y2 = long_h - 1
                    x1 = _x1 % long_w
                    x2 = long_w-1
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(0) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2 - x1) * (y2 - y1) > square_src * drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(0)

                    y1 = y1%long_h
                    y2 = long_h-1
                    x1 = 1
                    x2 = _x2 % long_w
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(1) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2 - x1) * (y2 - y1) > square_src * drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(1)

                    y1 = 1
                    y2 = _y2%long_h
                    x1 = _x1%long_w
                    x2 = long_w-1
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(2) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2 - x1) * (y2 - y1) > square_src * drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(2)

                    y1 = 1
                    y2 = _y2 % long_h
                    x1 = 1
                    x2 = _x2 % long_w
                    box_dict = {}
                    box_dict['name'] = i[0:-4] + '_' + str(3) + '.jpg'
                    box_dict['defect_name'] = d['defect_name']
                    box_dict['bbox'] = [x1, y1, x2, y2]
                    if (x2 - x1) * (y2 - y1) > square_src * drop_rate:
                        dict_new.append(box_dict)
                        id_list.append(3)


        for id, image in enumerate(img_part_list):
            path = os.path.join(img_save_path, i[0:-4] + '_' + str(id) + '.jpg')
            # print(path)
            if id in id_list:
                cv2.imwrite(path, image)


        # cv2.rectangle(img, (511, 512, 0, 1024), (255, 0, 0), 5)
        if 0 in id_list:
            cv2.rectangle(img, (0, 0, 512, 512), (0, 255, 0), 8)
        if 1 in id_list:
            cv2.rectangle(img, (512, 0, 1024, 512), (0, 255, 0), 8)
        if 2 in id_list:
            cv2.rectangle(img, (0, 512, 512, 1024), (0, 255, 0), 8)
        if 3 in id_list:
            cv2.rectangle(img, (512, 512, 1024, 1024), (0, 255, 0), 8)
        cv2.line(img, (0, 512),( 1024, 512), (0, 0, 255), 2)
        cv2.line(img, (512, 0),( 512, 1024), (0, 0, 255), 2)
        img = cv2.resize(img, (600, 600))

        # cv2.imshow('window', img)
        # cv2.waitKey()



        # return


    with open(json_save_path, 'w') as f:
        json.dump(dict_new, f)



def enhance(img_path,json_path,classes_list,flip_rate=0.5, w=512, h=512):
    print('data_enhance')
    img_list = os.listdir(img_path)
    print("all:",len(img_list))
    dict_new=[]

    with open(json_path, 'r') as f:
        dict = json.load(f)

    # img_list=['005bcbfd126fb67c1413297296.jpg','00fc63aff57def8d0947060525.jpg']

    for idx,i in enumerate(tqdm.tqdm(img_list)):
        for D in dict:
            if D['name']==i and D['defect_name'] in classes_list:

                img = cv2.imread(os.path.join(img_path, i))
                if np.random.rand()<flip_rate:
                    new = {}

                    new['name'] = D['name'][0:-4] + '_e.jpg'
                    new['defect_name'] = D['defect_name']
                    # 水平翻转
                    if 0.5<np.random.rand()<1:
                        a = w - D['bbox'][2]+1
                        b = D['bbox'][1]
                        c = w - D['bbox'][0]-1
                        d = D['bbox'][3]
                        bbox = [a, b, c, d]
                        new['bbox'] = bbox
                        img = cv2.flip(img, 1)

                    # 垂直翻转
                    else:
                        a = D['bbox'][0]
                        b = h - D['bbox'][3]+1
                        c = D['bbox'][2]
                        d = h - D['bbox'][1]-1
                        bbox = [a, b, c, d]
                        new['bbox'] = bbox
                        img = cv2.flip(img, 0)


                    dict_new.append(new)
                    # print(new['name'])
                    path = os.path.join(img_path, new['name'])
                    # print(path)
                    cv2.imwrite(path, img)


                # print(new['name'], 'done. ')
    for i in dict_new:
        dict.append(i)
    with open(json_path, 'w') as f:
        json.dump(dict, f)
    # print(new)

def test():
    img = cv2.imread('../img/12d7f68ed19b67f11230379432.jpg')
    print(img[999][999][0])
        # break
def data_transform(img_save_path, json_save_path,trans_dict):
    print('data_transform')
    delete_file(dir_path=img_save_path)
    print('delete_file done.')
    anno_path = r"D:\DeepLearning\dataset\Cloth\train2_datasets\guangdong1_round1_train2_20190828\Annotations\anno_train.json"
    img_path = r"D:\DeepLearning\dataset\Cloth\train2_datasets\guangdong1_round1_train2_20190828\defect_Images"

    dict_new = []
    classes_num = {}

    class_list=[]
    class_new_num={}
    for i in trans_dict:
        class_list.append(i)
    for i in class_list:
        classes_num[i] = 0
    for i in trans_dict:
        class_new_num[trans_dict[i]]=0
    with open(anno_path, 'r') as f:
        dict = json.load(f)
    print(dict[0])

    img_list = os.listdir(img_path)
    for idx,p in enumerate(tqdm.tqdm(img_list)):
        for i in dict:
            if i['name'] == p and i['defect_name'] in class_list:
                classes_num[i['defect_name']] += 1

                i['defect_name'] = trans_dict[i['defect_name']]
                class_new_num[i['defect_name']]+=1


                name = i['name']
                pre_path = os.path.join(img_path, name)
                fin_path = os.path.join(img_save_path, name)
                copyfile(pre_path, fin_path)
                # i['defect_name'] = eng_name_dict[i['defect_name']]
                dict_new.append(i)

    with open(json_save_path, 'w') as f:
        json.dump(dict_new, f)
    print(classes_num)
    print(class_new_num)
    print(len(dict_new))

    # for i in dict:
    #     if i['defect_name'] in class_list:
    #
    #         img_id=str(classes_num[i['defect_name']]).zfill(5)
    #         i['name']=eng_name_dict[i['defect_name']]+'_'+img_id+'.jpg'
    #
    #         pre_path=os.path.join(img_path,name)
    #         fin_path=os.path.join(img_save_path,i['name'])
    #         # copyfile(pre_path, fin_path)
    #         classes_num[i['defect_name']]+=1
    #
    #         i['defect_name'] = eng_name_dict[i['defect_name']]
    #         dict_new.append(i)
    #         print(i['defect_name'],i['name'],'done.')
    # with open(json_save_path, 'w') as f:
    #     json.dump(dict_new, f)
    # print(classes_num)
def json2xml(pic_path,json_path,xml_path):
    print('json2xml')
    delete_file(xml_path)
    print('delete_file done.')
    def __indent(elem, level=0):
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                __indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
    with open(json_path, 'r') as f:
        dict = json.load(f)

    img_paths = os.listdir(pic_path)
    # print(img_paths)


    for idx,img_path in enumerate(tqdm.tqdm(img_paths)):
        save=True
        # img_path=os.path.join(PIC_PATH,img_path)
        xml_name=img_path
        root = ET.Element('annotation')
        tree = ET.ElementTree(root)

        folder = ET.Element("folder")
        folder.text = 'JPEGImages'

        filename = ET.Element("filename")
        filename.text = img_path

        path = ET.Element("path")
        path.text = str(os.path.join(pic_path, img_path))

        sourse = ET.Element("sourse")

        database = ET.Element("database")
        database.text = 'Unknown'
        sourse.append(database)

        img = cv2.imread(os.path.join(pic_path, img_path))

        size = ET.Element("size")
        width = ET.Element("width")
        height = ET.Element("height")
        depth = ET.Element("depth")

        width.text = str(img.shape[1])
        height.text = str(img.shape[0])
        depth.text = str(img.shape[2])

        size.append(width)
        size.append(height)
        size.append(depth)

        segmented = ET.Element("segmented")
        segmented.text = '0'

        root.append(folder)
        root.append(filename)
        root.append(path)
        root.append(sourse)
        root.append(size)
        root.append(segmented)

        for label in dict:
            if label.get('name')==img_path:

                name=label.get('defect_name')
                # print(label,name)
                box = label.get('bbox')
                if box[0]<0 or box[1]<0 or box[2]<0 or box[3]<0:
                    save=False
                    break

                obj = ET.Element("object")
                class_name=ET.Element("name")


                class_name.text=str(name)

                difficult=ET.Element("difficult")
                difficult.text='0'
                bndbox=ET.Element("bndbox")

                xmin = ET.Element("xmin")
                ymin = ET.Element("ymin")
                xmax = ET.Element("xmax")
                ymax = ET.Element("ymax")

                xmin.text=str(int(box[0]))
                ymin.text=str(int(box[1]))
                xmax.text=str(int(box[2]))
                ymax.text=str(int(box[3]))

                bndbox.append(xmin)
                bndbox.append(ymin)
                bndbox.append(xmax)
                bndbox.append(ymax)

                pose= ET.Element("pose")
                pose.text='Unspecified'

                truncated=ET.Element("truncated")
                truncated.text='0'

                obj.append(class_name)
                obj.append(pose)
                obj.append(truncated)
                obj.append(difficult)
                obj.append(bndbox)

                root.append(obj)
        __indent(root)
        # tree.write(xml_path+xml_name[0:-4]+".xml", encoding='utf-8', xml_declaration=True)
        if save:
            tree.write(os.path.join(xml_path,xml_name[0:-4])+".xml", encoding='utf-8', xml_declaration=True)
        else:
            print(xml_name)
        # print()


        # break

def rh(src1,src2):

    src1=cv2.imread(src1)
    src2=cv2.imread(src2)
    print(src2.shape)
    src1=cv2.resize(src1,(src2.shape[0],src2.shape[1]))

    dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)
    # dst = src1 * 1 + src2 * 1 + 0
    cv2.imshow('ss',dst)
    cv2.waitKey()

if __name__ == '__main__':
    # Cloth -> cloth_cut -> cloth_new -> cloth_512

    # rh(r"D:\DeepLearning\yolov7-pytorch-master\yolov7-pytorch-master\image_512\heatmap_0\4f70a30b5c8b607b1434233482_1_3.jpg",
    #    r"D:\DeepLearning\yolov7-pytorch-master\yolov7-pytorch-master\image_512\result_0\4f70a30b5c8b607b1434233482_1_3.png")

    # eng_name_dict={'结头':'jietou','破洞':'podong','三丝':'sansi','星跳':'xingtiao','整经结':'zhengjingjie','跳花':'tiaohua','粗经':'cujing'}
    # trans_dict={'结头':'hole','破洞':'hole','整经结':'hole','毛粒':'hole','断氨纶':'hole',
    #
    #             '粗维':'line','三丝':'line','纬缩':'line','粗经':'line','纬纱不良':'line','修痕':'line','断经':'line',
    #             '跳花':'line','星跳':'line','百脚':'line','松经':'line','吊经':'line','轧痕':'line','双纬':'line',
    #             '双经':'line','跳纱':'line','筘路':'line','花板跳':'line',
    #
    #             '水渍':'stain','油渍':'stain','污渍':'stain','磨痕':'stain','云织':'stain',
    #
    #             '浆斑':'float','烧毛痕':'float',}
    trans_dict={'结头':'hole','破洞':'hole',
                    '粗维':'line','三丝':'line','粗经':'line',#'纬缩':'line',
                '水渍':'stain','油渍':'stain','污渍':'stain',
                    '浆斑':'float',}
    # '色差档': 'float', '烧毛痕': 'float', '磨痕': 'float'
    # '烧毛痕': 'float', '磨痕': 'float'
    # data_transform(img_save_path=r'D:\DeepLearning\dataset\cloth_test\img',json_save_path=r'D:\DeepLearning\dataset\cloth_test\anno\anno.json',trans_dict=trans_dict)
    # img_cut_2(img_save_path=r'D:\DeepLearning\dataset\cloth_cut\train\img')
    # img_cut_4(img_save_path=r'D:\DeepLearning\dataset\cloth_512\img',json_save_path=r'D:\DeepLearning\dataset\cloth_512\anno\anno.json')

    show_dataset(img_path=r'D:\DeepLearning\dataset\cloth_test\img',anno_path=r"D:\DeepLearning\dataset\cloth_test\anno\anno.json",image_size=(1500,600))

    JSON_PATH = r'D:\DeepLearning\dataset\cloth_new\train\anno\anno.json'
    PIC_PATH = r'D:\DeepLearning\dataset\cloth_new\train\image'
    # test()
    # enhance(img_path=r'D:\DeepLearning\dataset\cloth_512\img', json_path=r'D:\DeepLearning\dataset\cloth_512\anno\anno.json', classes_list=['float'], flip_rate=1, w=512, h=512)
    # json2xml(pic_path=r'D:\DeepLearning\dataset\cloth_512\img', json_path=r'D:\DeepLearning\dataset\cloth_512\anno\anno.json', xml_path=r'D:\DeepLearning\dataset\cloth_512\xml')
