import json
import os
import xml.etree.ElementTree as ET

import cv2
import tqdm


def json2xml(pic_path,json_path,xml_path):
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
        tree.write(os.path.join(xml_path,xml_name[0:-4])+".xml", encoding='utf-8', xml_declaration=True)
        # print()


        # break

JSON_PATH = r'D:\DeepLearning\dataset\cloth_new\train\anno\anno.json'
PIC_PATH = r'D:\DeepLearning\dataset\cloth_new\train\image'

CLASS_DICT={'破洞':1,'水渍':2,'油渍':2,'污渍':2,'三丝':3,'结头':4,'花板跳':5,'百脚':6,'毛粒':7,'粗经':8,'松经':9,'断经':10,'吊经':11,
            '粗维':12,'纬缩':13,'浆斑':14,'整经结':15,'星跳':16,'跳花':16,'断氨纶':17,'稀密档':18,'浪纹档':18,'色差档':18,'磨痕':19,
            '轧痕':19,'修痕':19,'烧毛痕':19,'死皱':20,'云织':20,'双纬':20,'双经':20,'跳纱':20,'筘路':20,'纬纱不良':20}

# 结头,破洞,三丝,断氨纶,粗维,粗经,花板跳,,
CLASS_DICT2={}

with open(JSON_PATH, 'r') as f:
    dict = json.load(f)
# with open(JSON_PATHb, 'r') as f:
#     dictB = json.load(f)

# pic_paths=os.listdir(PIC_PATH)
# print(pic_paths)
print(len(dict))
print(dict[0])
name=[]
num={}
for idx,i in enumerate(dict):
    pic_path=i.get('name')
    class_name = i.get('defect_name')
    if class_name not in name:
        num[class_name]=0
        name.append(class_name)
        CLASS_DICT2[class_name]=class_name
    num[class_name]+=1

print(CLASS_DICT2)
for i in name:
    print(i)
# print(len(num))
# print(sorted(num.items(), key=lambda x: x[1],reverse=True))

# print(len(CLASS_DICT))
json2xml(pic_path=PIC_PATH,json_path=JSON_PATH,xml_path=r'D:\DeepLearning\dataset\cloth_new\train\xml')