import math


def at_boundary(result,th):
    for x in range(1, 5):
        if result[x] % 512 < th or result[x] % 512 > 512 - th:
            return True
    return False

def distance_point(x1,y1,x2,y2):
    return math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

def distance_line(x1,x2,th):
    if abs(x2-x1)<th*1.2:
        return True
    else:
        return False

# 判断是否相隔
def xiangge_x(i_y1, i_y2, j_y1, j_y2):
    i_y = (i_y1+i_y2)/2
    j_y = (j_y1+j_y2)/2

    # i在上面
    if i_y<j_y:
        return True if i_y2 < j_y1 else False
    # j在上面
    else:
        return True if j_y2 < i_y1 else False


def xiangge_y(i_x1, i_x2, j_x1, j_x2):
    i_x = (i_x1+i_x2)/2
    j_x = (j_x1+j_x2)/2

    # i在左侧
    if i_x<j_x:
        return True if i_x2 < j_x1 else False
    # j在左面
    else:
        return True if j_x2 < i_x1 else False



