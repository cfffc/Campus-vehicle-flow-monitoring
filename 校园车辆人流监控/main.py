#1.准备tracker和detector 2.计算上下方向人数  3.添加功能碰撞 4.添加功能测速 5.绘制ui界面 6.打包成exe文件（使用pyinstaller）
#绘制图像
# collision_list初始化位置有问题 碰撞有重复
#速度计算未找到合适的k
#没有图像绘制
#可以补充发生碰撞的时间
import math
import tracker
from detector import Detector
import cv2

if __name__ == '__main__':
    #绘制准备
    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX

    # 初始化检测器
    detector = Detector()

    #设置计算速度参数
    k=1.25
    v_a=4
    # v_a = 2.5
    # v_a = 2.5
    # v_a = 1
    v_b=0.5
    # v_b = 1

    # 用于控制碰撞检测
    distance_error = 10

    # 打开视频和存储视频
    capture = cv2.VideoCapture(r'./video/school_2.mp4')
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("!!!!!!size",size)

    output_video = cv2.VideoWriter("./output.mp4", -1,fps, (960,540))
    # capture = cv2.VideoCapture('TownCentreXVID.avi')
    # out = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30)

    list_bboxs = []
    # 设置上一帧的list_box，用于计算速度
    list_bboxs_2 = []
    i = 1

    while True:
        print(f'第{i}帧！！！！！！！！！！！！！')
        # 读取图片
        _, im = capture.read()
        if im is None:
            break

        #调整大小，便于计算
        im = cv2.resize(im, (960, 540))

        #通过检测器检测出目标
        #返回四个点坐标 类别和置信度
        bboxes = detector.detect(im)

        #如果存在检测对象用tracker更新到list里面并且进行绘制方框，绘制到output中
        #tracker主要有两个功能，一是用deepsort算法匹配而是用cv画框
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
            pass
        else:
            output_image_frame = im
        pass

        colision_list = []

        #在已经跟踪到的list中处理撞线
        if len(list_bboxs) > 0:

            #一次循环处理
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox
                #修正一下实际碰框的点
                y1_offset = int(y1 + ((y2 - y1) * 0.6))
                y = y1_offset
                x = x1
                #print(f'序号为：{track_id}的{label}的高度是{y2-y}')
                # print(f'y2-y1={y2-y1}')
                # print(f'x2-x1={x2-x1}')

        #在最后更新list,用于存储上一帧的内容
        list_bboxs_2 = tracker.update(bboxes, im)

        #print(output_image_frame.shape)
        output_video.write(output_image_frame)
        cv2.imshow('demo', output_image_frame)
        i+=1
        cv2.waitKey(1)
        pass

    #释放资源
    capture.release()
    cv2.destroyAllWindows()
