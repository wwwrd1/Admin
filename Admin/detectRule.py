from ultralytics import YOLO
import os

def calculate_iou(box1, box2):
    # box1: (x1, y1, x2, y2), box2: (x1, y1, x2, y2)
    x_inter = max(box1[0], box2[0])
    y_inter = max(box1[1], box2[1])
    x_inter_prime = min(box1[2], box2[2])
    y_inter_prime = min(box1[3], box2[3])
    # 交集宽高
    w_inter = max(0, x_inter_prime - x_inter)
    h_inter = max(0, y_inter_prime - y_inter)
    # 交集面积
    area_inter = w_inter * h_inter

    return area_inter*1.0/(box2[2]-box2[0])*(box2[3]-box2[1])


def judge(results): # 输入model.predict()结果，返回对应结果列表
    res = []
    for result in results:
        boxes = result.boxes
        cls = boxes.cls.int().tolist()
        if len(cls) <= 2: # 3个以上目标才可能违规
            res.append(False)
            continue
        if 0 not in cls:
            res.append(False)
            continue
        car = None
        for i in range(len(cls)):
            if cls[i] == 0:
                car = boxes.xyxy[i]
                break
        cnt = 0
        for i in range(len(cls)):
            if cls[i] == 0:
                continue
            tbox = boxes.xyxy[i]
            if calculate_iou(car, tbox) >= threshold:
                cnt += 1
        if cnt >= 2:
            res.append(True)
        else:
            res.append(False)
    return res

standard = dict()
standard_path = './datasets/detect_table.txt'
with open(standard_path, "r") as file:
    for line in file:
        name, ans = line.strip().split()
        name = name + '.jpg'
        ans = bool(int(ans))
        standard[name] = ans

testdir = './datasets/tricycle/val/images/'
allpic = os.listdir(testdir)
threshold = 0.5

model = YOLO("xx/best.pt")
results = model.predict(source=testdir, conf=0.45)
res = judge(results)
corr = 0

for ans, result in zip(res,results):
    if standard[os.path.basename(result.path)] == ans:
        corr += 1

print("Precicsion:%.4f%%" % (corr*100.0/len(res)))