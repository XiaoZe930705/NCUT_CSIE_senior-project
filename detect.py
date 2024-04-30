# YOLOv5 ğŸš€ by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch   
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import torch
import time
from pathlib import Path
from datetime import datetime, timedelta
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from PIL import Image, ImageDraw

# 創建 CSV 檔案
csv_path = 'detections.csv'
header = ['Date', 'Time', 'Persons Count']  # 表頭欄位
def create_csv(file_path, header):
    #檢查檔案是否存在
    csv_path = 'detections.csv'
    if not os.path.isfile(csv_path):
        try:
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)  # 寫入 CSV 標頭
            print(f"CSV 檔 {csv_path} 創建成功！")
        except Exception as e:
            print(f"CSV 檔 {csv_path} 創建失敗，錯誤訊息：{e}")


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print("Left button clicked at ({}, {})".format(x, y))
        cv2.destroyAllWindows()  # 關閉窗口
        exit()  # 結束程式

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # 模型路徑或 Triton URL
        source=ROOT / 'data/images',  # 檔案/目錄/URL/過濾/螢幕/0(網路攝影機)#source=ROOT / 'data/images',
        data=ROOT / 'data/coco128.yaml',  # 資料集.yaml路徑
        imgsz=(640, 640),  # 推論尺寸（高度，寬度）#imgsz=(640, 640)
        conf_thres=0.25,  # 置信度閾值
        iou_thres=0.45,  # NMS IoU 閾值
        max_det=1000,  # 每張圖片的最大偵測數量
        device='',  # CUDA 設備，例如 0 或 0,1,2,3 或 CPU
        view_img=False,  # 顯示結果TRUE為鏡頭模式
        save_txt=False,  # 儲存結果為 *.txt
        save_csv=False,  # 儲存結果為 CSV 格式
        save_conf=False,  # 在 --save-txt 標籤中儲存置信度
        save_crop=False,  # 儲存裁剪的預測框
        nosave=False,  # 不儲存影像/影片
        classes=None,  # 依類別篩選：--class 0 或 --class 0 2 3
        agnostic_nms=False,  # 類別不可知的 NMS
        augment=False,  # 增強推論
        visualize=False,  # 視覺化特徵
        update=False,  # 更新所有模型
        project=ROOT / 'runs/detect',  # 儲存結果的專案/名稱
        name='exp',  # 儲存結果的專案/名稱
        exist_ok=False,  # 存在的專案/名稱是可接受的，不增加
        line_thickness=3,  # 邊界框厚度（像素）
        hide_labels=False,  # 隱藏標籤
        hide_conf=False,  # 隱藏置信度
        half=False,  # 使用 FP16 半精度推論
        dnn=False,  # 使用 OpenCV DNN 進行 ONNX 推論
        vid_stride=1,  # 影片幀率步長
        save_dir=Path(''), # 文件保存路径 如果执行val.py就为‘’ , 如果执行train.py就会传入save_dir(runs/train/expn)
        csv_path='detections.csv',  # 這裡添加 csv_path 參數
):
    #使用者輸入空間密度
    while True:
        try:
            spatial_density = float(input("請輸入此區域空間密度(平方公尺): "))
            while(spatial_density == 0):
                    print("請勿輸入無效參數!!")
                    spatial_density = float(input("請輸入此區域空間密度(平方公尺):"))
            break
        except ValueError:
            print("您输入的不是有效的数字，请重新输入。")
            continue
    #詢問使用者是否要調整預設容納人數(變更1，無變更0)")
    question =str(input("請問是否需要變更每平方公尺容納人數參數(需要變更請輸入1，無須變更請輸入0):"))
    while True:
        if(question == '1'):
            while True:
                try:
                    limit_the_number_of_people = int(input("請輸入每平方公尺容納人數(預設值為:3):"))
                    while(limit_the_number_of_people == 0):
                        print("請勿輸入無效參數!!")
                        limit_the_number_of_people = int(input("請輸入每平方公尺容納人數(預設值為:3):"))
                    break
                except ValueError:
                    print("您输入的不是有效的数字，请重新输入。")
                    continue
            break
        elif(question == '0'):
            limit_the_number_of_people =2
            break
        else:
            print("請輸入正確數值")
            question =str(input("請問是否需要變更每平方公尺容納人數計算(需要變更請輸入1,無須變更請輸入0)"))

    create_csv(csv_path, header)  # 在程式碼的開始處呼叫 create_csv 函式
    print("Saving results to:", save_dir) 
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        cv2.namedWindow("Webcam Viewer")
        cv2.setMouseCallback("Webcam Viewer", on_mouse_click)#關閉視窗
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    
    #檢查檔案是否存在
    csv_path = 'detections.csv'
    if not os.path.isfile(csv_path):
        try:
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Date', 'Time', 'Persons Count'])  # 寫入 CSV 標頭
            print(f"CSV 檔 {csv_path} 創建成功！")
        except Exception as e:
            print(f"CSV 檔 {csv_path} 創建失敗，錯誤訊息：{e}")

    n = 0      
    for path, im, im0s, vid_cap, s in dataset:
        
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        #pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            
            # 计算特定类别（例如人）的数量
            persons_count = (det[:, 5] == 0).sum()  # 假设人的类别索引是 0，根据需要进行修改
            
            current_datetime = datetime.now()  # 獲得當前的日期和時間
            current_date = current_datetime.date()  # 當天日期
            current_time = current_datetime.time() # 當天時間
            if webcam:  
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
                # 更新時間
                current_datetime = datetime.now()
                with open(csv_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([current_date, current_time, persons_count])
                
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                
                current_datetime = datetime.now()  # 獲得當前的日期和時間
                if ((seen % 30)==0): #24
                    with open(csv_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([current_date, current_time, persons_count])
            #呼叫空間計算
            capacity =spatial_computing(spatial_density,limit_the_number_of_people)
            #壅塞、舒適判斷
            if(persons_count>=capacity):
                cv2.circle(im0,(20,20),20,(0,0,255),thickness=cv2.FILLED) #(來源,圓心,半徑,顏色,寬度) 
                #cv2.putText(im0, "Crowd congestion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1)  測試用
            else:
                cv2.circle(im0,(20,20),20,(0,255,),thickness=cv2.FILLED) #(來源,圓心,半徑,顏色,寬度)
             
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # 每個類別的偵測數量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 加入到字串中
                    # 取得特定類別（例如人）的數量
                    persons_count = (det[:, 5] == 0).sum()  # 假設人的類別索引是 0，根據需要進行修改#
                    # 調整文字寬度和高度
                    text = f"Persons: {persons_count}"
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=2)
                    text_width, text_height = text_size
                    
    
                    # 設定右上角的位置
                    margin = 10
                    x_pos = im0.shape[1] - text_width - margin
                    y_pos = text_height + margin

                    # 顯示文字 BGR（藍、綠、紅）的顏色設定為 (0, 0, 0)
                    cv2.putText(im0, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

                   
                 # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'


                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

                if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC鍵退出程式
                    cv2.destroyAllWindows()
                    exit()
                
            # 处理视频写入前检查並初始化 vid_path 列表
            if dataset.mode != 'image':  # 只有在處理影像時才进行以下操作
                while len(vid_path) <= i:  # 确保 vid_path 的长度足够支持当前索引 i
                    vid_path.append(None)  # 在 vid_path 列表末尾添加元素，直到列表长度大于等于 i
                    
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    while len(vid_path) <= i:
                        vid_path.append(None)  # 或者根据需要添加适当的默认值

                    while len(vid_writer) <= i:
                        vid_writer.append(None)  # 或者根据需要添加适当的默认值
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='0',help='file/dir/URL/glob/screen/0(webcam)')#default=ROOT / 'data/images'，default='0'
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int,default=[640],  help='inference size h,w')#default=[640]
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=True ,help='show results')#顯示即時影像 #action='store_false'、 default=True,
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', default=True,help='save results in CSV format')  #CSV default=True
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,default='0', help='filter by class: --classes 0, or --classes 0 2 3')  #default='0'為只檢測人
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',default=True, help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))
#空間計算
def spatial_computing(spatial_density,limit_the_number_of_people):
    result = spatial_density*limit_the_number_of_people
    return result
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    # 計算影片檔案的幀率
    folder_path = 'C:\\research-data\\yolov5-master\\data\\images'  # 修改為你的資料夾路徑
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mov')]
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0.0:
            # 如果無法獲得正確的幀率，可以使用一個預設值
            fps = 30.0  # 這裡使用 30.0 做為預設值
        print(f"影片 {video_file} 的幀率: {fps}")
        video.release()