import torch
import glob


def inference_yolov7(image, weights_path='car_plates/yolov7_out/best_47eph.pt'):
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model=weights_path)

    results = model(image)

    results.save()
    results.crop(save=True)
    print(results.pandas().xyxy[0])

    detections = glob.glob("runs/detect/exp2/crops/carplate/*")
    print(detections)

    return detections
