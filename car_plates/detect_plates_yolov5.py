import torch
import glob


def inference_yolov5(image, weights_path='car_plates/yolov5_out/best_100eph.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

    results = model(image)

    results.save()
    results.crop(save=True)
    print(results.pandas().xyxy[0])

    # print(results.crop()[0]['im'][0])
    detections = glob.glob("runs/detect/exp2/crops/carplate/*")
    print(detections)

    return detections
