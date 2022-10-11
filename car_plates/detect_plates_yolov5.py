import torch
import glob


def inference_yolov5(image, weights_path):
    """
    Load trained PyTorch yolov5 model to recognize car plates on image

    :param image: Path to image
    :param weights_path: Path to weights
    :return: A list of paths to found plates
    """
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

    results = model(image)

    # results.save()
    results.crop(save=True)
    print(results.pandas().xyxy[0])

    # print(results.crop()[0]['im'][0])
    detections = glob.glob("runs/detect/exp/crops/carplate/*")
    print(detections)

    return detections
