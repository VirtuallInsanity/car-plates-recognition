import torch
import glob


def inference_yolov7(image, weights_path):
    """
    Load trained PyTorch yolov7 model to recognize car plates on image

    :param image: Path to image
    :param weights_path: Path to weights
    :return: A list of paths to found plates
    """
    model = torch.hub.load('WongKinYiu/yolov7', 'custom',
                           path_or_model=weights_path)

    results = model(image)

    # results.save()
    results.crop(save=True)
    print(results.pandas().xyxy[0])

    detections = glob.glob("runs/detect/exp/crops/carplate/*")
    print(detections)

    return detections


def load_model_yolov7(weights_path):
    """
        Load trained PyTorch yolov7 model to recognize car plates

        :param weights_path: Path to weights
        :return: A loaded model
        """
    return torch.hub.load('WongKinYiu/yolov7', 'custom', path=weights_path)
