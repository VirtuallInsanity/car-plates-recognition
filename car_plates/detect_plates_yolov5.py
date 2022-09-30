import torch

model = torch.hub.load('yolov5', 'custom', path='yolov5_out/best_100eph.pt', source='local')

im = 'car1.jpg'

# Inference
results = model(im)
results.save()
print(results.pandas().xyxy[0])