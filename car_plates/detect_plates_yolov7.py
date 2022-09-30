import torch

model = torch.hub.load('yolov7', 'custom', path='yolov7_out/best_47eph.pt', source='local')

im = 'car1.jpg'

# Inference
results = model(im)
results.save()
print(results.pandas().xyxy[0])


