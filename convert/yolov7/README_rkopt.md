# YOLOv7 - RKNN optimize


## Source

  Base on https://github.com/WongKinYiu/yolov7  with commit id as 44d8ab41780e24eba563b6794371f29db0902271



## What different

With inference result values unchanged, the following optimizations were applied:

- Optimize SPP block, getting better performance with same result
- Change output node, remove post_process from the model. (post process block in model is unfriendly for quantization)



## How to use

```
python export.py --rknpu --weight yolov7-tiny.pt
```

- Replace 'yolov7-tiny.pt' with your model path
- A file name "RK_anchors.txt" would be generated and it could be use during doing post_process in the outside. 
- **NOTICE: Please call with --rknpu param, do not changing the default rknpu value in export.py.** 



## Deploy demo

Please refer https://github.com/airockchip/rknn_model_zoo
