# YOLOv7 - RKNN optimize


## Source

  Base on https://github.com/WongKinYiu/yolov7  with commit id as 44d8ab41780e24eba563b6794371f29db0902271



## What different

Inference result unchanged:

- Optimize SPP block, getting better performance with same result
- Change output node, remove post_process from the model. (post process is unfriendly in quantization)



## How to use

```
python export.py --rknpu {rk_platform} --weight yolov7-tiny.pt
```

- rk_platform support  rk1808, rv1109, rv1126, rk3399pro, rk3566, rk3562, rk3568, rk3588, rv1103, rv1106. (Actually the exported models are the same in spite of the exact platform )

- Replace 'yolov7-tiny.pt' with your model path
- A file name "RK_anchors.txt" would be generated and it could be use during doing post_process in the outside. 
- **NOTICE: Please call with --rknpu param, do not changing the default rknpu value in export.py.** 

