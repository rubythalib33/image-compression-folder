# Image Compression Shrink Binary Compression and EfficientSRGAN for low lattency of Recovery

so the idea is to change residual block from resnet to MBConvolution block from Efficient net, and it's result amazingly the model become 5x more small and the inference speed is become 3-4x then the previous method

Model Name | Params | GFlops 
--- | --- | --- 
SRGAN | 1.549 M | 9.128 GFlops
EfficientSRGAN | 0.319 M | 2.102 GFlops

## Sample of the recovered Image
![](./asset/0882x4.png)