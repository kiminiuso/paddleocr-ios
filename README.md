# 仿照官方代码移植了一下paddle lite 到ios


PaddleOCR 官方iOSdemo，2.1没有，只能参照2.1的android demo和1.1的ios demo写了一下

https://github.com/PaddlePaddle/PaddleOCR/tree/release/1.1/deploy/ios_demo

opencv2是从官方拉了最新的源码编译的，具体方法看:

https://docs.opencv.org/master/d5/da3/tutorial_ios_install.html

opencv2引入后还需要在Frameworks，Library，and Emebedded 里引入CoreMedia和CoreVideo库，不然会报一堆opencv Undefined symbol:_AVxxxxx


有几个问题还需解决：用 [这里](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.1/deploy/lite )提供的模型，输入识别模型以后，无输出，最后用了1.1的模型

1.1的模型在ios上摄像头实时检测效果不太行，图片检测也一般，整个包体积50M+

