//
//  ViewController.m
//  CardRecognize
//
//  Created by kiminiuso on R 3/06/16.
//

//opencv放最前
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/imgproc/types_c.h>
#import <opencv2/videoio/cap_ios.h>


#import "ViewController.h"
#import "paddle_api.h"
#import "paddle_use_ops.h"
#import "paddle_use_kernels.h"
#import "timer.h"
#import "pdocr/ocr_db_post_process.h"
#import "pdocr/ocr_crnn_process.h"
#import "OcrData.h"
#import "BoxLayer.h"

#import <string>

#import "Utils.h"

using namespace paddle::lite_api;
using namespace cv;

std::shared_ptr<PaddlePredictor> net_det;
std::shared_ptr<PaddlePredictor> net_cls;
std::shared_ptr<PaddlePredictor> net_rec;

Timer tic;


cv::Mat resize_img_type0(const cv::Mat &img, int max_size_len, float *ratio_h, float *ratio_w) {
    int w = img.cols;
    int h = img.rows;

    float ratio = 1.f;
    int max_wh = w >= h ? w : h;
    if (max_wh > max_size_len) {
        if (h > w) {
            ratio = float(max_size_len) / float(h);
        } else {
            ratio = float(max_size_len) / float(w);
        }
    }

    int resize_h = int(float(h) * ratio);
    int resize_w = int(float(w) * ratio);
    if (resize_h % 32 == 0)
        resize_h = resize_h;
    else if (resize_h / 32 < 1)
        resize_h = 32;
    else
        resize_h = (resize_h / 32 - 1) * 32;

    if (resize_w % 32 == 0)
        resize_w = resize_w;
    else if (resize_w / 32 < 1)
        resize_w = 32;
    else
        resize_w = (resize_w / 32 - 1) * 32;

    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(resize_w, resize_h));

    *ratio_h = float(resize_h) / float(h);
    *ratio_w = float(resize_w) / float(w);
    return resize_img;
}

void neon_mean_scale(const float* din, float* dout, int size, std::vector<float> mean, std::vector<float> scale) {
    if (mean.size() != 3 || scale.size() != 3) {
        NSLog(@"[ERROR] mean or scale size must equal to 3");
        return;
      }

      float32x4_t vmean0 = vdupq_n_f32(mean[0]);
      float32x4_t vmean1 = vdupq_n_f32(mean[1]);
      float32x4_t vmean2 = vdupq_n_f32(mean[2]);
      float32x4_t vscale0 = vdupq_n_f32(scale[0]);
      float32x4_t vscale1 = vdupq_n_f32(scale[1]);
      float32x4_t vscale2 = vdupq_n_f32(scale[2]);

      float *dout_c0 = dout;
      float *dout_c1 = dout + size;
      float *dout_c2 = dout + size * 2;

      int i = 0;
      for (; i < size - 3; i += 4) {
        float32x4x3_t vin3 = vld3q_f32(din);
        float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
        float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
        float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
        float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
        float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
        float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
        vst1q_f32(dout_c0, vs0);
        vst1q_f32(dout_c1, vs1);
        vst1q_f32(dout_c2, vs2);

        din += 12;
        dout_c0 += 4;
        dout_c1 += 4;
        dout_c2 += 4;
      }
      for (; i < size; i++) {
        *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
        *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
        *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
      }
}

void fill_tensor_with_cvmat(const cv::Mat& img_in, Tensor& tout, int width, int height,
                            std::vector<float> mean, std::vector<float> scale, bool is_scale) {
    if (img_in.channels() == 4) {
        cv::cvtColor(img_in, img_in, CV_RGBA2RGB);
    }
    cv::Mat im;
    cv::resize(img_in, im, cv::Size(width, height), 0.f, 0.f);
    cv::Mat imgf;
    float scale_factor = is_scale? 1 / 255.f : 1.f;
    im.convertTo(imgf, CV_32FC3, scale_factor);
    const float* dimg = reinterpret_cast<const float*>(imgf.data);
    float* dout = tout.mutable_data<float>();
    neon_mean_scale(dimg, dout, width * height, mean, scale);
}

#pragma mark - OC start -

@interface ViewController ()<CvVideoCameraDelegate>

@property (nonatomic,strong) UIImageView* imageView;
@property (nonatomic,strong) UIImage* image;
@property (nonatomic,strong) CALayer *boxLayer;

@property (nonatomic,strong) CvVideoCamera *videoCamera;
@property (nonatomic) cv::Mat cvimg;
@property (nonatomic,strong) UIImageView *preView;

@property(nonatomic) std::vector<float> scale;
@property(nonatomic) std::vector<float> mean;

//中文文字标签
@property(nonatomic) NSArray *labels;

@property(nonatomic)NSTimeInterval step;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    [self.view setBackgroundColor:[UIColor whiteColor]];
    [self initPaddleLite];
    
//    [self loadImage];
//    cv::Mat originImage;
//    UIImageToMat(self.image, originImage);
//    [self detImage:originImage];
    [self cameraInit];
}

- (void)viewDidAppear:(BOOL)animated
{
    [super viewDidAppear:animated];
    [self.videoCamera start];
}

- (void)loadImage {
    _image = [UIImage imageNamed:@"ocr.png"];
    if (_image != nil) {
        printf("load image successed\n");
        [self.imageView setFrame:CGRectMake(0, 0, _image.size.width, _image.size.height)];
        self.boxLayer.frame = CGRectMake(0, 0, self.imageView.frame.size.width, self.imageView.frame.size.height);
        self.imageView.center = self.view.center;
        self.imageView.image = _image;
    } else {
        printf("load image failed\n");
    }
}

- (void)initPaddleLite
{
//    模型地址
    NSString *bundlePath = [[NSBundle mainBundle] bundlePath];
    std::string model_dir = std::string([bundlePath UTF8String]);
    
//    std::string paddle_detect_dir = model_dir + "/ch_ppocr_mobile_v2.0_det_opt.nb";
    std::string paddle_detect_dir = model_dir + "/ch_det_mv3_db_opt.nb";
    std::string paddle_classify_dir = model_dir + "/ch_ppocr_mobile_v2.0_cls_opt.nb";
//    std::string paddle_recognize_dir = model_dir + "/ch_ppocr_mobile_v2.0_rec_opt.nb";
    std::string paddle_recognize_dir = model_dir + "/ch_rec_mv3_crnn_opt.nb";
    
    NSString *labelPath = [[NSBundle mainBundle] pathForResource:@"ppocr_keys_v1" ofType:@"txt"];
    self.labels = [Utils readLabelsFromFile:labelPath];
    
    MobileConfig config1;
    config1.set_model_from_file(paddle_detect_dir);
    config1.set_power_mode(LITE_POWER_HIGH);
    config1.set_threads(4);
    net_det = CreatePaddlePredictor<MobileConfig>(config1);
    
//    MobileConfig config2;
//    config2.set_model_from_file(paddle_classify_dir);
//    config2.set_power_mode(LITE_POWER_HIGH);
//    config2.set_threads(4);
//    net_cls = CreatePaddlePredictor<MobileConfig>(config2);
    
    MobileConfig config3;
    config3.set_model_from_file(paddle_recognize_dir);
    config3.set_power_mode(LITE_POWER_HIGH);
    config3.set_threads(4);
    net_rec = CreatePaddlePredictor<MobileConfig>(config3);
    
    self.mean = {0.5f, 0.5f, 0.5f};
//    self.scale = {1.0f, 1.0f, 1.0f};
    self.scale = {0.5f,0.5f,0.5f};
}

- (void)detImage:(cv::Mat &)originImage
{
    
    NSArray *rec_out = [self ocr_infer:originImage];
    
    [self.boxLayer.sublayers makeObjectsPerformSelector:@selector(removeFromSuperlayer)];
    CGFloat h = _boxLayer.frame.size.height;
    CGFloat w = _boxLayer.frame.size.width;
    std::ostringstream result2;
    NSInteger cnt = 0;
    for (id obj in rec_out) {
        OcrData *data = obj;
        BoxLayer *singleBox = [[BoxLayer alloc] init];
        [singleBox renderOcrPolygon:data withHeight:h withWidth:w withLabel:YES];
        [_boxLayer addSublayer:singleBox];
        result2<<[data.label UTF8String] <<","<<data.accuracy<<"\n";
        cnt += 1;
    }
    
    printf("结束");
}

- (void)cameraInit
{
    self.preView = [[UIImageView alloc]initWithFrame:CGRectMake(0, 0, 360, 480)];
    self.preView.center = self.view.center;
    [self.view insertSubview:self.preView atIndex:0];
    self.videoCamera = [[CvVideoCamera alloc] initWithParentView:self.preView];
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    self.videoCamera.rotateVideo = 90;
    self.videoCamera.defaultFPS = 30;
}



#pragma mark - private -
- (void)processImage:(cv::Mat &)image {
    if( CACurrentMediaTime() - self.step >= 1 ){
        self.step = CACurrentMediaTime();
        dispatch_async(dispatch_get_main_queue(), ^{
            if (image.channels() == 4) {
                cvtColor(image, self->_cvimg, CV_RGBA2RGB);
            }
            [self detImage:self->_cvimg];
        });
    }
}

- (NSArray *) ocr_infer:(cv::Mat) originImage{
    int max_side_len = 960;
    float ratio_h{};
    float ratio_w{};
    cv::Mat image;
    cv::cvtColor(originImage, image, cv::COLOR_RGB2BGR);

    cv::Mat img;
    image.copyTo(img);
    

    img = resize_img_type0(img, max_side_len, &ratio_h, &ratio_w);
    cv::Mat img_fp;
    img.convertTo(img_fp, CV_32FC3, 1.0 / 255.f);

    std::unique_ptr<Tensor> input_tensor(net_det->GetInput(0));
    input_tensor->Resize({1, 3, img_fp.rows, img_fp.cols});
    auto *data0 = input_tensor->mutable_data<float>();
    const float *dimg = reinterpret_cast<const float *>(img_fp.data);
    
    neon_mean_scale(dimg, data0, img_fp.rows * img_fp.cols, self.mean, self.scale);
    tic.clear();
    tic.start();
    net_det->Run();
    std::unique_ptr<const Tensor> output_tensor(net_det->GetOutput(0));
    auto *outptr = output_tensor->data<float>();
    auto shape_out = output_tensor->shape();

    int64_t out_numl = 1;
    double sum = 0;
    for (auto i : shape_out) {
        out_numl *= i;
    }

    int s2 = int(shape_out[2]);
    int s3 = int(shape_out[3]);

    cv::Mat pred_map = cv::Mat::zeros(s2, s3, CV_32F);
    memcpy(pred_map.data, outptr, s2 * s3 * sizeof(float));
    cv::Mat cbuf_map;
    pred_map.convertTo(cbuf_map, CV_8UC1, 255.0f);

    const double threshold = 0.1 * 255;
    const double maxvalue = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);

    auto boxes = boxes_from_bitmap(pred_map, bit_map);

    std::vector<std::vector<std::vector<int>>> filter_boxes = filter_tag_det_res(boxes, ratio_h, ratio_w, image);


    cv::Point rook_points[filter_boxes.size()][4];

    for (int n = 0; n < filter_boxes.size(); n++) {
        for (int m = 0; m < filter_boxes[0].size(); m++) {
            rook_points[n][m] = cv::Point(int(filter_boxes[n][m][0]), int(filter_boxes[n][m][1]));
        }
    }

    NSMutableArray *result = [[NSMutableArray alloc] init];

    for (int i = 0; i < filter_boxes.size(); i++) {
        cv::Mat crop_img;
        crop_img = get_rotate_crop_image(image, filter_boxes[i]);
        OcrData *r = [self paddleOcrRec:crop_img];
//        OcrData *r = [OcrData new];
//        r.label = [NSString stringWithFormat:@"%d",i];
//        r.accuracy = 0.0f;
        NSMutableArray *points = [NSMutableArray new];
        for (int jj = 0; jj < 4; ++jj) {
            NSValue *v = [NSValue valueWithCGPoint:CGPointMake(
                    rook_points[i][jj].x / CGFloat(originImage.cols),
                    rook_points[i][jj].y / CGFloat(originImage.rows))];
            [points addObject:v];
        }
        r.polygonPoints = points;
        [result addObject:r];
    }
    NSArray* rec_out =[[result reverseObjectEnumerator] allObjects];
    tic.end();
    std::cout<<"infer time: "<<tic.get_sum_ms()<<"ms"<<std::endl;
    return rec_out;
}

- (OcrData *)paddleOcrRec:(cv::Mat)image {

    OcrData *result = [OcrData new];

    std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

    cv::Mat crop_img;
    image.copyTo(crop_img);
    cv::Mat resize_img;

    float wh_ratio = float(crop_img.cols) / float(crop_img.rows);

    resize_img = crnn_resize_img(crop_img, wh_ratio);
    resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);

    const float *dimg = reinterpret_cast<const float *>(resize_img.data);

    std::unique_ptr<Tensor> input_tensor0(std::move(net_rec->GetInput(0)));
    input_tensor0->Resize({1, 3, resize_img.rows, resize_img.cols});
    auto *data0 = input_tensor0->mutable_data<float>();

    neon_mean_scale(dimg, data0, resize_img.rows * resize_img.cols, mean, scale);

    /// Run CRNN predictor
    net_rec->Run();

    // Get output and run postprocess
    std::unique_ptr<const Tensor> output_tensor0(std::move(net_rec->GetOutput(0)));
    auto *rec_idx = output_tensor0->data<int>();

    auto rec_idx_lod = output_tensor0->lod();
    auto shape_out = output_tensor0->shape();
    NSMutableString *text = [[NSMutableString alloc] init];
    for (int n = int(rec_idx_lod[0][0]); n < int(rec_idx_lod[0][1] * 2); n += 2) {
        if (rec_idx[n] >= self.labels.count) {
            std::cout << "Index " << rec_idx[n] << " out of text dict range!" << std::endl;
            continue;
        }
        [text appendString:self.labels[rec_idx[n]]];
    }
    std::cout << "Label: " << text << std::endl;
    result.label = text;
    // get score
    std::unique_ptr<const Tensor> output_tensor1(net_rec->GetOutput(1));
    auto *predict_batch = output_tensor1->data<float>();
    auto predict_shape = output_tensor1->shape();

    auto predict_lod = output_tensor1->lod();

    int argmax_idx;
    int blank = predict_shape[1];
    float score = 0.f;
    int count = 0;
    float max_value = 0.0f;

    for (int n = predict_lod[0][0]; n < predict_lod[0][1] - 1; n++) {
        argmax_idx = int(argmax(&predict_batch[n * predict_shape[1]], &predict_batch[(n + 1) * predict_shape[1]]));
        max_value = float(*std::max_element(&predict_batch[n * predict_shape[1]], &predict_batch[(n + 1) * predict_shape[1]]));

        if (blank - 1 - argmax_idx > 1e-5) {
            score += max_value;
            count += 1;
        }

    }
    score /= count;
    result.accuracy = score;
    return result;
}

#pragma mark - getter -
- (UIImageView *)imageView
{
    if (!_imageView) {
        _imageView = [[UIImageView alloc]initWithFrame:CGRectMake(0, 0, 300, 300)];
        _imageView.center = self.view.center;
        _imageView.contentMode = UIViewContentModeScaleAspectFit;
        [self.view addSubview:_imageView];
    }
    return _imageView;
}

- (CALayer *)boxLayer
{
    if(!_boxLayer)
    {
        _boxLayer = [[CALayer alloc] init];
        _boxLayer.frame = CGRectMake(0, 0, self.imageView.frame.size.width, self.imageView.frame.size.height);
        [self.imageView.layer addSublayer:self.boxLayer];
    }
    return _boxLayer;
}

@end
