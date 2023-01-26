DeepStreamWrapperBin позволяет составлять пайплайн с использованием функционала deepstream SDK.

Operating System
    NVIDIA Jetson™: Ubuntu 20.04
    NVIDIA Tesla® GPUs (x86): Ubuntu 20.04
    NVIDIA dGPU:Ubuntu 20.04

Hardware:
    Jetson platforms: AGX Xavier, Jetson NX, Jetson Orin
    dGPU (“discrete GPU”) to refer to NVIDIA GPU expansion card products such as NVIDIA Tesla® T4,
    NVIDIA GeForce® GTX 1080, NVIDIA GeForce® RTX 2080 and NVIDIA GeForce® RTX 3080

Dependencies
    DS 6.1
    GCC 8.4.0
    NVIDIA CUDA®: 11.6.1, 
    NVIDIA cuDNN: 8.4.0.27, 
    NVIDIA TensorRT™: 8.2.5.1, 
    OpenCV: 4.2.0, 
    GStreamer 1.16.2

Tesla GPUs (x86): Driver: R515+, CUDA: 11.7 Update 1, cuDNN: 8+, TensorRT: 8.4.1.5, Triton 22.07, GStreamer 1.16.3

Обязательные параметры:
  --video-files : sources (видеофайлы, изображения)
  --net-type : тип cети (Detector, Classification, Segmentation, Instance Segmentation)
  --architecture-type : архитектура (efficientNet, resNet34, Yolo)
  --gpu-id : номер девайса (0, 1, 2..)
  --infer-dims : требуемый размер сети ({3 300 300})
  --output-blob-names : названия выходных слоев сети (For detector: output-blob-names=coverage;bbox)
  --gie-unique-id : GPU inference engine, номер движка необходим в случае использования последовательности моделей (0)
  
Для классификационных сетей дополнительные обязательные параметры:
  --num-detected-classes : количество классов
  --classifier-threshold : порог отнесения к классу

Для ssd сетей дополнительные обязательные параметры:
  --num-detected-classes : количество классов
  --parse-bbox-func : название функции для кастомного определения рамок

Присутствует возможность поддержки:
- Caffe Model support (обязательные аргументы --model-caffe-file, --proto-file)
- UFF Model support (обязательные аргументы --uff-file)
- ONNX Model support (обязательные аргументы onnx-file)

Присутствует возможность кэширования (при первом использовании модели строится desirialized engine, далее рекомендуется указывать --engine-path)

  Пример: 
   ./DeepstreamBin --video-files file:///media/daria/HDD2/devel/videos/deepstream/CLASS1.avi --net-type Classification --architecture-type efficientNet --gpu-id 0 --gie-unique-id 1 --infer-dims 3 112 112 --onnx-file /media/daria/HDD2/devel/helper/deepstream/test_batch/test_bs1_cls3_dynamic.onnx --engine-file /media/daria/HDD2/devel/helper/deepstream/test_batch/test_bs1_cls3_dynamic.onnx_b1_gpu0_fp32.engine --network-mode 0 --offsets 0 0 0 --model-color-format 0 --batch-size 1 --scale-factor 0.003921 --num-detected-classes 3 --classifier-threshold 0.5 --print-output-tensor 1 --output-blob-names output
   
   Классификационная сетка: 
    Batch = 1  
    ./DeepstreamBin --video-files file:///media/daria/HDD2/devel/videos/deepstream/CLASS1.avi
    --net-type Classification --architecture-type efficientNet --gpu-id 0 --gie-unique-id 1 --infer-dims 3 112 112
    --onnx-file /media/daria/HDD2/devel/helper/deepstream/test_batch/test_bs1_cls3_dynamic.onnx
    --engine-file /media/daria/HDD2/devel/helper/deepstream/test_batch/test_bs1_cls3_dynamic.onnx_b1_gpu0_fp32.engine
    --network-mode 0 --offsets 0 0 0 --model-color-format 0 --batch-size 1 --scale-factor 0.003921 --num-detected-classes 3
    --classifier-threshold 0.5 --print-output-tensor 1 --output-blob-names output
    
    Batch = 2 
    ./DeepstreamBin --video-files file:///media/daria/HDD2/devel/videos/deepstream/CLASS1.avi
    file:///media/daria/HDD2/devel/videos/deepstream/CLASS1.avi --net-type Classification --architecture-type efficientNet
    --gpu-id 0 --gie-unique-id 1 --infer-dims 3 112 112 --onnx-file /media/daria/HDD2/devel/helper/deepstream/test_batch/test_bs1_cls3_dynamic.onnx
    --engine-file /media/daria/HDD2/devel/helper/deepstream/test_batch/test_bs1_cls3_dynamic.onnx_b2_gpu0_fp32.engine --network-mode 0
    --offsets 0 0 0 --model-color-format 0 --batch-size 2 --scale-factor 0.003921 --num-detected-classes 3 --classifier-threshold 0.5 --print-output-tensor 1 --output-blob-names output
    
    Batch = 3 
    ./DeepstreamBin --video-files file:///media/daria/HDD2/devel/videos/deepstream/CLASS1.avi
    file:///media/daria/HDD2/devel/videos/deepstream/CLASS1.avi
    file:///media/daria/HDD2/devel/videos/deepstream/CLASS1.avi
    --net-type Classification --architecture-type efficientNet --gpu-id 0 --gie-unique-id 1 --infer-dims 3 112 112 
    --onnx-file /media/daria/HDD2/devel/helper/deepstream/test_batch/test_bs1_cls3_dynamic.onnx --engine-file 
    /media/daria/HDD2/devel/helper/deepstream/test_batch/test_bs1_cls3_dynamic.onnx_b3_gpu0_fp32.engine 
    --network-mode 0 --offsets 0 0 0 --model-color-format 0 --batch-size 3 --scale-factor 0.003921 
    --num-detected-classes 3 --classifier-threshold 0.5 --print-output-tensor 1 --output-blob-names output
    
    SSD:
    ./DeepstreamBin --video-files file:///media/daria/HDD2/devel/videos/720p.avi --net-type Detector --gpu-id 0 --gie-unique-id 1 --infer-dims 3 300 300 --onnx-file /media/daria/HDD2/devel/helper/deepstream/ssd/output_1.onnx --engine-file /media/daria/HDD2/devel/helper/deepstream/ssd/output_1.onnx_b1_gpu0_fp32.engine --network-mode 0 --offsets 123 117 104 --model-color-format 0 --batch-size 1 --scale-factor 1.0  --num-detected-classes 2 --print-output-tensor 1 --parse-bbox-func NvDsInferParseCustomTfSSD --custom-lib-path /opt/nvidia/deepstream/deepstream-6.1/lib/libnvds_infercustomparser.so --output-blob-names output --architecture-type ResNet
    
    YOLO:
    ./DeepstreamBin --video-files file:///media/daria/HDD2/devel/videos/720p.avi --net-type Detector --gpu-id 0 --gie-unique-id 1 --infer-dims 3 416 416 --onnx-file /media/daria/HDD2/devel/helper/deepstream/ssd/Yolo/dpe_1991_yolo_m_human_onnx_1 --engine-file /media/daria/HDD2/devel/helper/deepstream/ssd/Yolo/dpe_1991_yolo_m_human_onnx_1_b1_gpu0_fp32.engine --network-mode 0 --offsets 123 117 104 --model-color-format 0 --batch-size 1 --scale-factor 1.0  --num-detected-classes 2 --print-output-tensor 1 --parse-bbox-func NvDsInferParseCustomTfSSD --custom-lib-path /opt/nvidia/deepstream/deepstream-6.1/lib/libnvds_infercustomparser.so --output-blob-names detection_out keep_count  --architecture-type Yolo 
    ./DeepstreamBin --video-files file:///media/daria/HDD2/devel/videos/720p.avi --net-type Detector --gpu-id 0 --gie-unique-id 1 --infer-dims 3 640 1152 --onnx-file /media/daria/HDD2/devel/helper/deepstream/ssd/Yolo/weights.onnx --network-mode 0 --offsets 0 0 0 --model-color-format 0 --batch-size 1 --scale-factor 1.0  --num-detected-classes 7 --print-output-tensor 0 --parse-bbox-func NvDsInferParseCustomTfSSD --custom-lib-path /opt/nvidia/deepstream/deepstream-6.1/lib/libnvds_infercustomparser.so --output-blob-names detection_out keep_count  --architecture-type Yolo --engine-file /media/daria/HDD2/devel/helper/deepstream/ssd/Yolo/weights.onnx_b1_gpu0_fp32.engine

  вызвать справку:
  cryptoWrapperBin.exe -h или cryptoWrapperBin.exe --help
