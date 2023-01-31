#include "DeepStreamWrapperParams.h"

#include <string>
#include <vector>
#include <memory>
#include <fstream>

namespace DeepStreamWrapper
{

struct InferPluginParams
{
    std::string annFile;
    std::string onnxFile;
    std::string modelFile;
    std::string prototxtFile;
    int networkType = 0;/*0: Detector, 1: Classifier, 2: Segmentation,3: Instance Segmentation*/
    int networkArchitecture = 0;/*0: Detector, 1: Classifier, 2: Segmentation,3: Instance Segmentation*/
    int gpuId = 0; /*Device ID of GPU to use for pre-processing/inference (dGPU only)*/
    std::vector<int> inferDims;/*Binding dimensions to set on the image input layer {number of chsnnel, width, height}*/
    std::string modelEnginePath; /*0: efficientNet, 1: resNet, 2: Yolo*/
    std::string labelfilePath;
    int networkMode = 0;/*0: FP32 1: INT8 2: FP16*/
     std::vector<double> netScaleFactor;
    std::vector<double> offsets;
    int modelColorFormat = 0;/*0: RGB 1: BGR 2: GRAY*/
    int batchSize = 1;
    bool outputTensorMeta = false;
    int gieUniqueId = 0;/*Uniqie id of model*/
    /*for classifiers*/
    int numDetectedClasses = 1;
    double classifierThreshold = 0.5;

    std::string bboxFuncName;
    std::string customLibPath;
    std::vector<std::string> outputBlobNames;
};

class NvInferPlugin
{
public:
    NvInferPlugin (const InferPluginParams& inferParams, const std::string configFileName);
    void GenerateConfig(const std::string configFileName);

private:
    InferPluginParams m_inferParams;
};

class Detection
{
public:
    Detection (float* buffer, size_t sizeOfBuff, architectureType type);
public:
    int m_imageId = 0;
    int m_label = 0;
    double m_score = 0.0;
    double m_xMin = 0.0;
    double m_yMin = 0.0;
    double m_xMax = 0.0;
    double m_yMax = 0.0;
};

typedef struct 
{ 
    int netType;
    int netArch; 
} transferData;

}

