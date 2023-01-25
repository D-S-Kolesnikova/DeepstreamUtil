#include "DeepStreamWrapper.h"

#include <fstream>

namespace DeepStreamWrapper
{

NvInferPlugin::NvInferPlugin (const InferPluginParams& inferParams, const std::string configFileName)
{
    m_inferParams = inferParams;
}
void NvInferPlugin::GenerateConfig(const std::string configFileName)
{
    std::ofstream ofs(configFileName);
    ofs << "[property]\n";
    ofs << "gpu-id=" << m_inferParams.gpuId << "\n";
    ofs << "gie-unique-id=" << m_inferParams.gieUniqueId << "\n";
    ofs << "batch-size=" << m_inferParams.batchSize << "\n";
    ofs << "network-mode=" << m_inferParams.networkMode << "\n";
    ofs << "offsets=" << m_inferParams.offsets[0] << ";" << m_inferParams.offsets[1] << ";" <<  m_inferParams.offsets[2] << "\n";
    ofs << "model-color-format=" << m_inferParams.modelColorFormat << "\n";
    ofs << "network-type=" << m_inferParams.networkType << "\n";
    ofs << "net-scale-factor=" << m_inferParams.netScaleFactor << "\n";
    ofs << "infer-dims=" << m_inferParams.inferDims[0] << ";" << m_inferParams.inferDims[1] << ";" <<  m_inferParams.inferDims[2] << "\n";
    ofs << "output-tensor-meta=" << m_inferParams.outputTensorMeta << "\n";
    if (m_inferParams.networkType == networkType::classification)
    {
        ofs << "num-detected-classes=" << m_inferParams.numDetectedClasses << "\n";
        ofs << "classifier-threshold=" << m_inferParams.classifierThreshold << "\n";
    }
    else if (m_inferParams.networkType == networkType::SSD)
    {
        ofs << "num-detected-classes=" << m_inferParams.numDetectedClasses << "\n";
        if(!m_inferParams.bboxFuncName.empty())
        {
            ofs << "parse-bbox-func-name=" << m_inferParams.bboxFuncName << "\n";
        }
    }
    if(!m_inferParams.onnxFile.empty())
    {
        ofs << "onnx-file=" << m_inferParams.onnxFile << "\n";
    }
    if(!m_inferParams.modelFile.empty())
    {
        ofs << "model-file=" << m_inferParams.modelFile << "\n";
    }
    if(!m_inferParams.prototxtFile.empty())
    {
        ofs << "proto-file=" << m_inferParams.prototxtFile << "\n";
    }
    if(!m_inferParams.modelEnginePath.empty())
    {
        ofs << "model-engine-file=" << m_inferParams.modelEnginePath << "\n";
    }
    if(!m_inferParams.customLibPath.empty())
    {
        ofs << "custom-lib-path=" << m_inferParams.customLibPath << "\n";
    }
    if(!m_inferParams.outputBlobNames.empty())
    {
        ofs << "output-blob-names=";
        for(int i=0; i < m_inferParams.outputBlobNames.size(); i++)
        {
            if (i == m_inferParams.outputBlobNames.size()-1)
            {
                ofs << m_inferParams.outputBlobNames[i] << "\n";
                break;
            }
            ofs << m_inferParams.outputBlobNames[i] << ";";
        }
    }
    ofs.close();
}

Detection::Detection(float* buffer, size_t sizeOfBuff, architectureType type)
{
    switch (type)
    {
    case architectureType::resNet:
    {
        m_xMin = buffer[0];
        m_yMin = buffer[1];
        m_xMax = buffer[2];
        m_yMax = buffer[3];
        m_score = buffer[5];
        m_imageId = buffer[6];
        break;
    }
    case architectureType::Yolo:
    {
        m_imageId = buffer[0];
        m_label = buffer[1];
        m_score = buffer[2];
        m_xMin = buffer[3];
        m_yMin = buffer[4];
        m_xMax = buffer[5];
        m_yMax = buffer[6];
    }
    default:
        break;
    }
}

}
