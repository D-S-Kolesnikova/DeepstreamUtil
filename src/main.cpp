#include "DeepStreamWrapper.h"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <string>
#include <functional>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <array>
#include <regex>

#include <gst/gst.h>
#include <glib.h>

#include "gst-nvmessage.h"
#include "nvds_analytics_meta.h"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"

#include <NetworkInformationLib.h>
#include <ItvCvUtils/ItvCvDefs.h>

#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

std::istream& operator>>(std::istream& in, networkType& type)
{
    std::string token;
    in >> token;
    if (token == "Classification")
    {
        type = networkType::classification;
    } 
    else if (token == "Detector")
    {
        type = networkType::SSD;
    }
    else if (token == "Siamese")
    {
        type = networkType::siamese;
    }
    return in;
}

std::istream& operator>>(std::istream& in, architectureType& type)
{
    std::string token;
    in >> token;
    if (token == "ResNet")
    {
        type = architectureType::resNet;
    } 
    else if (token == "efficientNet")
    {
        type = architectureType::efficientNet;
    }
    else if (token == "Yolo")
    {
        type = architectureType::Yolo;
    }
    return in;
}

std::istream& operator>>(std::istream& in, modeType& type)
{
    std::string token;
    in >> token;

    if (token == "GPU")
    {
        type = modeType::modeGPU;
    }
    else if (token == "CPU")
    {
        type = modeType::modeCPU;
    }
    return in;
}

std::istream& operator>>(std::istream& in, boost::array<int, 3>& arr)
{
    for (size_t i = 0; i < arr.size(); i++)
    {
        in >> arr[i];
    }
    return in;
}

std::istream& operator>>(std::istream& in, boost::array<double, 3>& arr)
{
    for (size_t i = 0; i < arr.size(); i++)
    {
        in >> arr[i];
    }
    return in;
}

void ParseArgsToConfig(DeepStreamWrapper::InferPluginParams& inferParams, std::string& configFileName, const boost::program_options::variables_map& vm)
{
    inferParams.networkType = vm["net-type"].as<networkType>();
    auto currentDir = boost::filesystem::current_path();
    switch (inferParams.networkType)
    {
        case 0:
        {
            configFileName = currentDir.string() + "/DetectionFileconfig.txt";
            break;
        }
        case 1:
        {
            configFileName = currentDir.string() + "/ClassifierFileconfig.txt";
            break;
        }
        case 2:
        {
            configFileName = currentDir.string() + "/SegmentationFileconfig.txt";
            break;
        }
        case 3:
        {
            configFileName = currentDir.string() + "/InstanceSegmentationFileconfig.txt";
            break;
        }
    }
    inferParams.gpuId =  vm["gpu-id"].as<modeType>();
    inferParams.inferDims = vm["infer-dims"].as<std::vector<int>>();
    inferParams.gieUniqueId = vm["gie-unique-id"].as<int>();
    inferParams.outputBlobNames = vm["output-blob-names"].as<std::vector<std::string>>();
    inferParams.networkArchitecture = vm["architecture-type"].as<architectureType>();
    if(vm.count("ann-file"))
    {
        inferParams.annFile = vm["ann-file"].as<std::string>();
    }
    if(vm.count("onnx-file"))
    {
        inferParams.onnxFile = vm["onnx-file"].as<std::string>();
    }
    if(vm.count("model-caffe-file"))
    {
        inferParams.modelFile = vm["model-caffe-file"].as<std::string>();
    }
    if(vm.count("proto-file"))
    {
        inferParams.prototxtFile = vm["proto-file"].as<std::string>();
    }
    if(vm.count("engine-file"))
    {
        inferParams.modelEnginePath = vm["engine-file"].as<std::string>();
        const std::regex r(R"(_b\d+_)");
        std::cmatch m;
        if (std::regex_search(inferParams.modelEnginePath.c_str(), m,r))
        {
            auto result =  m.str(0);
            char eraseSymbols[2] = {'b', '_'};
            for (char&c : eraseSymbols)
            {
                result.erase(std::remove(result.begin(), result.end(), c), result.end());
            }
            if (std::stoi(result) != inferParams.batchSize)
            {
                std::cout << "WARNING: Enginefile does not suitable for model description! Check batch-size.\n";
            }
        }
    }
    if(vm.count("label-path"))
    {
        inferParams.labelfilePath = vm["label-path"].as<std::string>();
    }
    if(vm.count("batch-size"))
    {
        inferParams.batchSize = vm["batch-size"].as<size_t>();
    }
    if(vm.count("model-color-format"))
    {
        inferParams.modelColorFormat = vm["model-color-format"].as<size_t>();
    }
    if(vm.count("network-mode"))
    {
        inferParams.networkMode = vm["network-mode"].as<size_t>();
    }
    if(vm.count("scale-factor"))
    {
        inferParams.netScaleFactor = vm["scale-factor"].as<std::vector<double>>();
    }
    if(vm.count("offsets"))
    {
        inferParams.offsets = vm["offsets"].as<std::vector<double>>();
    }
    if(vm.count("print-output-tensor"))
    {
        inferParams.outputTensorMeta = vm["print-output-tensor"].as<bool>();
    }
    if(vm.count("num-detected-classes"))
    {
        inferParams.numDetectedClasses = vm["num-detected-classes"].as<size_t>();
    }
    if(vm.count("classifier-threshold"))
    {
        inferParams.classifierThreshold = vm["classifier-threshold"].as<double>();
    }
    if(vm.count("parse-bbox-func"))
    {
        inferParams.bboxFuncName = vm["parse-bbox-func"].as<std::string>();
    }
    if(vm.count("custom-lib-path"))
    {
        inferParams.customLibPath = vm["custom-lib-path"].as<std::string>();
    }
    
    DeepStreamWrapper::NvInferPlugin inferPlugin(inferParams, configFileName);
    inferPlugin.GenerateConfig(configFileName);
}

static void NewPad (GstElement* decodebin, GstPad* decoder_src_pad, gpointer data)
{
    std::cout << "In cb_newpad\n";
    GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
    const GstStructure *str = gst_caps_get_structure (caps, 0);
    const gchar *name = gst_structure_get_name (str);
    GstElement *source_bin = (GstElement *) data;
    GstCapsFeatures *features = gst_caps_get_features (caps, 0);

    if (!strncmp (name, "video", 5)) 
    {
        if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) 
        {
            /* Get the source bin ghost pad */
            GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
            if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad), decoder_src_pad)) 
            {
                std::cout <<"Failed to link decoder src pad to source bin ghost pad\n";
            }
            gst_object_unref (bin_ghost_pad);
        } 
        else 
        {
            std::cout <<"Error: Decodebin did not pick nvidia decoder plugin.\n";
        }
    }
}

static void DecodeChildAdded (GstChildProxy * child_proxy, GObject * object, gchar * name, gpointer user_data)
{
    std::cout <<"Decodebin child added: " << name << "\n";
    if (g_strrstr (name, "decodebin") == name) 
    {
        g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (DecodeChildAdded), user_data);
    }
}

static GstElement* CreateSourceBin (guint index, gchar * uri)
{
    GstElement *bin = NULL, *uri_decode_bin = NULL;
    gchar bin_name[16] = { };

    g_snprintf (bin_name, 15, "source-bin-%02d", index);
    /* Create a source GstBin to abstract this bin's content from the rest of the pipeline */
    bin = gst_bin_new (bin_name);
    uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");
    if (!bin || !uri_decode_bin) 
    {
        std::cout <<"One element in source bin could not be created.\n";
        return NULL;
    }
    /* We set the input uri to the source element */
    g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

    /* Connect to the "pad-added" signal of the decodebin which generates a
    * callback once a new pad for raw data has beed created by the decodebin */
    g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added", G_CALLBACK (NewPad), bin);
    g_signal_connect (G_OBJECT (uri_decode_bin), "child-added", G_CALLBACK (DecodeChildAdded), bin);
    gst_bin_add (GST_BIN (bin), uri_decode_bin);
    if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src", GST_PAD_SRC))) 
    {
        std::cout <<"Failed to add ghost pad in source bin\n";
        return NULL;
    }
    return bin;
}

static int count = 0;
static GstPadProbeReturn SrcPadBufferProbe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data)
{
    NvDsObjectMeta *obj_meta = NULL;
    guint smokeCount = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));
    DeepStreamWrapper::transferData *userData = static_cast<DeepStreamWrapper::transferData*>(u_data);
    int netType = userData->netType;
    int netArch = userData->netArch;

    std::cout <<"ResultNumber = " << count << "\n";
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL;l_frame = l_frame->next)
    { 
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
        /* Iterate user metadata in frames to search PGIE's tensor metadata */
        for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next)
        {
            NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
            if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
            {
                std::cout <<"meta_type = " << user_meta->base_meta.meta_type << "\n";
            }
            NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
            for (unsigned int i = 0; i < meta->num_output_layers; i++) 
            {
                NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                info->buffer = meta->out_buf_ptrs_host[i];
                NvDsInferDimsCHW dims;
                getDimsCHWFromDims (dims, meta->output_layers_info[i].inferDims);
                if(netType == 0)
                {
                    auto detectionCount = 0;
                    while(detectionCount < dims.c)
                    {
                        auto result = ((float*)info->buffer);
                        DeepStreamWrapper::Detection detection(result + detectionCount * dims.h, dims.h, architectureType(netArch));
                        if (detection.m_score > 0.3)
                        {
                            std::cout << "NumOfSuccessDetection: " << detectionCount << "\t"
                            << detection.m_xMin << "\t"
                            << detection.m_yMin << "\t"
                            << detection.m_xMax << "\t"
                            << detection.m_yMax << "\n";
                        }
                        detectionCount++;
                    }
                }
                else if(netType == 1)
                {
                    unsigned int numClasses = dims.c;
                    float *outputCoverageBuffer = (float *) meta->output_layers_info[0].buffer;
                    for (unsigned int c = 0; c < numClasses; c++)
                    {
                        float probability = outputCoverageBuffer[c];
                        std::cout <<"Output = " << c << " probability = " << probability;
                    }
                    std::cout << "\n";
                }
            }
        }
    } 
    count ++;
    return GST_PAD_PROBE_OK;
}

static gboolean BusCall(GstBus* bus, GstMessage* msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE (msg)) 
    {
    case GST_MESSAGE_EOS: 
        std::cout <<"End of stream\n";
        g_main_loop_quit (loop);
        break;
    case GST_MESSAGE_WARNING:
    {
        gchar *debug;
        GError *error;
        gst_message_parse_warning (msg, &error, &debug);
        std::cout << "WARNING from element: " << GST_OBJECT_NAME (msg->src) << " " << error->message << "\n";
        g_free (debug);
        std::cout <<"Warning: " << error->message<< "\n";
        g_error_free (error);
        break;
    }
    case GST_MESSAGE_ERROR:
    {
        gchar *debug;
        GError *error;
        gst_message_parse_error (msg, &error, &debug);
        std::cout <<"ERROR from element : "<< GST_OBJECT_NAME (msg->src) << " " << error->message << "\n";
        if (debug)
          std::cout <<"Error details: " << debug << "\n";
        g_free (debug);
        g_error_free (error);
        g_main_loop_quit (loop);
        break;
    }
#ifndef PLATFORM_TEGRA
    case GST_MESSAGE_ELEMENT:
    {
      if (gst_nvmessage_is_stream_eos (msg)) 
      {
        guint stream_id;
        if (gst_nvmessage_parse_stream_eos (msg, &stream_id)) 
        {
          std::cout <<"Got EOS from stream :" << stream_id << "\n";
        }
      }
      break;
    }
#endif
    default:
      break;
    }
    return TRUE;
}

int main (int argc, char **argv)
{
    namespace po = boost::program_options;

    po::options_description modelDesc("Model options");
    modelDesc.add_options()
        ("net-type", po::value<networkType>()->required(), "network type {Detector, Classification, Segmentation, Instance Segmentation}")
        ("architecture-type", po::value<architectureType>()->required(), "architecture type {efficientNet, resNet, Yolo}")
        ("onnx-file", po::value<std::string>(), "path to the ONNX model file")
        ("model-caffe-file", po::value<std::string>(), "path to the caffemodel file")
        ("proto-file", po::value<std::string>(), "path to the prototxt file")
        ("ann-file", po::value<std::string>(), "ann-file added");

    po::options_description sysDesc("System options");
    sysDesc.add_options()
        ("gpu-id", po::value<modeType>()->required(), "device ID of GPU to use for pre-processing/inference")
        ("gie-unique-id", po::value<int>()->required(), "unique ID to be assigned to the GIE to enable the application and other elements to identify detected bounding boxes and labels")
        ("infer-dims", po::value<std::vector<int>>()->multitoken()->required(), "blinding dimensions to set on the image input layer {number of chsnnel, width, height}")
        ("batch-size", po::value<size_t>()->default_value(1), "threads amount to be used for calling inference concurrently.")
        ("engine-file", po::value<std::string>(), "path to the pre-generated serialized engine file for the mode")
        ("label-path",  po::value<std::string>(), "path to the text file containing the labels for the model")
        ("video-files", po::value<std::vector<std::string>>()->multitoken(), "path to video files as an input")
        ("network-mode", po::value<size_t>()->default_value(0), "0: FP32 1: INT8 2: FP16, default FP32")
        ("scale-factor", po::value<std::vector<double>>()->multitoken(), "pixel normalization factor, default 1.0")
        ("offsets", po::value<std::vector<double>>()->multitoken(), "array of mean values of color components to be subtracted from each pixel. Array length must equal the number of color components in the frame: {255, 255, 255}")
        ("model-color-format", po::value<size_t>()->default_value(0), "0: RGB 1: BGR 2: GRAY, default RGB")
        ("print-output-tensor", po::value<bool>()->default_value(false), "get raw tensor output, default false")
        ("num-detected-classes", po::value<size_t>(), "number of classes detected by the network")
        ("classifier-threshold", po::value<double>(), "minimum threshold label probability")
        ("parse-bbox-func", po::value<std::string>(), "name of the custom bounding box parsing function")
        ("custom-lib-path",po::value<std::string>(), "absolute pathname of a library containing custom method implementations for custom models")
        ("output-blob-names", po::value<std::vector<std::string>>()->multitoken()->required(), "array of output layer names");

    po::options_description desc("DeepStream");
    desc.add_options()("help", "print help");
    desc.add(sysDesc).add(modelDesc);

    po::variables_map vm;
    bool printHelpAndExit{ false };
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
        if (vm.count("help") || argc == 1)
        {
            printHelpAndExit = true;
        }
    }
    catch (const std::exception& e)
    {
        printHelpAndExit = true;
        std::cout << "exception: " << e.what() << std::endl;
    }
    if (printHelpAndExit)
    {
        std::cout
            << desc
            << "\n"
            << "Usage templates:\n"
            << "./DeepstreamBin --video-files file:///media/daria/HDD2/devel/videos/deepstream/CLASS1.avi --net-type Classification --architecture-type ResNet --gpu-id 0 --gie-unique-id 1 --infer-dims 3 112 112 --onnx-file /media/daria/HDD2/devel/helper/deepstream/test_batch/test_bs1_cls3_dynamic.onnx  --engine-file /media/daria/HDD2/devel/helper/deepstream/test_batch/test_bs1_cls3_dynamic.onnx_b1_gpu0_fp32.engine  --network-mode 0 --offsets 0 0 0 --model-color-format 0 --batch-size 1 --scale-factor 0.003921 --num-detected-classes 3 --classifier-threshold 0.5 --print-output-tensor 1 \n"
            << std::endl;
        return 0;
    }
    std::vector<std::string> inputThreads = vm["video-files"].as< std::vector<std::string>>();

    std::string configFileName;
    DeepStreamWrapper::InferPluginParams inferParams;
    ParseArgsToConfig(inferParams, configFileName, vm);


    GMainLoop *loop = NULL;
    GstElement  *pipeline = NULL, 
                *streammux = NULL, 
                *sink = NULL, 
                *pgie = NULL, 
                *nvdslogger = NULL,
                *queue1 = NULL,
                *queue2 = NULL, 
                *queue3 = NULL;
    GstBus *bus = NULL;

    guint numSources = inputThreads.size();
    
    /* Standard GStreamer initialization */
    gst_init (&argc, &argv);
    loop = g_main_loop_new (NULL, FALSE);
    
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new ("nvdsanalytics-test-pipeline");

    /* Create nvstreammux instance to form batches from one or more sources. */
    streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

    if (!pipeline || !streammux) 
    {
        std::cout <<"One element could not be created. Exiting.\n";
        return -1;
    }
    gst_bin_add (GST_BIN (pipeline), streammux);

    for (guint i = 0; i < numSources; i++) 
    {
        GstPad *sinkpad, *srcpad;
        gchar pad_name[16] = { };
        GstElement *source_bin = CreateSourceBin(i, (gchar *)(inputThreads[i].c_str()));

        if (!source_bin) 
        {
            std::cout <<"Failed to create source bin. Exiting.\n";
            return -1;
        }

        gst_bin_add (GST_BIN (pipeline), source_bin);
        g_snprintf (pad_name, 15, "sink_%u", i);
        sinkpad = gst_element_get_request_pad (streammux, pad_name);

        if (!sinkpad)
        {
            std::cout <<"Streammux request sink pad failed. Exiting.\n";
            return -1;
        }

        srcpad = gst_element_get_static_pad (source_bin, "src");
        if (!srcpad)
        {
          std::cout <<"Failed to get src pad of source bin. Exiting.\n";
          return -1;
        }

        if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) 
        {
            std::cout <<"Failed to link source bin to stream muxer. Exiting.\n";
            return -1;
        }

        gst_object_unref (srcpad);
        gst_object_unref (sinkpad);
    }


    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
    queue1 = gst_element_factory_make ("queue", "queue1");
    queue2 = gst_element_factory_make ("queue", "queue2");
    queue3 = gst_element_factory_make ("queue", "queue3");
    sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
    nvdslogger = gst_element_factory_make ("nvdslogger", "nvdslogger");

    if (!pgie || !sink || !queue1 || !queue2 || !queue3) 
    {
        std::cout << "One plugin element in DeepStream could not be created. Exiting.\n";
        return -1;
    }

    g_object_set (G_OBJECT (streammux),  "width", 1920, "height", 1080, "batch-size", numSources, "batched-push-timeout", 4000000, NULL);
    if ( !boost::filesystem::exists( configFileName ) )
    {
        std::cout << "Can't find config-file-path! Exiting.\n" << "\n";
        return -1;
    }
    g_object_set (G_OBJECT (pgie), "config-file-path", configFileName.c_str(), NULL);
    guint pgieBatchSize;
    g_object_get (G_OBJECT (pgie), "batch-size", &pgieBatchSize, NULL);
    if (pgieBatchSize != numSources)
    {
        std::cout << "WARNING: Dynamic batching is OFF. Batch-size is " << pgieBatchSize << " with number of sources " << numSources << " \n";
    }

    /* add a message handler */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    guint busWatchId = gst_bus_add_watch (bus, BusCall, loop);
    gst_object_unref (bus);

    gst_bin_add_many (GST_BIN (pipeline), queue1, pgie, queue2, nvdslogger, queue3, sink, NULL);
    if (!gst_element_link_many (streammux, queue1, pgie, queue2, nvdslogger, queue3, sink, NULL))
    {
      std::cout << "Deepstream elements could not be linked. Exiting.\n";
      return -1;
    }
    if (inferParams.outputTensorMeta)
    {
        GstPad* srcPad = gst_element_get_static_pad (pgie, "src");
        DeepStreamWrapper::transferData userData = {inferParams.networkType, inferParams.networkArchitecture};

        if (!srcPad)
        {
            std::cout << "Unable to get src pad\n";
        }
        else
        {
            gst_pad_add_probe (srcPad, GST_PAD_PROBE_TYPE_BUFFER, SrcPadBufferProbe, &userData, NULL);
        }  
        gst_object_unref (srcPad);
    }

    /* Set the pipeline to "playing" state */
    std::cout <<"Now playing:";
    for (gint i = 0; i < numSources; i++) 
    {
        std::cout << inputThreads[i];
    }
    std::cout << "\n";
    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    std::cout << "Running...\n";
    g_main_loop_run (loop);

    /* Out of the main loop, clean up nicely */
    std::cout << "Returned, stopping playback\n";
    gst_element_set_state (pipeline, GST_STATE_NULL);
    std::cout << "Deleting pipeline\n";
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (busWatchId);
    g_main_loop_unref (loop);

    return 0;
}
