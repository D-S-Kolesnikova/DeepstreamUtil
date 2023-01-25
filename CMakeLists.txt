project(DeepstreamBin)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CUDA_VER 11.8)
set(NVDS_VERSION 6.1)

set(LIB_INSTALL_DIR /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/)
set(APP_INSTALL_DIR /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/)


find_package(PkgConfig REQUIRED)
set(CMAKE_FIND_DEBUG_MODE TRUE)
find_package(GStreamer REQUIRED)
set(CMAKE_FIND_DEBUG_MODE FALSE)
find_package(DeepstreamAppCommon REQUIRED)
set(CMAKE_FIND_DEBUG_MODE TRUE)
message("VVVVVVVV")
find_package(NVDS REQUIRED)
message("^^^^^^^^")
set(CMAKE_FIND_DEBUG_MODE FALSE)
find_package(Boost REQUIRED COMPONENTS program_options filesystem)
find_package(CUDA REQUIRED)

pkg_search_module(GLIB REQUIRED glib-2.0)
pkg_check_modules(JSON-GLIB REQUIRED json-glib-1.0)

set(NetworkInformation_DIR /media/daria/HDD2/devel/axxon/repos/computervision/!build_computervision/NetworkInformation)
set(ItvCvUtils_DIR /media/daria/HDD2/devel/axxon/repos/computervision/!build_computervision/ItvCvUtils)
set(cryptoWrapper_DIR /media/daria/HDD2/devel/axxon/repos/computervision/!build_computervision/cryptoWrapper)
set(HpePostProcessing_DIR /media/daria/HDD2/devel/axxon/repos/computervision/!build_computervision/HpePostProcessing)
set(InferenceWrapper_DIR /media/daria/HDD2/devel/axxon/repos/computervision/!build_computervision/InferenceWrapper)

find_package(NetworkInformation REQUIRED)

include_directories(
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_VIDEO_INCLUDE_DIRS}
    ${GSTREAMER_RTSPSERVER_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${JSON-GLIB_INCLUDE_DIRS}
    ${NVDS_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    src
    ${LIB_INSTALL_DIR}
    /opt/nvidia/deepstream/deepstream-6.1/sources/includes
    /usr/local/cuda-$(CUDA_VER)/include
)

add_executable(${PROJECT_NAME}
    src/DeepStreamWrapper.cpp
    src/DeepStreamWrapper.h
    src/DeepStreamWrapperParams.h
    src/main.cpp
    ${APP_INSTALL_DIR}
)

target_compile_options(${PROJECT_NAME} PRIVATE -Wno-parentheses -Wno-sign-compare -O0 -g -fPIC -lboost_program_options -lboost_filesystem)

target_link_libraries(${PROJECT_NAME}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_VIDEO_LIBRARIES}
    ${GSTREAMER_RTSPSERVER_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${JSON-GLIB_LIBRARIES}
    ${NVDS_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CMAKE_DL_LIBS}
    Boost::filesystem
    Boost::program_options
    NetworkInformation
)
