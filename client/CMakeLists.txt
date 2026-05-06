cmake_minimum_required(VERSION 3.16)
project(remote_vlm_client LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(X11 REQUIRED)

# gRPC + Protobuf
find_package(Protobuf CONFIG)
if(NOT Protobuf_FOUND)
  find_package(Protobuf REQUIRED)
endif()
find_package(gRPC CONFIG REQUIRED)

# libdatachannel (WebRTC). Prefer config package, fall back to pkg-config.
find_package(LibDataChannel CONFIG QUIET)
if(NOT LibDataChannel_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(DATACHANNEL REQUIRED libdatachannel)
endif()

find_package(nlohmann_json 3.7.0 QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(json)
endif()

# ---------------------------------------------------------------------------
# Generated proto sources
# ---------------------------------------------------------------------------
set(PROTO_DIR ${CMAKE_SOURCE_DIR}/../proto)
set(GEN_DIR   ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GEN_DIR})

set(PROTO_FILES ${PROTO_DIR}/prompt_app.proto)

set(PROTO_SRCS ${GEN_DIR}/prompt_app.pb.cc ${GEN_DIR}/prompt_app.grpc.pb.cc)
set(PROTO_HDRS ${GEN_DIR}/prompt_app.pb.h  ${GEN_DIR}/prompt_app.grpc.pb.h)

get_target_property(GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin LOCATION)

add_custom_command(
  OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
  COMMAND $<TARGET_FILE:protobuf::protoc>
          --proto_path=${PROTO_DIR}
          --cpp_out=${GEN_DIR}
          --grpc_out=${GEN_DIR}
          --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
          ${PROTO_FILES}
  DEPENDS ${PROTO_FILES}
  COMMENT "Generating gRPC sources from prompt_app.proto"
  VERBATIM)

add_library(vlm_proto STATIC ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(vlm_proto PUBLIC ${GEN_DIR})
target_link_libraries(vlm_proto PUBLIC gRPC::grpc++ protobuf::libprotobuf)

# ---------------------------------------------------------------------------
# Main client binary
# ---------------------------------------------------------------------------
add_executable(remote_vlm_client
  src/main.cpp
  src/capture/screen_capture.cpp
  src/scd/scene_change_detector.cpp
  src/aff/adaptive_frame_filter.cpp
  src/webrtc/webrtc_client.cpp
  src/grpc_service/prompt_app_service.cpp
  src/pipeline/capture_pipeline.cpp)

target_include_directories(remote_vlm_client PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
  ${X11_INCLUDE_DIR})

target_link_libraries(remote_vlm_client PRIVATE
  vlm_proto
  ${OpenCV_LIBS}
  ${X11_LIBRARIES}
  Xext
  Threads::Threads
  nlohmann_json::nlohmann_json)

if(LibDataChannel_FOUND)
  target_link_libraries(remote_vlm_client PRIVATE LibDataChannel::LibDataChannel)
else()
  target_include_directories(remote_vlm_client PRIVATE ${DATACHANNEL_INCLUDE_DIRS})
  target_link_libraries(remote_vlm_client PRIVATE ${DATACHANNEL_LIBRARIES})
endif()
