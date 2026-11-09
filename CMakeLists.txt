cmake_minimum_required(VERSION 3.20)
project(dragonedge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(DRAGON_WITH_QNN "Build the QNN HTP backend and the on-device tools (needs the QAIRT/QNN SDK headers)" OFF)
set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}" CACHE PATH "QAIRT / QNN SDK root (contains include/QNN)")
set(DRAGON_TEST_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/../build/manifest.json" CACHE FILEPATH
    "manifest.json used by the unit tests")

find_package(Threads REQUIRED)

add_library(dragonedge STATIC
  src/arena.cpp
  src/file_io.cpp
  src/json.cpp
  src/log.cpp
  src/manifest.cpp
  src/mem_accounting.cpp
  src/memory_plan.cpp
  src/png.cpp
  src/sampler.cpp
  src/streamer.cpp
  src/unet_runner.cpp
)
target_include_directories(dragonedge PUBLIC include)
target_link_libraries(dragonedge PUBLIC Threads::Threads)

if(MSVC)
  target_compile_options(dragonedge PUBLIC /W4 /permissive- /utf-8)
  target_compile_definitions(dragonedge PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX)
  target_link_libraries(dragonedge PUBLIC psapi)
else()
  target_compile_options(dragonedge PUBLIC -Wall -Wextra)
endif()

if(DRAGON_WITH_QNN)
  if(NOT EXISTS "${QNN_SDK_ROOT}/include/QNN/QnnInterface.h")
    message(FATAL_ERROR "DRAGON_WITH_QNN=ON but ${QNN_SDK_ROOT}/include/QNN/QnnInterface.h was not found; set QNN_SDK_ROOT")
  endif()
  target_sources(dragonedge PRIVATE src/qnn_backend.cpp)
  target_include_directories(dragonedge PRIVATE "${QNN_SDK_ROOT}/include/QNN")
  target_compile_definitions(dragonedge PUBLIC DRAGON_WITH_QNN=1)
  target_link_libraries(dragonedge PUBLIC ${CMAKE_DL_LIBS})

  foreach(tool npu_selftest shard_bench run_unet)
    add_executable(${tool} tools/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE dragonedge)
  endforeach()
endif()

if(NOT ANDROID)
  enable_testing()
  add_executable(engine_tests tests/engine_tests.cpp tests/mock_backend.cpp)
  target_include_directories(engine_tests PRIVATE tests)
  target_link_libraries(engine_tests PRIVATE dragonedge)
  target_compile_definitions(engine_tests PRIVATE DRAGON_TEST_MANIFEST="${DRAGON_TEST_MANIFEST}")
  add_test(NAME engine_tests COMMAND engine_tests)
endif()
