cmake_minimum_required(VERSION 3.10)
project(MotionComp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# HDF5 (for reading M3ED .h5 event + RGB data)
set(HDF5_PREFER_PARALLEL FALSE)
find_package(HDF5 REQUIRED COMPONENTS C CXX)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
)

# -- MOSSE tracker with event-camera flow + re-ID --
add_executable(tracker
    src/tracker_main.cpp
    src/rgb_reader.cpp
    src/event.cpp
    src/reidentifier.cpp
)
target_link_libraries(tracker
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${HDF5_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
)

# -- Camera auto-alignment tool --
add_executable(align
    src/align_main.cpp
    src/rgb_reader.cpp
    src/event.cpp
)
target_link_libraries(align
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${HDF5_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
)

# -- Offline calibration tool --
add_executable(calibrate
    src/calibrate_main.cpp
    src/rgb_reader.cpp
    src/event.cpp
)
target_link_libraries(calibrate
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${HDF5_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
)
