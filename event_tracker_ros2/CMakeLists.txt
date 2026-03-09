cmake_minimum_required(VERSION 3.8)
project(event_tracker)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Find dependencies ----
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
set(HDF5_PREFER_PARALLEL FALSE)
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# ---- Generate custom messages / services ----
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Event.msg"
  "msg/EventArray.msg"
  "msg/TrackingStatus.msg"
  "srv/Calibrate.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

# After generating messages, we need the typesupport for our own targets
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ---- Core library (event I/O, RGB reader, re-ID) ----
add_library(event_tracker_core SHARED
  src/event.cpp
  src/rgb_reader.cpp
  src/reidentifier.cpp
)
target_include_directories(event_tracker_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(event_tracker_core
  ${OpenCV_LIBS}
  Eigen3::Eigen
  ${HDF5_LIBRARIES}
  ${HDF5_CXX_LIBRARIES}
)
ament_target_dependencies(event_tracker_core rclcpp)

# ---- H5 Event Publisher node ----
add_executable(h5_event_publisher src/h5_event_publisher_node.cpp)
target_link_libraries(h5_event_publisher event_tracker_core ${cpp_typesupport_target})
ament_target_dependencies(h5_event_publisher
  rclcpp std_msgs sensor_msgs cv_bridge image_transport
)

# ---- Tracker node ----
add_executable(tracker_node src/tracker_node.cpp)
target_link_libraries(tracker_node event_tracker_core ${cpp_typesupport_target})
ament_target_dependencies(tracker_node
  rclcpp std_msgs sensor_msgs geometry_msgs cv_bridge image_transport
)

# ---- Calibrate node ----
add_executable(calibrate_node src/calibrate_node.cpp)
target_link_libraries(calibrate_node event_tracker_core ${cpp_typesupport_target})
ament_target_dependencies(calibrate_node
  rclcpp std_msgs sensor_msgs cv_bridge
)

# ---- Align node ----
add_executable(align_node src/align_node.cpp)
target_link_libraries(align_node event_tracker_core ${cpp_typesupport_target})
ament_target_dependencies(align_node
  rclcpp std_msgs sensor_msgs cv_bridge
)

# ---- Install ----
install(TARGETS event_tracker_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  h5_event_publisher
  tracker_node
  calibrate_node
  align_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
