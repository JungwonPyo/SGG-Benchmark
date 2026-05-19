cmake_minimum_required(VERSION 3.8)
project(scene_understanding_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

set(msg_files
  "msg/BoundingBox3D.msg"
  "msg/CameraModel.msg"
  "msg/DetectedObject3D.msg"
  "msg/SceneRelation.msg"
  "msg/SituationHypothesis.msg"
  "msg/SceneContext.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()