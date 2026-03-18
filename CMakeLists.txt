cmake_minimum_required(VERSION 3.8)
project(g1_nav)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  g1_nav/waypoint_manager.py
  g1_nav/planner.py
  g1_nav/controller.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()