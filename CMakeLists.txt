cmake_minimum_required(VERSION 3.24)
project(radar_fast_livo2)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# FIXME: ASAN 曾用于定位 imu_processing.cpp 的 Clock UAF（已修复，见 git log）。
# 修复后不再需要强制开启；生产 Release 构建默认不带 ASAN（1.5-2x CPU / 2x 内存
# 开销不可接受），需要排查新问题时用 -DENABLE_ASAN=ON 单独打开。
option(ENABLE_ASAN "Build with AddressSanitizer" OFF)

set(CMAKE_CXX_FLAGS_DEBUG          "-O1 -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O1 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O1 -g -DNDEBUG")

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# ARM 平台优化（Jetson Orin / TX2 等 aarch64）
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native -mtune=native -ffast-math")
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters search kdtree)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

ament_auto_find_build_dependencies()

# ── Core library (传感器适配 + 算法核心) ───────────────────────────
# 包含 FAST-LIVO2 的 VoxelMapManager（已 vendor 到 src/voxel_map.cpp）
add_library(${PROJECT_NAME}_core STATIC
    src/preprocess.cpp
    src/imu_processing.cpp
    src/voxel_map.cpp
    src/vio.cpp
)
ament_target_dependencies(${PROJECT_NAME}_core
    rclcpp
    sensor_msgs
    geometry_msgs
    visualization_msgs
    pcl_conversions
)
target_link_libraries(${PROJECT_NAME}_core
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)
target_include_directories(${PROJECT_NAME}_core PUBLIC
    include
    ${PCL_INCLUDE_DIRS}
)

# ── ROS2 node ────────────────────────────────────────────────────────
ament_auto_add_executable(${PROJECT_NAME}_node
    src/livmapper_node.cpp
)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME}_core)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS ${PROJECT_NAME}_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
ament_package()
