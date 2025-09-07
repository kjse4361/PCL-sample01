cmake_minimum_required(VERSION 3.8)
project(samples)

# set(CMAKE_CXX_FLAGS_DEBUG "--coverage -g -D_DEEPDEBUG=10 -D_DEBUG=9 -lgcov -D_AUTO_PTM")
set(CMAKE_CXX_FLAGS_DEBUG "-g -D_DEEPDEBUG=10 -D_DEBUG=9 -D_AUTO_PTM")
set(CMAKE_CXX_FLAGS_RELEASE "-O -O2 -DPCL_ENABLE_SSE=true -D_USE_OMP -D_DEBUG=5 -D_AUTO_PTM")

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)

find_package(ament_cmake REQUIRED)
find_package(pcl_conversions REQUIRED)
# find_package(PCL REQUIRED)

include_directories(
  include
  ${PROJECT_SOURCE_DIR}/../../../include
  ${PCL_INCLUDE_DIRS}
  ${pcl_conversions_INCLUDE_DIRS}
  ${pcl_ros_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

message("CMAKE_BUILD_TYPE=" ${CMAKE_BUILD_TYPE})
message("PCL_LIBRARIES=" ${PCL_LIBRARIES})

#===============================================================================
# Redirector Program
#

add_executable(sample01
  src/sample01.cpp
)
ament_target_dependencies(sample01
  EIGEN3
  PCL
)
target_link_libraries(sample01
  ${PCL_LIBRARIES}
  stdc++fs
  boost_program_options
  boost_thread
)

#===============================================================================


#===============================================================================
# Proximity Program Install
#

# install(TARGETS
#   ${PROJECT_NAME}
#   DESTINATION lib/${PROJECT_NAME}
# )

#===============================================================================

ament_package()
