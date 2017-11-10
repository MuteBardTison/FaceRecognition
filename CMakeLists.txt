cmake_minimum_required(VERSION 3.2)
project(FaceRecognition)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED COMPONENTS system filesystem )

include_directories( include ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})  #${Eigen_INCLUDE_DIRS}

set(SOURCE_FILES FaceRecognition.cpp)
add_executable(FaceRecognition ${SOURCE_FILES})
target_link_libraries(FaceRecognition ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY})

