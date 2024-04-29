cmake_minimum_required( VERSION 3.10.2 )

project( ORBEXTRACT )

set( CMAKE_CXX_FLAGS "-std=c++11" )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(orb_cv orb_cv.cpp)

target_link_libraries( orb_cv ${OpenCV_LIBS} )
