cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
add_definitions(-w)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# please change this to the path where you put libtorch
# TODO: torch path should not hard coded in cmake file
# list(APPEND CMAKE_PREFIX_PATH "/home/fuzb/lib/libtorch-1.9-cu111/")
list(APPEND CMAKE_PREFIX_PATH "/home/fuzb/lib/libtorch-1.13/libtorch/")

project(GNNmini)
set(ENV{LANG} "C")
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
option (CUDA_ENABLE "DEFAULT ENABLE CUDA" ON)
message(STATUS "NeutronStar buid with CUDA ${CUDA_ENABLE}")

find_package(PythonInterp REQUIRED)
find_package(MPI REQUIRED)
find_package(Torch REQUIRED)

if (CUDA_ENABLE)
        if (NOT DEFINED TORCH_CUDA_LIBRARIES) 
                message(FATAL_ERROR "libtorch does not support gpu, please download the gpu version of libtorch.")
        endif()
        add_definitions(-DCUDA_ENABLE)
        find_package(CUDA REQUIRED)
        add_subdirectory(cuda)
        set(EXTRA_LIBS ${EXTRA_LIBS} cuda_propagate)
endif()

#exec_program("mpic++ -showme:compile" OUTPUT_VARIABLE MPI_COMPILE_FLAGS)
#exec_program("mpic++ -showme:incdirs" OUTPUT_VARIABLE MPI_INCDIRS)
#exec_program("mpic++ -showme:link" OUTPUT_VARIABLE MPI_LINK_FLAGS)
#exec_program("mpic++ -showme:libdirs" OUTPUT_VARIABLE MPI_LIBDIRS)
#exec_program("mpic++ -showme:libs" OUTPUT_VARIABLE MPI_LIBS)

#string(REPLACE " " ";" MPI_INCDIRS "${MPI_INCDIRS}")
#string(REPLACE " " ";" MPI_LINK_FLAGS "${MPI_LINK_FLAGS}")
#string(REPLACE " " ";" MPI_LIBDIRS "${MPI_LIBDIRS}")
#string(REPLACE " " ";" MPI_LIBS "${MPI_LIBS}")

# Expected directory structure.
set(NEUTRONSTAR_BUILD_SUPPORT_DIR "${CMAKE_SOURCE_DIR}/build_support")
set(NEUTRONSTAR_CLANG_SEARCH_PATH "/usr/local/bin" "/usr/bin" "/usr/local/opt/llvm/bin" "/usr/local/opt/llvm@8/bin"
        "/usr/local/Cellar/llvm/8.0.1/bin")

# clang-format
find_program(CLANG_FORMAT_BIN
        NAMES clang-format clang-format-8
        HINTS ${NEUTRONSTAR_CLANG_SEARCH_PATH})
if ("${CLANG_FORMAT_BIN}" STREQUAL "CLANG_FORMAT_BIN-NOTFOUND")
    message(WARNING " NeutronStar/main couldn't find clang-format.")
else()
    message(STATUS " NeutronStar/main found clang-format at ${CLANG_FORMAT_BIN}")
endif()

# set lint files
# TODO: add lint
file(GLOB_RECURSE NEUTRONSTAR_LINT_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h,"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp,"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp,"
        "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h,"
        "${CMAKE_CURRENT_SOURCE_DIR}/core/*.hpp,"
        "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp,"
        "${CMAKE_CURRENT_SOURCE_DIR}/toolkits/*.h,"
        "${CMAKE_CURRENT_SOURCE_DIR}/toolkits/*.hpp,"
        "${CMAKE_CURRENT_SOURCE_DIR}/toolkits/*.cpp,"
	"${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp,"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.h,"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.hpp,"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cpp,"
        "${CMAKE_CURRENT_SOURCE_DIR}/dep/gemini/*.hpp,"
        )

# set format dirs
string(CONCAT NEUTRONSTAR_FORMAT_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/src,"
        "${CMAKE_CURRENT_SOURCE_DIR}/core,"
	"${CMAKE_CURRENT_SOURCE_DIR}/dep/gemini,"
        "${CMAKE_CURRENT_SOURCE_DIR}/comm,"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda,"
        "${CMAKE_CURRENT_SOURCE_DIR}/toolkits,"
        )

# runs clang format and updates files in place.
add_custom_target(format ${NEUTRONSTAR_BUILD_SUPPORT_DIR}/run_clang_format.py
        ${CLANG_FORMAT_BIN}
        ${NEUTRONSTAR_BUILD_SUPPORT_DIR}/clang_format_exclusions.txt
        --source_dirs
        ${NEUTRONSTAR_FORMAT_DIRS}
        --fix
        # --quiet
        )

set(METIS_INCLUDE /home/fuzb/lib/metis/include/)
set(METIS_LIB /home/fuzb/lib/metis/lib/libmetis.so)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${METIS_INCLUDE}
./core
./comm
./cuda
./dep/gemini
${MPI_INCLUDE_PATH}
)

# setting debug flags
# set(CMAKE_CXX_FLAGS "-O3 -Wall -std=c++14 -g -fopenmp -march=native")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++14 -fopenmp -march=native -O0 -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls")
# set(CMAKE_CXX_FLAGS "-O0 -Wall -std=c++14 -g -fopenmp -march=native")
# set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -std=c++14 -g -fopenmp -march=native")

set(CMAKE_CXX_FLAGS "-O0 -Wall -std=c++17 -g -fopenmp -march=native -lcublas -lcudart")
# set(CMAKE_CXX_FLAGS "-O0 -Wall -std=c++17 -g -fopenmp -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -std=c++17 -fopenmp -O0 -march=native -fno-omit-frame-pointer -fno-optimize-sibling-calls")


#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++14 -fopenmp -march=native -O0 -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls")

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")

set(NEUTRONSTAR_SRC_FILES
        "toolkits/main.cpp"
	"core/GraphSegment.cpp"
	"comm/network.cpp"
)

set(NEUTRONSTAR_TEST_SRC_FILES
	"test/testcsr.cpp"
        "core/GraphSegment.cpp"
        "comm/network.cpp"
)

message(STATUS "CUDA_LIBRARIES = ${CUDA_LIBRARIES}.")

add_executable(nts ${NEUTRONSTAR_SRC_FILES})
if (CUDA_ENABLE)
        target_link_libraries(nts  ${TORCH_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS} numa ${CUDA_LIBRARIES} ${METIS_LIB} ${NCCL_LIBRARIES} cuda_propagate)
else()
        target_link_libraries(nts  ${TORCH_LIBRARIES} ${MPI_LIBRARIES} ${METIS_LIB} numa)
endif()

# add_executable(nts_test ${NEUTRONSTAR_TEST_SRC_FILES})
# if (CUDA_ENABLE)
#         target_link_libraries(nts_test  ${TORCH_LIBRARIES} ${MPI_LIBRARIES} numa ${CUDA_LIBRARIES} ${METIS_LIB} cuda_propagate)
# else()
#         target_link_libraries(nts_test  ${TORCH_LIBRARIES} ${MPI_LIBRARIES}  ${METIS_LIB} numa)
# endif()


# quick test for debugging purpose
# mpiexec -np $1 ./build/nts $2
# releated issue https://github.com/google/sanitizers/issues/629
# we have to export those variables to prevent cuda panic while using address sanitizer
