cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)
project(rotary_pos_emb_bench LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 14)

add_executable(rotary_pos_emb_bench benchmark.asc)
target_include_directories(rotary_pos_emb_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(rotary_pos_emb_bench PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)
target_compile_options(rotary_pos_emb_bench PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

add_executable(rotary_pos_emb_validate validate.asc)
target_include_directories(rotary_pos_emb_validate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(rotary_pos_emb_validate PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)
target_compile_options(rotary_pos_emb_validate PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
