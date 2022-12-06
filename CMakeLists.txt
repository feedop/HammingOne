cmake_minimum_required(VERSION 3.22)

project(HammingOne LANGUAGES CXX CUDA)

include_directories(${PROJECT_SOURCE_DIR})

add_executable(${PROJECT_NAME} kernel.cu radix_sort.cu binary_trie.cu trie_search.cu)