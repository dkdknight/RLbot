cmake_minimum_required(VERSION 3.15)
project(RLBotBridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set BakkesMod SDK path - adjust this to your BakkesMod SDK location
set(BAKKESMOD_SDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../bakkesmodsdk" CACHE PATH "Path to BakkesMod SDK")

# Plugin source files
set(SOURCES
    RLBotBridge.cpp
)

set(HEADERS
    RLBotBridge.h
)

# Create the plugin DLL
add_library(RLBotBridge SHARED ${SOURCES} ${HEADERS})

# Include BakkesMod SDK headers
target_include_directories(RLBotBridge PRIVATE
    ${BAKKESMOD_SDK_PATH}/include
)

# Link BakkesMod SDK and Windows socket library
target_link_libraries(RLBotBridge
    ${BAKKESMOD_SDK_PATH}/lib/pluginsdk.lib
    ws2_32
)

# Set output directory for the compiled plugin
set_target_properties(RLBotBridge PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Post-build: Copy to BakkesMod plugins directory (optional)
# Uncomment and adjust path if you want automatic deployment
# add_custom_command(TARGET RLBotBridge POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy
#     $<TARGET_FILE:RLBotBridge>
#     "$ENV{APPDATA}/bakkesmod/bakkesmod/plugins/"
# )
