cmake_minimum_required(VERSION 3.20)

project(nes-ai C)

find_program(CC65_COMPILER cl65)
if(NOT CC65_COMPILER)
    message(FATAL_ERROR "Compiler cl65 not found. Make sure cc65 is installed and present in the PATH.")
endif()

# Libraries header files (.h)
set(COMMON_INCLUDES
    -I${CMAKE_CURRENT_SOURCE_DIR}/lib/neslib
    -I${CMAKE_CURRENT_SOURCE_DIR}/lib/libfixmath
    -I${CMAKE_CURRENT_SOURCE_DIR}/mlp
)

# Libraries source files (.c)
set(COMMON_SOURCES
    lib/libfixmath/fix16.c
    lib/libfixmath/fix16_exp.c
    lib/libfixmath/fix16_div.c
    lib/neslib/crt0.s
    mlp/mlp.c
    mlp/weights.c
)

# Configure libfixmath compile options
set(COMMON_DEFINES
    # cc65 doesn't have uint64_t type
    -DFIXMATH_NO_64BIT
    # libfixmath readme file says we need 32KB+80KB (112KB) of memory, we can't afford it
    -DFIXMATH_NO_CACHE
    # NES CPU doesn't have division instructions, so we disable them
    -DFIXMATH_NO_HARD_DIVISION
    # 6502 is an 8 bit CPU, we use instructions optimized for it
    -DFIXMATH_OPTIMIZE_8BIT
)

add_custom_target(
    nes-ai-cli ALL

    COMMAND ${CC65_COMPILER} -t nes -o nes-ai-cli.nes
            ${COMMON_INCLUDES}
            ${COMMON_DEFINES}
            ${COMMON_SOURCES}
            ${LIB_SOURCES}
            ${ASM_SOURCES}
            cli/main.c

    BYPRODUCTS nes-ai-cli.nes
    COMMAND_EXPAND_LISTS
    COMMENT "Building NES CLI ROM (nes-ai-cli.nes)"
)
