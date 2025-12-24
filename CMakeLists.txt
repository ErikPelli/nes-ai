cmake_minimum_required(VERSION 3.20)

project(nes-ai C)

find_program(CC65_COMPILER cl65)
if(NOT CC65_COMPILER)
    message(FATAL_ERROR "Compiler cl65 not found. Make sure cc65 is installed and present in the PATH.")
endif()

# Libraries header files (.h)
set(COMMON_INCLUDES
    -I${CMAKE_CURRENT_SOURCE_DIR}/lib/neslib
    -I${CMAKE_CURRENT_SOURCE_DIR}/lib/libfixmath-cc65/libfixmath
    -I${CMAKE_CURRENT_SOURCE_DIR}/mlp
)

# Libraries source files (.c)
set(COMMON_SOURCES
    lib/libfixmath-cc65/libfixmath/fix16.c
    lib/libfixmath-cc65/libfixmath/fix16_exp.c
    lib/neslib/crt0.s
    mlp/mlp.c
    mlp/weights.c
)

# Configure libfixmath compile options
set(COMMON_DEFINES
    # cc65 doesn't have uint64_t type
    -D FIXMATH_NO_64BIT
    # libfixmath readme file says we need 32KB+80KB (112KB) of memory, we can't afford it
    -D FIXMATH_NO_CACHE
    # NES CPU doesn't have division instructions, so we disable them
    -D FIXMATH_NO_HARD_DIVISION
    # 6502 is an 8 bit CPU, we use instructions optimized for it
    -D FIXMATH_OPTIMIZE_8BIT
)

add_custom_target(
    nes-ai-cli ALL

    # According to cc65 documentation:
    # Please note that the compiler does not support the C99 standard and never will.
    # c99 mode is actually c89 mode with a few selected C99 extensions.
    # cc65 is like c99 mode, but additional features are enabled (e.g. itoa).
    COMMAND ${CC65_COMPILER}
            -t nes -Oisr -o nes-ai-cli.nes --standard cc65 -C cli/nes.cfg
            ${COMMON_INCLUDES}
            ${COMMON_DEFINES}
            ${COMMON_SOURCES}
            ${LIB_SOURCES}
            ${ASM_SOURCES}
            cli/tileset.s
            cli/main.c

    BYPRODUCTS nes-ai-cli.nes
    COMMAND_EXPAND_LISTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building NES CLI ROM (nes-ai-cli.nes)"
)
