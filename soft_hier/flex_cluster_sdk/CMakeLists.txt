cmake_minimum_required(VERSION 3.13)

# Project name and language
project(SoftHier LANGUAGES C)

# Specify the RISC-V compiler
set(COMPILER "riscv32-unknown-elf-gcc")
set(OBJDUMPER "riscv32-unknown-elf-objdump")

# Default SRC_DIR to "app_example" if not provided
set(SRC_DIR ${SRC_DIR} CACHE STRING "Source directory")
if (NOT SRC_DIR)
    set(SRC_DIR "${CMAKE_SOURCE_DIR}/app_example")
    # Add the selected subdirectory
    add_subdirectory(${SRC_DIR})
else()
    # Add the selected subdirectory
    add_subdirectory(${SRC_DIR} ${CMAKE_BINARY_DIR}/app)
endif()


if (NOT INCLUDE_DIRS)
    set(INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/runtime/include)
endif()

# Set the linker script
set(LINKER_SCRIPT ${CMAKE_SOURCE_DIR}/runtime/flex_memory.ld)

######################################################################
## Mempool Config Integration
######################################################################

set(NUM_CORES 0 CACHE STRING "Number of mempool cores (0=legacy)")
set(NUM_GROUPS 0 CACHE STRING "Number of groups")
set(NUM_CORES_PER_TILE 4 CACHE STRING "Cores per tile")
set(BANKING_FACTOR 4 CACHE STRING "L1 banking factor")
set(L1_BANK_SIZE 1024 CACHE STRING "L1 bank size in bytes")
set(L2_BASE 2147483648 CACHE STRING "L2 base address")
set(L2_SIZE 4194304 CACHE STRING "L2 size in bytes")
set(SEQ_MEM_SIZE 512 CACHE STRING "Sequential memory per core")
set(STACK_SIZE 512 CACHE STRING "Stack size per core")
set(XQUEUE_SIZE 0 CACHE STRING "XQueue size per bank")
set(BOOT_ADDR 2684354560 CACHE STRING "Boot address")

set(START_FILE ${CMAKE_SOURCE_DIR}/runtime/flex_start.S)
set(EXTRA_SOURCES "")
set(EXTRA_DEFINES "")
set(EXTRA_LINK_FLAGS "")
set(EXTRA_DEPS "")

if(NOT NUM_CORES EQUAL 0)
    # Compute log2 values using CMake math
    # log2(NUM_CORES_PER_TILE)
    math(EXPR _tmp "${NUM_CORES_PER_TILE}")
    set(LOG2_NUM_CORES_PER_TILE 0)
    while(_tmp GREATER 1)
        math(EXPR _tmp "${_tmp} / 2")
        math(EXPR LOG2_NUM_CORES_PER_TILE "${LOG2_NUM_CORES_PER_TILE} + 1")
    endwhile()

    # log2(SEQ_MEM_SIZE)
    math(EXPR _tmp "${SEQ_MEM_SIZE}")
    set(LOG2_SEQ_MEM_SIZE 0)
    while(_tmp GREATER 1)
        math(EXPR _tmp "${_tmp} / 2")
        math(EXPR LOG2_SEQ_MEM_SIZE "${LOG2_SEQ_MEM_SIZE} + 1")
    endwhile()

    # log2(STACK_SIZE)
    math(EXPR _tmp "${STACK_SIZE}")
    set(LOG2_STACK_SIZE 0)
    while(_tmp GREATER 1)
        math(EXPR _tmp "${_tmp} / 2")
        math(EXPR LOG2_STACK_SIZE "${LOG2_STACK_SIZE} + 1")
    endwhile()

    # Derived values
    math(EXPR NUM_CORES_PER_GROUP "${NUM_CORES} / ${NUM_GROUPS}")
    math(EXPR NUM_TILES_PER_GROUP "${NUM_CORES_PER_GROUP} / ${NUM_CORES_PER_TILE}")
    math(EXPR NUM_BANKS "${BANKING_FACTOR} * ${NUM_CORES}")

    # Build define string for preprocessor
    set(MEMPOOL_DEFINES
        -DNUM_CORES=${NUM_CORES}
        -DNUM_GROUPS=${NUM_GROUPS}
        -DNUM_CORES_PER_TILE=${NUM_CORES_PER_TILE}
        -DBANKING_FACTOR=${BANKING_FACTOR}
        -DNUM_BANKS=${NUM_BANKS}
        -DNUM_CORES_PER_GROUP=${NUM_CORES_PER_GROUP}
        -DNUM_TILES_PER_GROUP=${NUM_TILES_PER_GROUP}
        -DLOG2_NUM_CORES_PER_TILE=${LOG2_NUM_CORES_PER_TILE}
        -DL1_BANK_SIZE=${L1_BANK_SIZE}
        -DL2_BASE=${L2_BASE}
        -DL2_SIZE=${L2_SIZE}
        -DSEQ_MEM_SIZE=${SEQ_MEM_SIZE}
        -DLOG2_SEQ_MEM_SIZE=${LOG2_SEQ_MEM_SIZE}
        -DSTACK_SIZE=${STACK_SIZE}
        -DLOG2_STACK_SIZE=${LOG2_STACK_SIZE}
        -DXQUEUE_SIZE=${XQUEUE_SIZE}
        -DBOOT_ADDR=${BOOT_ADDR}
    )

    # Preprocess arch.ld.c -> arch.ld
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/arch.ld
        COMMAND ${COMPILER} -P -E ${MEMPOOL_DEFINES}
                ${CMAKE_SOURCE_DIR}/runtime/mempool/arch.ld.c
                -o ${CMAKE_BINARY_DIR}/arch.ld
        DEPENDS ${CMAKE_SOURCE_DIR}/runtime/mempool/arch.ld.c
        COMMENT "Preprocessing arch.ld.c -> arch.ld"
    )
    add_custom_target(arch_ld DEPENDS ${CMAKE_BINARY_DIR}/arch.ld)

    set(EXTRA_SOURCES ${CMAKE_SOURCE_DIR}/runtime/mempool/synchronization.c)
    set(EXTRA_DEFINES ${MEMPOOL_DEFINES})
    set(EXTRA_LINK_FLAGS -L${CMAKE_BINARY_DIR})
    set(EXTRA_DEPS ${CMAKE_BINARY_DIR}/arch.ld)
endif()

add_custom_command(
    OUTPUT softhier.elf
    DEPENDS ${EXTRA_DEPS}
    COMMAND ${COMPILER} -I ${CMAKE_SOURCE_DIR}/runtime/include -I ${CMAKE_SOURCE_DIR}/runtime/mempool -I ${INCLUDE_DIRS} ${EXTRA_DEFINES} -T ${LINKER_SCRIPT} -nostartfiles -mabi=ilp32d -mcmodel=medlow -march=${RISCV_ARCH} -g -O3 -ffast-math -fno-builtin -fno-tree-vectorize -fno-common -ffunction-sections -Wl,--gc-sections ${EXTRA_LINK_FLAGS} -o softhier.elf ${START_FILE} ${EXTRA_SOURCES} ${SOURCES}
    COMMENT "Generating softhier.elf"
)

add_custom_target(build_output ALL DEPENDS softhier.elf)

add_custom_command(
    OUTPUT softhier.dump
    DEPENDS softhier.elf
    COMMAND ${OBJDUMPER} -D softhier.elf > softhier.dump
    COMMENT "Creating disassembly output.dump"
)

add_custom_target(dump ALL DEPENDS softhier.dump)
