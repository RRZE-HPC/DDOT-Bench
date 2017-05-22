# =======================================================================================
#
#      Filename:  Makefile
#
#      Description:  Central Makefile
#
#      Date: Jun 24, 2014
#
#      Author:  Jan Treibig (jt), jan.treibig@gmail.com
#      Project:  AREVA CAF / MPI study
#
# =======================================================================================
#
include config.mk

export COMPILER = $(strip $(TOOLCHAIN))

SRC_DIR    = ./src
MAKE_DIR   = ./make

#DO NOT EDIT BELOW

TARGET = DDOT-$(COMPILER)

include $(MAKE_DIR)/include_$(COMPILER).mk

BUILD_DIR  = ./$(COMPILER)
Q         ?= @
VPATH     = $(SRC_DIR)
SRC       = $(wildcard $(SRC_DIR)/*.F90)
OBJ       = $(patsubst $(SRC_DIR)/%.F90, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.F90))
OBJ       += $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.c))
OBJ       += $(patsubst $(SRC_DIR)/%.s, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.s))
ASM       += $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.s,$(wildcard $(SRC_DIR)/*.c))
INCLUDES  += -I$(BUILD_DIR)/ -I./includes

LIBS :=  $(LIBS)
CPPFLAGS += $(INCLUDES) $(DEFINES)

ifeq ($(strip $(ENABLE_MPI)),YES)
FC = $(MPIFC)
FCFLAGS += $(MPIFLAGS)
DEFINES += -D_MPI
endif

ifeq ($(strip $(ENABLE_OMP)),YES)
FCFLAGS := $(OMPFLAGS) $(FCFLAGS)
DEFINES += -D_OMP
endif

DEFINES += -DTIMINGLEV=$(strip $(TIMER_LEVEL))

all:  $(TARGET)

$(TARGET): $(BUILD_DIR) $(OBJ)
	@echo "===>  LINK  $@"
	$(Q)$(CC)  $(LFLAGS) $(CPPFLAGS) $(CFLAGS) -o $@ $(OBJ) $(LIBS)

$(BUILD_DIR):
	@mkdir $(BUILD_DIR)

$(BUILD_DIR)/%.o:  %.F90
	@echo "===>  COMPILE  $@"
	$(Q)$(FC) -c  $(CPPFLAGS)  $(FCFLAGS) $< -o $@

$(BUILD_DIR)/%.o:  %.c
	@echo "===>  COMPILE  $@"
	$(Q)$(CC) -c  $(CPPFLAGS)  $(CFLAGS) $< -o $@

$(BUILD_DIR)/%.o:  %.s
	@echo "===>  COMPILE  $@"
	$(Q)$(CC) -c  $(CPPFLAGS)  $(CFLAGS) $< -o $@

$(BUILD_DIR)/%.s: %.c
	@echo "===>  COMPILE  $@"
	$(Q)$(CC) -c  $(CPPFLAGS)  $(CFLAGS) $< -S -fsource-asm -masm=intel -o $@

.PHONY: clean distclean

asm: $(BUILD_DIR) $(ASM)

clean:
	@echo "===>  CLEAN"
	@rm -rf $(BUILD_DIR)

distclean: clean
	@echo "===>  DIST CLEAN"
	@rm -f $(TARGET)


include  $(MAKE_DIR)/dep.mk
