FC  = ifort
CC  = icc
AR  = ar

OMPFLAGS = -openmp
OPT =
DEBUGFLAGS =  #-g
ARCHFLAGS =
FCFLAGS   =  $(OPT) $(ARCHFLAGS) $(DEBUGFLAGS) -coarray -module ./$(COMPILER)
MPIFLAGS   = -mt_mpi
CFLAGS   =  -Ofast -std=c99 -xHost -fno-alias -openmp -fma
ASFLAGS  = -gdwarf-2
CPPFLAGS =
DEFINES  = -D_GNU_SOURCE
LFLAGS   = -lmpfr -lgmp
INCLUDES =
LIBS     =


