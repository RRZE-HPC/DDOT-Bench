FC  = gfortran
CC  = gcc
AR  = ar

OMPFLAGS = -fopenmp
OPT =
DEBUGFLAGS =  #-g
ARCHFLAGS =
FCFLAGS   =  $(OPT) $(ARCHFLAGS) $(DEBUGFLAGS) -coarray -module ./$(COMPILER)
MPIFLAGS   = -mt_mpi
CFLAGS   =  -O3 -ffast-math -std=c99 -march=native -funroll-loops $(OMPFLAGS)
ASFLAGS  = -gdwarf-2
CPPFLAGS =
DEFINES  = -D_GNU_SOURCE
INCLUDES =
LIBS     = -lmpfr -lgmp -lm


