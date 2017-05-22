FC  = ifort
CC  = icc
AR  = ar

OMPFLAGS = -openmp
OPT =
DEBUGFLAGS =  #-g
ARCHFLAGS =
FCFLAGS   =  $(OPT) $(ARCHFLAGS) $(DEBUGFLAGS) -coarray -module ./$(COMPILER)
MPIFLAGS   = -mt_mpi
CFLAGS   =  -O2 -std=c99 -Igmp-6.0.0-inst/include/ -Impfr-3.1.2-inst/include/ -xHost -fno-alias -openmp
ASFLAGS  = -gdwarf-2
CPPFLAGS =
DEFINES  = -D_GNU_SOURCE
LFLAGS   = -Lmpfr-3.1.2-inst/lib/ -lmpfr -Lgmp-6.0.0-inst/lib/ -lgmp
INCLUDES =
LIBS     =


