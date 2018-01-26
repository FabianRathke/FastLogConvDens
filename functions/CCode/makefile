#MATLABDIR = /opt/MATLAB/R2016b
MATLABDIR = /usr/local/MATLAB/R2016a/
MEXFILE = $(MATLABDIR)/bin/mex
MATLABINCLUDE = $(MATLABDIR)/extern/include/

#FLAGSGCC = -03 -std=c89 -fPIC -pedantic-errors -Wall -fopenmp -lm
FLAGSGCC = -O3 -funroll-loops -ftree-vectorize -march=native -std=c99 -fPIC -pedantic-errors -Wall -fopenmp -lm
FLAGSMATLAB = -fopenmp -std=c99 -pedantic-errors -Wall -march=native -Ofast

clean:
	rm *.o *.mexa64

all: kernelDens calcExactIntegral evalObjective bfgsFull bfgsInit

kernelDens:
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" kernelDensC.c

bfgsInit: makeGridC.o calcGradFullAVXC.o calcGradFloatC.o CNS.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" $@C.c $^

bfgsFull: makeGridC.o CNS.o calcGradAVX.o calcGradC.o preCondGradAVXC.o calcGradFastFloatC.o calcGradFastC.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" $@C.c $^

calcExactIntegral: calcExactIntegralC.o recalcParamsC.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" -lmwlapack -lmwblas calcExactIntegral.c $^

evalObjective: evalObjectiveC.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" $@.c $<

%.o: %.c
	gcc -c $(FLAGSGCC) -I$(MATLABINCLUDE) $<
