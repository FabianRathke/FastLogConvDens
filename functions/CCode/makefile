#MATLABDIR = /opt/MATLAB/R2016b
MATLABDIR = /usr/local/MATLAB/R2016a/
MEXFILE = $(MATLABDIR)/bin/mex
MATLABINCLUDE = $(MATLABDIR)/extern/include/
NUMBEROFCORES = 2

#FLAGSGCC = -03 -std=c89 -fPIC -pedantic-errors -Wall -fopenmp -lm -D NUMCORES=$(NUMBEROFCORES)
FLAGSGCC = -Ofast -funroll-loops -ftree-vectorize -march=native -std=c99 -fPIC -pedantic-errors -Wall -fopenmp -lm -D NUMCORES=$(NUMBEROFCORES)
FLAGSMATLAB = -fopenmp -std=c99 -pedantic-errors -Wall -march=native -Ofast

clean:
	rm *.o *.mexa64

all: kernelDens calcNewtonStep calcExactIntegral makeGrid calcGradFloat calcGradFloatAVX calcGrad evalObjective preCondGrad preCondGradFloat preCondGradAVX calcGradFast calcGradFastFloat calcGradFull

kernelDens:
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB) -D NUMCORES=$(NUMBEROFCORES)" LDFLAGS="\$$LDFLAGS -fopenmp" kernelDensC.c

calcNewtonStep:
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB) -D NUMCORES=$(NUMBEROFCORES)" LDFLAGS="\$$LDFLAGS -fopenmp" calcNewtonStepC.c

bfgsInit: makeGridC.o calcGradFullAVXC.o calcGradFloatC.o CNS.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB) -D NUMCORES=$(NUMBEROFCORES)" LDFLAGS="\$$LDFLAGS -fopenmp" $@C.c $^

calcNewtonStepFloat:
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB) -D NUMCORES=$(NUMBEROFCORES)" LDFLAGS="\$$LDFLAGS -fopenmp" calcNewtonStepFloatC.c

calcExactIntegral: calcExactIntegralC.o recalcParamsC.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" -lmwlapack -lmwblas calcExactIntegral.c calcExactIntegralC.o recalcParamsC.o

makeGrid: makeGridC.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" makeGrid.c makeGridC.o

calcGradFloat: calcGradAVX.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" $@.c $^

calcGradFloatAVX: calcGradFullAVXC.o calcGradFloatC.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" $@.c $^

calcGrad: calcGradC.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" $@.c $<

evalObjective: evalObjectiveC.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" $@.c $<

preCondGrad: preCondGradC.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" $@.c $<

preCondGradFloat: preCondGradFloatC.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" $@.c $<

preCondGradAVX: preCondGradAVXC.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" $@.c $<

calcGradFast: calcGradFastC.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" $@.c $<

calcGradFastFloat: calcGradFastFloatC.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" $@.c $<

calcGradFull: calcGradFullC.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" $@.c $<

newtonInit: makeGridC.o calcGradFullAVXC.o calcGradFloatC.o
	$(MEXFILE) CFLAGS="\$$CFLAGS $(FLAGSMATLAB)" LDFLAGS="\$$LDFLAGS -fopenmp" newtonBFGSLInitC.c $^

%.o: %.c
	gcc -c $(FLAGSGCC) -I$(MATLABINCLUDE) $<
