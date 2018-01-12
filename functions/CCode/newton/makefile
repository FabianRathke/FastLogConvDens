MATLABDIR = /usr/local/MATLAB/R2016a/
FILELOC = /home/fabian/Documents/Arbeit/Code/MyCode/LogConcave/code/newtonInit.mat
#MATLABDIR = /server/opt/archive/matlab2013
#FILELOC = /export/home/frathke/Documents/Code/MyCode/LogConcave/code/cudaWorkspace.mat
MATLABINCLUDE = $(MATLABDIR)/extern/include/
MATLABLIB = $(MATLABDIR)/bin/glnxa64/
NUMBEROFCORES = 4

#FLAGSGCC = -O3 -std=c++11 -fPIC -pedantic-errors -Wall -fopenmp -lm -D NUMCORES=$(NUMBEROFCORES)
FLAGSGCC = -O3 -funroll-loops -ftree-vectorize -march=native -std=c99 -fPIC -pedantic-errors -fopenmp -Wall -D NUMCORES=$(NUMBEROFCORES)

clean:
	rm *.o *.mexa64

all: newtonInit newton

newton: makeGridC.o calcGradFloatC.o CNS.o calcGradAVX.o
	gcc -g newtonTest.c -o runFull -I$(MATLABINCLUDE) -L$(MATLABLIB) $(FLAGSGCC) -Wl,-rpath,$(MATLABLIB) -DFILELOC='"$(FILELOC)"' $^ -lm -fopenmp -lmx -lmat -lmex 

newtonInit: makeGridC.o calcGradFullAVXC.o calcGradFloatC.o CNS.o
	gcc newtonTestInit.c -o runInit -I$(MATLABINCLUDE) -L$(MATLABLIB) $(FLAGSGCC) -Wl,-rpath,$(MATLABLIB) -DFILELOC='"$(FILELOC)"' $^ -lm -fopenmp -lmx -lmat -lmex 

%.o: %.c
	gcc -c $(FLAGSGCC) -I$(MATLABINCLUDE) $<
