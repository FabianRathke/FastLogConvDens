MATLABDIR = /usr/local/MATLAB/R2016a/
FILELOC = /home/fabian/Documents/Arbeit/Code/MyCode/LogConcave/code/newtonInit.mat
#MATLABDIR = /server/opt/archive/matlab2013
#FILELOC = /export/home/frathke/Documents/Code/MyCode/LogConcave/code/cudaWorkspace.mat
MATLABINCLUDE = $(MATLABDIR)/extern/include/
MATLABLIB = $(MATLABDIR)/bin/glnxa64/
NUMBEROFCORES = 4

#FLAGSGCC = -O3 -std=c++11 -fPIC -pedantic-errors -Wall -fopenmp -lm -D NUMCORES=$(NUMBEROFCORES)
FLAGSGCC = -O3 -funroll-loops -ftree-vectorize -march=native -std=c99 -fPIC -pedantic-errors -fopenmp -Wall -Wextra -D NUMCORES=$(NUMBEROFCORES)

clean:
	rm *.o *.mexa64

all: newton debug

newton: makeGridC.o CNS.o calcGradAVX.o calcGradC.o preCondGradAVXC.o calcGradFastFloatC.o calcGradFastC.o
	gcc newtonTest.c -o runFull -I$(MATLABINCLUDE) -L$(MATLABLIB) $(FLAGSGCC) -Wl,-rpath,$(MATLABLIB) -DFILELOC='"$(FILELOC)"' $^ -lm -fopenmp -lmx -lmat -lmex 

debug: makeGridC.o CNS.o calcGradAVX.o calcGradC.o preCondGradAVXC.o calcGradFastFloatC.o calcGradFastC.o
	gcc -g newtonTest.c -o runDebug -I$(MATLABINCLUDE) -L$(MATLABLIB) $(FLAGSGCC) -O0 -Wl,-rpath,$(MATLABLIB) -DFILELOC='"$(FILELOC)"' $^ -lm -fopenmp -lmx -lmat -lmex 

newtonInit: makeGridC.o calcGradFullAVXC.o calcGradFloatC.o CNS.o
	gcc -g newtonTestInit.c -o runInit -I$(MATLABINCLUDE) -L$(MATLABLIB) $(FLAGSGCC) -Wl,-rpath,$(MATLABLIB) -DFILELOC='"$(FILELOC)"' $^ -lm -fopenmp -lmx -lmat -lmex 

%.o: %.c
	gcc -g -c $(FLAGSGCC) -O0 -I$(MATLABINCLUDE) $<	

