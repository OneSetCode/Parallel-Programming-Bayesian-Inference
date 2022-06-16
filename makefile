MPI_LSTDFLG = -lstdc++ -llapack -lcblas -lm -lgsl -lgslcblas -llapack_atlas
MPI_INCLUDE = -I/usr/include/ -I/usr/include/atlas/
MPI_LIB = -L/usr/lib/atlas/ -L/usr/lib/ -L/usr/lib/atlas-base/
MPI_OBJS = main

all:	${MPI_OBJS}
	rm -f *.o

matrices.o: matrices.cpp matrices.h
	gcc -g -c matrices.cpp -o matrices.o ${MPI_INCLUDE}

regmodels.o: regmodels.cpp regmodels.h
	gcc -g -c regmodels.cpp -o regmodels.o ${MPI_INCLUDE}

Rfunctions.o: Rfunctions.cpp matrices.h regmodels.h Rfunctions.h
	gcc -g -c Rfunctions.cpp -o Rfunctions.o ${MPI_INCLUDE}

main.o: main.cpp matrices.h regmodels.h Rfunctions.h
	mpic++ -g -c main.cpp -o main.o ${MPI_INCLUDE}

main: main.o matrices.o regmodels.o Rfunctions.o
	mpic++ main.o Rfunctions.o regmodels.o matrices.o -o main ${MPI_LIB} ${MPI_LSTDFLG}

clean:
	rm -f *.o
	rm -f ${MPI_OBJS}