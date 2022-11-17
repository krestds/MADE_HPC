
DEBAG=-g
OPT=-O3

all: matrix_mul

matrix_mul: main.o matrix.o
	g++ main.o matrix.o -o matrix_mul

main.o: main.cpp
	g++ $(DEBAG) $(OPT) -c main.cpp

matrix.o: matrix.cpp
	g++ $(DEBAG) $(OPT) -c matrix.cpp

clean:
	rm -rf *.o matrix_mul


