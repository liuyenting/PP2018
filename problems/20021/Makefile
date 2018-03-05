all: main main_ref

main: main.c DRS.o
	gcc -std=c99 -Ofast -march=native -Wa,-q main.c DRS.o -o main

DRS.o: DRS.c
	gcc -std=c99 -Ofast -march=native -Wa,-q DRS.c -c -o DRS.o

main_ref: main.c DRS_ref.o
	gcc -std=c99 -Ofast -march=native -Wa,-q main.c DRS_ref.o -o main_ref

DRS_ref.o: DRS_ref.c
	gcc -std=c99 -Ofast -march=native -Wa,-q DRS_ref.c -c -o DRS_ref.o

clean:
	rm *.o main main_ref
