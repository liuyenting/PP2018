all: main ref

main: main.c
	gcc -std=c99 -O3 -march=native main.c -o main -lm

ref: main_ref.c
	gcc -std=c99 -O3 -march=native main_ref.c -o main_ref -lm

test:
	time -p ./main_ref < 1.in > 1.ref.out
	time -p ./main < 1.in > 1.out
