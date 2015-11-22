all: pk

demo_cross: daxpy.c Makefile
	mkdir -p pocl
	g++ -g -gdwarf-2 -O0 -fno-rtti -DDEBUGPOCL_LLVM_API=1 daxpy.c `pkg-config --libs --cflags pocl` -odemo_cross -lhwloc -L$(OPENCL_NATIVE)/lib -lclang -lLLVM-3.7 -lstdc++ -lm -lltdl

pk: kernel.rv.S glue.c
	riscv64-unknown-elf-gcc -Wa,-march=RVIMAFDXhwacha -o daxpy.pk.riscv glue.c kernel.rv.S

crt.o: ../common
	riscv64-unknown-elf-gcc -static -Wa,-march=RVIMAFDXhwacha -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -DPREALLOCATE=1 -DHOST_DEBUG=0 -D__ASSEMBLY__=1 \
		                     -c -I../env -I../common ../common/crt.S -o crt.o
syscalls.o: ../common
	riscv64-unknown-elf-gcc -static -Wa,-march=RVIMAFDXhwacha -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -DPREALLOCATE=1 -DHOST_DEBUG=0 \
	                      -c -I../env -I../common ../common/syscalls.c -o syscalls.o

kernel.rv.S: demo_cross pocl
	rm -rf pocl/*
	./demo_cross
	llc -march=riscv64 -mcpu=RV64IMAFDXhwacha -o=kernel.rv.S pocl/*/*/daxpy/*/target_parallel.bc

glue.c: kernel.rv.S
	#do nothing

bare: kernel.rv.S crt.o syscalls.o glue.c
	riscv64-unknown-elf-gcc -static -Wa,-march=RVIMAFDXhwacha -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -DPREALLOCATE=1 -DHOST_DEBUG=0 \
	                      -c -I../env -I../common -I. glue.c -o glue.o
	riscv64-unknown-elf-gcc -static -Wa,-march=RVIMAFDXhwacha -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -DPREALLOCATE=1 -DHOST_DEBUG=0 -D__ASSEMBLY__=1 \
		                     -c -I../env -I../common -I. kernel.rv.S -o kernel.rv.o
	riscv64-unknown-elf-gcc -T ../common/test.ld -I../env -I../common -I. glue.o syscalls.o crt.o kernel.rv.o -o daxpy.riscv -nostdlib -nostartfiles -ffast-math -lc -lgcc

hex: daxpy.riscv
	elf2hex 16 2097152 daxpy.riscv > daxpy.riscv.hex

clean:
	rm -rf input_*.h wg_array.h kernel.cl.bin kernel.rv.S glue.c *.o demo_cross pocl daxpy.pk.riscv daxpy.riscv daxpy.riscv.hex
