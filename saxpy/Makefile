all: x86 riscv

#export LD_LIBRARY_PATH=$OPENCL_NATIVE/lib
#export PKG_CONFIG_PATH=$OPENCL_NATIVE/lib/pkgconfig

demo_cross: saxpy.c Makefile
	g++ -g -gdwarf-2 -fno-rtti -DDEBUGPOCL_LLVM_API=1 saxpy.c `pkg-config --libs --cflags pocl` -odemo_cross -lhwloc -L$(OPENCL_NATIVE)/lib -lclang -lLLVM-3.7 -lstdc++ -lm -lltdl

pk: demo_cross
	mkdir -p pocl
	./demo_cross
	llc -march=riscv64 -mcpu=RV64IMAFDXhwacha -o=kernel.rv.S pocl/*/*/simple_demo/*/target_parallel.bc
	riscv64-unknown-elf-gcc -Wa,-march=RVIMAFDXhwacha -o saxpy.pk.riscv glue.c kernel.rv.S

clean:
	rm -rf input_*.h wg_array.h kernel.cl.bin kernel.rv.S glue.c demo_cross pocl saxpy.pk.riscv
