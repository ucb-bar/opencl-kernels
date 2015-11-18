all: x86 riscv

#export LD_LIBRARY_PATH=$OPENCL_NATIVE/lib
#export PKG_CONFIG_PATH=$OPENCL_NATIVE/lib/pkgconfig

demo_cross: saxpy.c Makefile
	g++ -g -gdwarf-2 -fno-rtti -DDEBUGPOCL_LLVM_API=1 saxpy.c `pkg-config --libs --cflags pocl` -odemo_cross -lhwloc -L$(OPENCL_NATIVE)/lib -lclang -lLLVM-3.7 -lstdc++ -lm -lltdl

pk: demo_cross
	mkdir -p pocl
	./demo_cross
	llc -march=riscv64 -mcpu=RV64IMAFDXhwacha -o=kernel.rv.S pocl/*/*/simple_demo/*/target_parallel.bc
	#llc --debug -march=riscv64 -mcpu=RV64IMAFDXhwacha -o=kernel.rv.S pocl/*/*/

clean:
	rm -rf input_*.h kernel.cl.bin demo_cross pocl
