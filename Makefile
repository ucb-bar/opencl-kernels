
bench = daxpy dfilter-unroll dgemm-unroll hsaxpy hsfilter-unroll hsgemm-unroll mask-dfilter-unroll mask-hsfilter-unroll mask-sdfilter-unroll mask-sfilter-unroll saxpy sdaxpy sdfilter-unroll sdgemm-unroll sfilter-unroll sgemm-unroll

all: $(bench)


$(bench):
	echo $@
	echo $@
	echo $@
	echo $@
	cd $@ && make mali && cd ..



.PHONY: $(bench)
