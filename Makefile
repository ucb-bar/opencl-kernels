
bench = daxpy dfilter dfilter-unroll dgemm-single dgemm-unroll hgemm-single hsaxpy hsfilter hsfilter-unroll hsgemm-unroll mask-dfilter mask-dfilter-unroll mask-hsfilter mask-hsfilter-unroll mask-sdfilter mask-sdfilter-unroll mask-sfilter mask-sfilter-unroll saxpy sdaxpy sdfilter sdfilter-unroll sdgemm-single sdgemm-unroll sfilter sfilter-unroll sgemm-single sgemm-unroll

all: $(bench)


$(bench):
	echo $@
	echo $@
	echo $@
	echo $@
	cd $@ && make mali && cd ..



.PHONY: $(bench)
