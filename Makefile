
bench = daxpy dfilter dgemm-single hgemm-single hsaxpy hsfilter mask-dfilter mask-hsfilter mask-sdfilter mask-sfilter saxpy sdaxpy sdfilter sdgemm-single sfilter sgemm-single

all: $(bench)


$(bench):
	echo $@
	echo $@
	echo $@
	echo $@
	cd $@ && make mali && cd ..



.PHONY: $(bench)
