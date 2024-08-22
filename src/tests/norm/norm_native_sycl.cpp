#include "norm.hpp"

namespace Kokkidio::gpu
{

using K = Kernel;
constexpr Target dev { Target::device };

template<>
scalar norm<dev, K::cstyle_blockbuf>(const MatrixXs& mat, int nRuns){
	int
		nRows { mat.rows() },
		nCols { mat.cols() };

	auto uCols { static_cast<std::size_t>(nCols) };

	sycl::queue queue(sycl::gpu_selector{});

	auto alloc = [&](const auto& obj_h){
		return sycl::malloc_device<scalar>(obj_h.size(), queue);
	};
	auto allocAndCopy = [&](const auto& obj_h){
		auto data_d = alloc(obj_h);
		queue.memcpy( data_d, obj_h.data(), obj_h.size() * sizeof(scalar) );
		return data_d;
	};

	auto mat_d { allocAndCopy(mat) };

	/* adapted from 
	 https://github.khronos.org/SYCL_Reference/iface/reduction-variables.html
	 */

	/* no idea what this one is for, 
	 * because it will never automatically contain the actual max result,
	 * but apparently that's how the buffer is created. */
	scalar maxResult {0};
	// sycl::buffer<scalar> maxBuf { &maxResult, 1 };
	
	for (int iter = 0; iter < nRuns; ++iter){
		maxResult = 0;
		/* I didn't find a way to reset this buffer,
		 * as get_access(cgh) caused a segfault.
		 * So instead, we construct it in every iteration
		 * and hope that the overhead is small... */
		sycl::buffer<scalar> maxBuf { &maxResult, 1 };

		// /* host side buffer reset */
		// maxBuf.set_final_data(&sumResult);
		// maxBuf.set_write_back(true);

		// /* device side buffer reset */
		// queue.submit( [&](sycl::handler& cgh){
		// 	auto maxAcc = maxBuf.get_access<sycl::access::mode::read_write>(cgh);
		// 	cgh.single_task( [=](){
		// 		maxAcc[0] = 0;
		// 	});
		// });

		queue.submit( [&](sycl::handler& cgh){
			/* using the combiner sycl::plus, 
			 * we create an object with the reduction interface - 
			 * a "reduction variable" */
			auto maxReduction = sycl::reduction( maxBuf, cgh, sycl::maximum<>() );
			/* the order of reduction variables (here: maxReduction) has to match 
			 * the order of reducer arguments (here: max) of the lambda */
			cgh.parallel_for(
				sycl::range<1>{uCols},
				maxReduction,
				[=](sycl::id<1> tid, auto& max){
					int j = tid[0];
					/* we create a stack variable to track the maximum */
					scalar norm {0};

					for (int i = 0; i < nRows; ++i){
						int idx = j * nRows + i;
						norm += mat_d[idx] * mat_d[idx];
					}
					max.combine( detail::sqrt(norm) );
				}
			);
		} ).wait();

		printd(
			"maxResult = %f\n"
			"maxBuf... = %f\n"
			, maxResult
			, maxBuf.get_host_access()[0]
		);
		maxResult = maxBuf.get_host_access()[0];
	}

	sycl::free(mat_d, queue);
	/* do we need to free the maxBuf here? */
	/* this doesn't work, because it's not convertible to a void ptr */
	// sycl::free(maxBuf, queue);

	return maxResult;
}

} // namespace Kokkidio::gpu
