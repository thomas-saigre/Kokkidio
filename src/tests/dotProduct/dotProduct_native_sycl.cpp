#include "dotProduct.hpp"

namespace Kokkidio::gpu
{

using K = Kernel;
constexpr Target dev { Target::device };

template<>
scalar dotProduct<dev, K::cstyle_blockbuf>(
	const MatrixXs& m1, const MatrixXs& m2, int nRuns
){
	Index
		nRows { m1.cols() },
		nCols { m1.cols() };
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

	auto m1_d { allocAndCopy(m1) };
	auto m2_d { allocAndCopy(m2) };

	/* adapted from 
	 https://github.khronos.org/SYCL_Reference/iface/reduction-variables.html
	 */

	/* no idea what this one is for, 
	 * because it will never automatically contain the actual sum result,
	 * but apparently that's how the buffer is created. */
	scalar sumResult {0};
	
	for (int iter = 0; iter < nRuns; ++iter){
		sumResult = 0;
		/* I didn't find a way to reset this buffer,
		 * as get_access(cgh) caused a segfault.
		 * So instead, we construct it in every iteration
		 * and hope that the overhead is small... */
		sycl::buffer<scalar> sumBuf { &sumResult, 1 };

		queue.submit( [&](sycl::handler& cgh){
			// sumBuf.get_access(cgh)[0] = 0;
			/* using the combiner sycl::plus, 
			 * we create an object with the reduction interface - 
			 * a "reduction variable" */
			auto sumReduction = sycl::reduction( sumBuf, cgh, sycl::plus<>() );
			/* the order of reduction variables (here: sumReduction) has to match 
			 * the order of reducer arguments (here: sum) of the lambda */
			cgh.parallel_for(
				sycl::range<1>{uCols},
				sumReduction,
				[=](sycl::id<1> tid, auto& sum){
					int j = tid[0];
					/* we create a stack variable to track the sum */
					scalar dot_product {0};
					for (int i = 0; i < nRows; ++i)
					{
						int idx = j * nRows + i;
						dot_product += m1_d[idx] * m2_d[idx];
					}
					/* we only write to the result variable once (per column) */
					sum += dot_product;
					// /* equivalent, and more general: */
					// sum.combine(dot_product);
				}
			);
		} ).wait();

		printd(
			"sumResult = %f\n"
			"sumBuf... = %f\n"
			, sumResult
			, sumBuf.get_host_access()[0]
		);
		sumResult = sumBuf.get_host_access()[0];
	}

	sycl::free(m1_d, queue);
	sycl::free(m2_d, queue);
	/* do we need to free the sumBuf here? */
	/* this doesn't work, because it's not convertible to a void ptr */
	// sycl::free(sumBuf, queue);

	return sumResult;
}

} // namespace Kokkidio::gpu
