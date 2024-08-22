
using Kokkidio::detail::pow;
using Kokkidio::detail::sqrt;

scalar
	vNorm { sqrt( pow(v_d[2 * i], 2) + pow(v_d[2 * i + 1], 2) ) },
	chezyFac = phys::g * pow(n_d[i], 2) / pow(d_d[i], 1./3),
	fricFac = chezyFac * vNorm;
chezyFac /= d_d[i];

for (Index row = 0; row<2; ++row){
	flux_out_d[3 * i + row + 1] =
	(flux_in_d[3 * i + row + 1] - fricFac * v_d[2 * i + row] ) /
	( 1 + chezyFac * ( vNorm + pow(v_d[2 * i + row], 2) / vNorm ) );
}
