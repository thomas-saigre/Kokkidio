![Kokkidio logo][logo]

*Kokkidio* is a header-only template library designed to provide interoperability between the linear algebra library [Eigen] and the performance portability framework [Kokkos]. 
Its aim is to allow the easy-to-read, succinct source code of [Eigen] to be compiled to fast machine code on all the platforms supported by [Kokkos].  



*Kokkidio* consists of three interlocking elements:
* The Data structures [`EigenView`][EigenView] and [`EigenDualView`][EigenDualView],
* parallel dispatch functions (`parallel_for` and `parallel_reduce`), and
* an iteration range class (`ParallelRange`), which all elements of the calling thread to enable Eigen's vectorisation routines in a way that's easy to read and write.

<!-- It provides data structures, parallel dispatch functions, and an iteration range class a data structure called [`EigenView`][EigenView], which  -->
Here's an example using an `EigenDualView`:

```c++
/* "Target" is an enum containing "host" and "device".
 * Generally, you can just set it to "DefaultTarget",
 * which queries Kokkos' default execution space. */
constexpr target {DefaultTarget};
using MatrixView = EigenDualView<Eigen::MatrixXd, target>;
int nRows {4}, nCols {1000};

/* The EigenDualView constructors which take sizes allocate memory 
 * on both host and target: */
MatrixView mat (nRows, nCols);

/* The EigenDualView constructors which take Eigen objects only allocate
 * memory on the target, or none at all, if target==host: */
Eigen::MatrixXd existingMat (nRows, nCols);
MatrixView mat2 {existingMat};

/* You can use Kokkos routines via the view_[host|target]() member functions, 
 * e.g. for setting all elements to 123: */
Kokkos::deep_copy(mat.view_host(), 123);

/* or you can use Eigen routines via map_[host|target](): */
mat.map_host().setRandom();

/* Copy data to the target space (does nothing if target==host): */
mat.copyToTarget();

/* To operate on it, we create a functor, which is then passed to 
 * a parallel dispatch function. Our functor here gets the sum of our data: */
auto func = KOKKOS_LAMBDA(ParallelRange<target> rng, double& sum){
	/* pass an Eigen(Dual)View or Eigen object to a ParallelRange
	 * to get the elements associated with a thread (Eigen::Block) */
	sum += rng(mat);
}

double result {0};
/* parallel_[for|reduce] in the Kokkidio namespace allow passing functions 
 * which take a ParallelRange, but otherwise they work exactly like
 * Kokkos::parallel_[for|reduce]: */
parallel_reduce<DefaultTarget>( nCols, func, redux::sum(result) );
```

*Kokkidio* is maintained by the
Chair of Water Resources Management and Modelling of Hydrosystems of the
Technische Universität Berlin,
or *wahyd* for short ([Link][wahyd]).
It is distributed under a [GPLv3] ([License text][License]).
License types for the libraries used in *Kokkidio*
are listed in the [LICENSE.README] file.

The name *Kokkidio* is based on the assumptions that 
1. [Kokkos] refers to the Greek *Κόκκος* (engl.: *grain*, though possibly a play on *kernel*), and that 
2. [Eigen] refers to eigenvalues and eigenvectors.

The latter are *ιδιοτιμή* (idiotimí) and *ιδιοδιάνυσμα* (idiodiánysma) in Greek, 
from which the prefix *ιδιο* (idio) was taken
(engl.: *same*, though it could also be from *ίδιος* = own, or self, 
which is the meaning of *eigen* in German). 
*κοκκίδιο* (kokkídio) could be seen as a [portmanteau] of *Kokkos* and *idio*, 
but is in fact the Greek word for *granule*, so not far off *Kokkos* itself.

The logo is a stretched/sheared map of a recolouration of the [Kokkos logo], 
with the eigenvectors of that mapping drawn as arrows.




[logo]: ./media/Kokkidio_Logo.svg "Kokkidio logo"

[Eigen]: https://eigen.tuxfamily.org/
[Kokkos]: https://kokkos.org/
[Kokkos logo]: https://kokkos.org/img/kokkos-logo.png
[GPLv3]: https://www.gnu.org/licenses/gpl-3.0.en.html

[License]: ./LICENSE
[LICENSE.README]: ./LICENSE.README
[wahyd]: https://www.wahyd.tu-berlin.de/

[EigenView]: ./include/Kokkidio/EigenView.hpp
[EigenDualView]: ./include/Kokkidio/EigenDualView.hpp
[portmanteau]: https://en.wikipedia.org/wiki/Portmanteau