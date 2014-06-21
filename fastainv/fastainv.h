#ifndef FASTAINV_H
#define FASTAINV_H

#include "formats/formats.h"

typedef void (*progress_f)(void *);

template<class in_real, class out_real>
bool fastainv_sym(csr_matrix<out_real> &Ainv1,
		  csr_matrix<out_real> &Ainv2,
		  const csr_matrix<in_real> &A,
		  progress_f progress, void *v,
		  long long max_elts = 8192000,
		  double droptol = 0.125);

extern template
bool fastainv_sym(csr_matrix<float> &Ainv1,
		  csr_matrix<float> &Ainv2,
		  const csr_matrix<double> &A,
		  progress_f progress, void *v,
		  long long max_elts = 8192000,
		  double droptol = 0.125);

extern template
bool fastainv_sym(csr_matrix<double> &Ainv1,
		  csr_matrix<double> &Ainv2,
		  const csr_matrix<double> &A,
		  progress_f progress, void *v,
		  long long max_elts = 8192000,
		  double droptol = 0.125);

#endif
