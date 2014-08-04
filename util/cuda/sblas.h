#ifdef TEMPLATE_DECLARATION
TEMPLATE_DECLARATION void copy_indexed(T *dst, const T *src, const int *i, int n_elts);
TEMPLATE_DECLARATION void add_indexed(T *dst, const T *src, const int *i, int n_elts);
TEMPLATE_DECLARATION void dot(const T *x, const T *y, T *s, int n_elts);

TEMPLATE_DECLARATION void ppcg_update_scalars(T *alpha, T *beta, const T *gamma, const T *gammaold, const T *delta);
TEMPLATE_DECLARATION void ppcg_update_vectors(T *resnorm, T *gamma, T *delta, const T *alpha, const T *beta,
					      const T *n, const T *m, T *p, T *s, T *q, T *z, T *x, T *r, T *u, T *w, int n_elts);
#else

#ifndef SBLAS_H
#define SBLAS_H

namespace sblas
{

#define TEMPLATE_DECLARATION template<typename T>
#include __FILE__
#undef TEMPLATE_DECLARATION

#define TEMPLATE_DECLARATION extern template
#define T float
#include __FILE__
#undef T

#define T double
#include __FILE__
#undef T
#undef TEMPLATE_DECLARATION

}

#endif

#endif
