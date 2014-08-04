#ifdef TEMPLATE_DECLARATION
TEMPLATE_DECLARATION
void
copy_indexed(DT *dst, const DT *src, const devmem<int> *i, int n_elts);

TEMPLATE_DECLARATION
void
add_indexed (DT *dst, const DT *src, const devmem<int> *i, int n_elts);

TEMPLATE_DECLARATION
void
dot(const DT *x, const DT *y, DT *s, int n_elts);

TEMPLATE_DECLARATION
void
ppcg_update_scalars(DT *alpha, DT *beta,
                    const DT *gamma, const DT *gammaold, const DT *delta);

TEMPLATE_DECLARATION
void
ppcg_update_vectors(DT *resnorm, DT *gamma, DT *delta,
                    const DT *alpha, const DT *beta,
                    const DT *n, const DT *m,
                    DT *p, DT *s, DT *q, DT *z, DT *x, DT *r, DT *u, DT *w,
                    int n_elts);
#else

#ifndef SBLAS_H
#define SBLAS_H

#define DT devmem<T>

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

#undef DT

#endif

#endif
