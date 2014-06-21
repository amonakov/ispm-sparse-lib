#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>

#include "fastainv/fastainv.h"

template<class real>
struct chunky_storage
{
  enum {CHUNK_SZ = 8};
  typedef int  rows_t[CHUNK_SZ];
  typedef real elts_t[CHUNK_SZ];

  int *first, *len, *next;
  rows_t *rows;
  elts_t *elts;
  int alloced, used;;

  chunky_storage(int nheads, int nchunks):
    first(new int[nheads]),
    len(new int[nheads]),
    next(new int[nchunks]),
    rows(new rows_t[nchunks]),
    elts(new elts_t[nchunks]),
    alloced(nchunks),
    used(0)
  {
    memset(len, 0, nheads * sizeof(int));
  }
  ~chunky_storage()
  {
    delete[] elts;
    delete[] rows;
    delete[] next;
    delete[] len;
    delete[] first;
  }
  struct iterator
  {
    int chunkno;
    int chunkoff;

    iterator(int c):
      chunkno(c),
      chunkoff(0)
    {}
  };
  iterator row_begin(int r)
  {
    return iterator(first[r]);
  }
  int row(iterator ci)
  {
    return rows[ci.chunkno][ci.chunkoff];
  }
  real elt(iterator ci)
  {
    return elts[ci.chunkno][ci.chunkoff];
  }
  void inext(iterator& ci)
  {
    if (++ci.chunkoff == CHUNK_SZ)
      {
	ci.chunkoff = 0;
	ci.chunkno = next[ci.chunkno];
      }
  }
  void replace(int r, int n, int *nrow, real *nelt)
  {
    int avail = (alloced - used + (len[r] + CHUNK_SZ - 1) / CHUNK_SZ) * CHUNK_SZ;
    n = std::min(n, avail);
    if (!n)
      return;
    if (!len[r])
      first[r] = used++;

    int chunkno = first[r];
    int i = 0;
    for (;;) {
      memcpy(&rows[chunkno][0], &nrow[i], sizeof(rows[0]));
      memcpy(&elts[chunkno][0], &nelt[i], sizeof(elts[0]));
      i += CHUNK_SZ;
      if (i >= n)
	break;
      if (i >= len[r])
	next[chunkno] = used++;
      chunkno = next[chunkno];
    }
    len[r] = n;
  }
};

template<class real>
struct rowmerge_heap
{
  struct ci_pair
  {
    int column;
    int idx;
    bool operator<(const ci_pair &other) const
    {
      return this->column < other.column;
    }
    void swap(ci_pair &other)
    {
      std::swap(column, other.column);
      std::swap(idx, other.idx);
    }
  };
  ci_pair *ci;
  real *factor;
  int *ofs;
  int *lim;
  int sz;

  rowmerge_heap(int n):
    ci(new ci_pair[n + 1]), factor(new real[n]),
    ofs(new int[n]), lim(new int[n]), sz(0)
  {}
  ~rowmerge_heap()
  {
    delete[] ci;
    delete[] factor;
    delete[] ofs;
    delete[] lim;
  }

  void add(int o, int l, int c, real f)
  {
    ci[sz].column = c;
    ci[sz].idx = sz;
    factor[sz] = f;
    ofs[sz] = o;
    lim[sz] = l;
    sz++;
  }

  int lc(int i)
  {
    return 2*i + 1;
  }
  int rc(int i)
  {
    return 2*i + 2;
  }

  void heapify_down_1(int i, int column, int idx)
  {
    if (lc(i) >= sz)
      {
	ci[i].column = column;
	ci[i].idx = idx;
	return;
      }
    int mcol = std::min(ci[lc(i)].column, ci[rc(i)].column);
    if (mcol < column)
      {
	int mc = (ci[lc(i)].column < ci[rc(i)].column ? lc(i) : rc(i));
	ci[i].column = mcol;
	ci[i].idx = ci[mc].idx;
	heapify_down_1(mc, column, idx);
      }
    else
      {
	ci[i].column = column;
	ci[i].idx = idx;
      }
  }

  void
  heapify_down(int i)
  {
    int column = ci[i].column;
    int idx = ci[i].idx;
    if (lc(i) >= sz)
      return;
    int mcol = std::min(ci[lc(i)].column, ci[rc(i)].column);
    if (mcol < column)
      {
	int mc = (ci[lc(i)].column < ci[rc(i)].column ? lc(i) : rc(i));
	ci[i].column = mcol;
	ci[i].idx = ci[mc].idx;
	heapify_down_1(mc, column, idx);
      }
  }

  void heapify_all()
  {
    ci[sz].column = INT_MAX;
    for (int i = sz / 2 - 1; i >= 0; i--)
      heapify_down(i);
  }

  void delete_min()
  {
    sz--;
    ci[0] = ci[sz];
    ci[sz].column = INT_MAX;
    heapify_down(0);
  }

  void increase_min(int nc)
  {
    ci[0].column = nc;
    heapify_down(0);
  }
};

template<class real, class in_real>
int __attribute((noinline))
sparse_sparse_mv(chunky_storage<real> &Wdata, int j, const csr_matrix<in_real> &A,
		 int *AWj_idx, in_real *AWj_elt)
{
  rowmerge_heap<real> eltheap(Wdata.len[j] + 1);

  // Populate the heap with first elements of each non-empty row
  typename chunky_storage<real>::iterator ci = Wdata.row_begin(j);
  for (int i = 0; i < Wdata.len[j]; i++)
    {
      int row = Wdata.row(ci);
      int idx = A.row_ptr[row];
      int lim = A.row_ptr[row + 1];
      if (idx < lim)
	eltheap.add(idx, lim, A.cols[idx], Wdata.elt(ci));
      Wdata.inext(ci);
    }
  // Account for the implicit diagonal element in Wj
  if (A.row_ptr[j] < A.row_ptr[j+1])
    eltheap.add(A.row_ptr[j], A.row_ptr[j+1], A.cols[A.row_ptr[j]], 1);
  eltheap.heapify_all();

  int* AWj_idx_orig = AWj_idx;
  AWj_elt--;
  int prevcol = -1;
  in_real prevelt = 0;
  do {
    int col = eltheap.ci[0].column;
    int idx = eltheap.ci[0].idx;
    in_real add = A.elms[eltheap.ofs[idx]] * eltheap.factor[idx];
    __builtin_prefetch(&A.elms[8 + eltheap.ofs[idx]], 0, 0);
    int next = -((prevcol - col) >> 31);
    prevelt = prevelt - prevelt * next;
    prevcol = col;
    AWj_elt += next;
    *AWj_idx = col;
    *AWj_elt = prevelt += add;
    AWj_idx += next;
    if (++eltheap.ofs[idx] < eltheap.lim[idx])
      {
	__builtin_prefetch(&A.cols[8 + eltheap.ofs[idx]], 0, 0);
	eltheap.increase_min(A.cols[eltheap.ofs[idx]]);
      }
    else
      eltheap.delete_min();
  } while (eltheap.sz > 0);

  return AWj_idx - AWj_idx_orig;
}

template<class real>
void __attribute((noinline))
update_Wji(int j, int i, chunky_storage<real> &Wdata, real Wjfactor, double droptol,
	   int *merge_row, real *merge_elt)
{
  int Wj_len = Wdata.len[j];
  int Wi_len = Wdata.len[i];
  typename chunky_storage<real>::iterator Wjci = Wdata.row_begin(j);
  typename chunky_storage<real>::iterator Wici = Wdata.row_begin(i);
  int wii = 0, wji = 0, merge_len = 0;
  int  row;
  real elt;

  while (wii < Wi_len && wji < Wj_len)
    {
      if (Wdata.row(Wjci) < Wdata.row(Wici))
	{
	  row = Wdata.row(Wjci);
	  elt = Wdata.elt(Wjci) * Wjfactor;
	  Wdata.inext(Wjci);
	  wji++;
	}
      else if (Wdata.row(Wjci) > Wdata.row(Wici))
	{
	  row = Wdata.row(Wici);
	  elt = Wdata.elt(Wici);
	  Wdata.inext(Wici);
	  wii++;
	}
      else
	{
	  row = Wdata.row(Wici);
	  elt = Wdata.elt(Wici) + Wdata.elt(Wjci) * Wjfactor;
	  Wdata.inext(Wici);
	  Wdata.inext(Wjci);
	  wii++;
	  wji++;
	}
      if (std::abs(elt) > droptol)
	{
	  merge_row[merge_len] = row;
	  merge_elt[merge_len] = elt;
	  merge_len++;
	}
    }

  // Implicit 1 in Wj
  while (wii < Wi_len && wji == Wj_len)
    {
      if (j < Wdata.row(Wici))
	{
	  row = j;
	  elt = Wjfactor;
	  wji++;
	}
      else if (j > Wdata.row(Wici))
	{
	  row = Wdata.row(Wici);
	  elt = Wdata.elt(Wici);
	  Wdata.inext(Wici);
	  wii++;
	}
      else
	{
	  row = Wdata.row(Wici);
	  elt = Wdata.elt(Wici) + Wjfactor;
	  Wdata.inext(Wici);
	  wii++;
	  wji++;
	}
      if (std::abs(elt) > droptol)
	{
	  merge_row[merge_len] = row;
	  merge_elt[merge_len] = elt;
	  merge_len++;
	}
    }

  while (wii < Wi_len)
    {
      row = Wdata.row(Wici);
      elt = Wdata.elt(Wici);
      Wdata.inext(Wici);
      wii++;
      if (std::abs(elt) > droptol)
	{
	  merge_row[merge_len] = row;
	  merge_elt[merge_len] = elt;
	  merge_len++;
	}
    }

  while (wji < Wj_len)
    {
      row = Wdata.row(Wjci);
      elt = Wdata.elt(Wjci) * Wjfactor;
      Wdata.inext(Wjci);
      wji++;
      if (std::abs(elt) > droptol)
	{
	  merge_row[merge_len] = row;
	  merge_elt[merge_len] = elt;
	  merge_len++;
	}
    }

  if (wji == Wj_len)
    {
      row = j;
      elt = Wjfactor;
      wji++;
      if (std::abs(elt) > droptol)
	{
	  merge_row[merge_len] = row;
	  merge_elt[merge_len] = elt;
	  merge_len++;
	}
    }

  Wdata.replace(i, merge_len, merge_row, merge_elt);
}

static long long
maxneed(long long n, int chunk_sz)
{
  n = (n + chunk_sz - 1) / chunk_sz;
  return n * (n + 1) / 2 * chunk_sz * chunk_sz;
}

template<class in_real, class out_real>
bool
fastainv_sym(csr_matrix<out_real> &Ainv1, csr_matrix<out_real> &Ainv2,
	     const csr_matrix<in_real> &A,
	     progress_f progress, void *vprogress,
	     long long max_elts, double droptol)
{
  typedef out_real real;
  int n = A.n_rows;

  if (n <= 0)
    return false;
  bool retval = false;

  max_elts = std::min(max_elts, maxneed(n, chunky_storage<real>::CHUNK_SZ));
  chunky_storage<real> Wdata(n, max_elts / chunky_storage<real>::CHUNK_SZ);

  real *D = new real[n];

  int  *AWj_idx_s = new  int[n];
  in_real *AWj_elt_s = new in_real[n];
  int  *merge_idx = new  int[n];
  real *merge_elt = new real[n];

  for (int j = 0; j < n; j++)
    {
      progress(vprogress);
      int Wj_len = Wdata.len[j];
      // u = A Wj
      const int  *AWj_idx;
      const in_real *AWj_elt;
      int AWn;
      // Shortcut for the common case when Wj has only one element
      real Djj = 0;
      int ui = 0;
      if (!Wj_len)
	{
	  AWj_idx = &A.cols[A.row_ptr[j]];
	  AWj_elt = &A.elms[A.row_ptr[j]];
	  AWn = A.row_ptr[j + 1] - A.row_ptr[j];
	}
      else
	{
	  AWj_idx = AWj_idx_s;
	  AWj_elt = AWj_elt_s;
	  AWn = sparse_sparse_mv(Wdata, j, A, AWj_idx_s, AWj_elt_s);
	  // Djj = u Wj
	  int Wji = 0;
	  typename chunky_storage<real>::iterator ci = Wdata.row_begin(j);
	  while (ui < AWn && Wji < Wj_len)
	    if (AWj_idx[ui] < Wdata.row(ci))
	      ui++;
	    else if (AWj_idx[ui] > Wdata.row(ci))
	      {
		Wji++;
		Wdata.inext(ci);
	      }
	    else
	      {
		Djj += AWj_elt[ui++] * Wdata.elt(ci);
		Wji++;
		Wdata.inext(ci);
	      }
	}

      while (ui < AWn)
	if (AWj_idx[ui] < j)
	  ui++;
	else if (AWj_idx[ui] > j)
	  break;
	else
	  Djj += AWj_elt[ui++];

      if (Djj == 0)   // Breakdown
	{
	  D[j] = 0;
	  continue;
	}

      Djj  = 1. / Djj;

      for (; ui < AWn; ui++)
	{
	  // Update W_i
	  int i = AWj_idx[ui];
	  real Wjfactor = - AWj_elt[ui] * Djj;

	  update_Wji(j, i, Wdata, Wjfactor, droptol,
		     merge_idx, merge_elt);
	}

      D[j] = Djj;
    }

  // Transpose W into Ainv2

  Ainv2.n_rows = Ainv2.n_cols = n;
  Ainv2.row_ptr.clear();
  Ainv2.row_ptr.resize(n+1);
  for (int i = 0; i < n; i++)
    {
      typename chunky_storage<real>::iterator ci = Wdata.row_begin(n-1-i);
      for (int j = 0; j < Wdata.len[n-1-i]; j++)
	{
	  int row = Wdata.row(ci);
	  Ainv2.row_ptr[row + 1]++;
	  Wdata.inext(ci);
	}
    }
  {
  int sum = 0;
  for (int i = 1; i <= n; i++)
    {
      sum += Ainv2.row_ptr[i] + 1;
      Ainv2.row_ptr[i] = sum;
    }
  Ainv2.n_nz = sum;
  Ainv2.cols.resize(sum);
  Ainv2.elms.resize(sum);
  std::vector<int> row_cur(n);
  for (int i = 0; i < n; i++)
    {
      typename chunky_storage<real>::iterator ci = Wdata.row_begin(n-1-i);
      real Dscal = std::sqrt(std::abs(D[n-1-i]));
      D[n-1-i] = copysign(Dscal, D[n-1-i]);
      for (int j = 0; j < Wdata.len[n-1-i]; j++)
	{
	  int row = Wdata.row(ci);
	  int off = Ainv2.row_ptr[row] + row_cur[row]++;
	  Ainv2.cols[off] = i;
	  Ainv2.elms[off] = Wdata.elt(ci) * Dscal;
	  Wdata.inext(ci);
	}
      int off = Ainv2.row_ptr[n-1-i] + row_cur[n-1-i]++;
      Ainv2.cols[off] = i;
      Ainv2.elms[off] = Dscal;
    }
  }

  // Copy W into Ainv1 and scale by D^-1

  Ainv1.n_rows = Ainv1.n_cols = n;
  Ainv1.row_ptr.clear();
  Ainv1.row_ptr.reserve(n+1);
  Ainv1.row_ptr.push_back(0);
  Ainv1.cols.clear();
  Ainv1.cols.reserve(Ainv2.n_nz);
  Ainv1.elms.clear();
  Ainv1.elms.reserve(Ainv2.n_nz);
  for (int i = 0; i < n; i++)
    {
      typename chunky_storage<real>::iterator ci = Wdata.row_begin(n-1-i);
      for (int j = 0; j < Wdata.len[n-1-i]; j++)
	{
	  int row = Wdata.row(ci);
	  Ainv1.cols.push_back(row);
	  Ainv1.elms.push_back(Wdata.elt(ci) * D[n-1-i]);
	  Wdata.inext(ci);
	}
      Ainv1.cols.push_back(n-1-i);
      Ainv1.elms.push_back(D[n-1-i]);
      Ainv1.row_ptr.push_back(Ainv1.cols.size());
    }
  Ainv1.n_nz = Ainv1.cols.size();
  retval = true;
  delete[] D;
  delete[] AWj_idx_s;
  delete[] AWj_elt_s;
  delete[] merge_idx;
  delete[] merge_elt;
  return retval;
}

template
bool fastainv_sym(csr_matrix<float> &Ainv1,
		  csr_matrix<float> &Ainv2,
		  const csr_matrix<double> &A,
		  progress_f progress, void *v,
		  long long max_elts = 8192000,
		  double droptol = 0.125);

template
bool fastainv_sym(csr_matrix<double> &Ainv1,
		  csr_matrix<double> &Ainv2,
		  const csr_matrix<double> &A,
		  progress_f progress, void *v,
		  long long max_elts = 8192000,
		  double droptol = 0.125);
