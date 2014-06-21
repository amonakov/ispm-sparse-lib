#ifndef MATRIX_FORMATS_H
#define MATRIX_FORMATS_H

#include "util/cuda/vector.h"

struct matrix_shape
{
  int n_rows, n_cols, n_nz;
};

template<class T>
struct coo_matrix: matrix_shape
{
  vector<int> rows, cols;
  vector<T> elms;
};

template<class T>
struct csr_matrix: matrix_shape
{
  vector<int, host_memory_space_tag> row_ptr, cols;
  vector<T, host_memory_space_tag> elms;
  void pad(int k)
    {
      k--;
      n_rows = (n_rows + k) & ~k;
      int last = row_ptr.back();
      while ((int)row_ptr.size() < n_rows + 1)
	row_ptr.push_back(last);
    }

  unsigned spmv_bytes() const
  {
    return (sizeof(unsigned) * (row_ptr.size() + cols.size())
	    + sizeof(T) * (elms.size() + 2 * n_rows));
  }
};

struct slell_params
{
  unsigned S, H, V, D;
};

template<typename E, typename memory_space>
struct slell_matrix: matrix_shape
{
  vector<unsigned, memory_space> slice_ptr;
  vector<unsigned, memory_space> cols;
  vector<E, memory_space> elms;
  unsigned slice_height;
  unsigned n_slices;
  unsigned hblock;
  bool var_height;
  bool diags;

  slell_matrix(const slell_matrix<E, host_memory_space_tag> &m)
   : matrix_shape(m),
     slice_ptr(m.slice_ptr),
     cols(m.cols),
     elms(m.elms),
     slice_height(m.slice_height),
     n_slices(m.n_slices),
     hblock(m.hblock),
     var_height(m.var_height),
     diags(m.diags)
  {}

  slell_matrix(const matrix_shape &m): matrix_shape(m) {}

  slell_matrix() {}

  unsigned spmv_bytes(bool effective = false)
  {
    return (sizeof(unsigned) * slice_ptr.size()
            + sizeof(unsigned) * (effective ? n_nz : cols.size())
            + sizeof(E)        * (effective ? n_nz : elms.size())
            + sizeof(E) * 2 * n_rows);
  }

  E *slice_elt_ptr(unsigned row, unsigned &slice_hint)
  {
    int slice_ents = slice_ptr.size() / (n_slices + 1);
    int slice;
    if (!var_height)
    {
      slice = row / slice_height;
      if (row % slice_height)
	return NULL;
    }
    else
    {
      for (slice = slice_hint;
	   slice <= n_slices && slice_ptr[slice * slice_ents + 1] != row;
	   slice++)
	{}
      if (slice > n_slices)
	return NULL;
      slice_hint = slice;
    }
    return &elms[slice_ptr[slice * slice_ents]];
  }
};

#endif
