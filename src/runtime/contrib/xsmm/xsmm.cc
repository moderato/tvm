/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/data_type.h>
#include <dmlc/logging.h>
#include "../cblas/gemm_common.h"
#include <libxsmm.h>
#include "xsmm_utils.h"


namespace tvm {
namespace contrib {

using namespace runtime;

// LIBXSMM only supports column-major
// Adapt the original column-major to row-major: B^TA^T = C^T (row) == AB = C
// m,n,k,A,lda,B,ldb,C,ldc ==> n,m,k,B,ldb,A,lda,C,ldc
inline void SGemm(TVMArgs args, TVMRetValue *ret) {
  typedef float value_type;
  libxsmm_init();

  DLTensor *A = args[0];
  DLTensor *B = args[1];
  DLTensor *C = args[2];
  libxsmm_blasint lda = args[3];
  libxsmm_blasint ldb = args[4];
  libxsmm_blasint ldc = args[5];

  CHECK_EQ(A->ndim, 2);
  CHECK_EQ(B->ndim, 2);
  CHECK_EQ(C->ndim, 2);

  CHECK(TypeEqual(A->dtype, B->dtype));
  // CHECK(TypeEqual(A->dtype, value_type));

  auto A_data = reinterpret_cast<value_type *>(static_cast<char *>(A->data) + A->byte_offset);
  auto B_data = reinterpret_cast<value_type *>(static_cast<char *>(B->data) + B->byte_offset);
  auto C_data = reinterpret_cast<value_type *>(static_cast<char *>(C->data) + C->byte_offset);

  const bool transa = false, transb = false;
  const char transa_char = transa ? 'T' : 'N', transb_char = transb ? 'T' : 'N';
  const libxsmm_blasint m = RowCount(A, transa), n = ColumnCount(B, transb), k = ColumnCount(A, transa);
  const value_type alpha = 1, beta = 1;

  /* C/C++ and Fortran interfaces are available */
  typedef libxsmm_mmfunction<value_type, value_type> kernel_type;
  /* generates and dispatches a matrix multiplication kernel (C++ functor) */
  kernel_type kernel(LIBXSMM_GEMM_FLAGS(transa, transb), n, m, k, ldb, lda, ldc, alpha, beta);
  assert(kernel);

  /* kernel multiplies and accumulates matrix products: C += Ai * Bi */
  libxsmm_gemm(&transa_char, &transb_char,
                n, m, k,
                &alpha,
                B_data, &ldb,
                A_data, &lda,
                &beta,
                C_data, &ldc);

  libxsmm_finalize();
}

TVM_REGISTER_GLOBAL("tvm.contrib.libxsmm.matmul")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  SGemm(args, ret);
});

}  // namespace contrib
}  // namespace tvm
