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

/*!
 * \file Use external cudnn utils function
 */

#ifndef TVM_RUNTIME_CONTRIB_LIBXSMM_LIBXSMM_UTILS_H_
#define TVM_RUNTIME_CONTRIB_LIBXSMM_LIBXSMM_UTILS_H_

#include <dmlc/logging.h>
#include <dlpack/dlpack.h>
#include <libxsmm.h>
#include <cstdint>

namespace tvm {
namespace contrib {

// #ifndef CHECK_CUBLAS_ERROR
// #define CHECK_CUBLAS_ERROR(fn)                  \
//   do {                                          \
//     int error = static_cast<int>(fn);                      \
//     CHECK_EQ(error, CUBLAS_STATUS_SUCCESS) << "CUBLAS: " << GetCublasErrorString(error); \
//   } while (0)  // ; intentionally left off.
// #endif  // CHECK_CUBLAS_ERROR


// struct LibxsmmThreadEntry {
//   LibxsmmThreadEntry();
//   ~LibxsmmThreadEntry();
//   libxsmm_dnn_layer* handle{nullptr};
//   libxsmm_dnn_err_t status;
//   static LibxsmmThreadEntry* ThreadLocal(libxsmm_dnn_conv_desc conv_desc);
// };  // LibxsmmThreadEntry

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_LIBXSMM_LIBXSMM_H_
