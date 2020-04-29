# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

file(GLOB XSMM_CONTRIB_SRC src/runtime/contrib/xsmm/*.cc)

if(IS_DIRECTORY ${USE_XSMM})
  find_library(XSMM_LIBRARY NAMES xsmm HINTS ${USE_XSMM}/lib/)
  if (XSMM_LIBRARY STREQUAL "XSMM_LIBRARY-NOTFOUND")
    message(WARNING "Cannot find XSMM library at ${USE_XSMM}.")
  else()
    include_directories(${USE_XSMM}/include)
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${XSMM_LIBRARY})
    list(APPEND RUNTIME_SRCS ${XSMM_CONTRIB_SRC})
    message(STATUS "Use XSMM library " ${XSMM_LIBRARY})
  endif()
elseif(USE_XSMM STREQUAL "ON")
  find_library(XSMM_LIBRARY libxsmm)
  if (XSMM_LIBRARY STREQUAL "XSMM_LIBRARY-NOTFOUND")
    message(WARNING "Cannot find XSMM library. Try to specify the path to XSMM library.")
  else()
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${XSMM_LIBRARY})
    list(APPEND RUNTIME_SRCS ${XSMM_CONTRIB_SRC})
    message(STATUS "Use XSMM library " ${XSMM_LIBRARY})
  endif()
elseif(USE_XSMM STREQUAL "OFF")
  # pass
else()
  message(FATAL_ERROR "Invalid option: USE_XSMM=" ${USE_XSMM})
endif()
