#!/bin/bash
# Debug / Release / RelWithDebInfo
set -e
if [[ -z ${BUILD_TYPE} ]];then
    BUILD_TYPE=Release
fi

TARGET_ARCH=aarch64
TARGET_PLATFORM=linux_native 

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_PLATFORM}_${BUILD_TYPE}

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}

cmake ../.. \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j8
make install