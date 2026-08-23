# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import itertools
from pathlib import Path
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_ext_src = Path("src/sam6d/pointnet2/_ext_src")
_ext_include = _ext_src.joinpath("include").resolve()
_ext_sources = map(str, itertools.chain(_ext_src.rglob("*.cu"), _ext_src.rglob("*.cpp")))

setup(
    ext_modules=[
        CUDAExtension(
            name="sam6d.pointnet2._ext",
            sources=sorted(_ext_sources),
            include_dirs = [str(_ext_include)],
            extra_compile_args={
                "cxx": [],
                "nvcc": ["-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]},)
    ],
    cmdclass={"build_ext": BuildExtension}
)
