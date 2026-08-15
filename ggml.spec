# Standalone ggml for OpenMandriva — shared by llama-cpp, whisper-cpp, …
# Optional accelerators ship as separate backend packages so non-AMD systems
# do not pull ROCm, and so Vulkan/OpenCL/BLAS can be installed independently.

# ROCm/HIP backend for AMD GPUs. Host CPU can be anything (Zen, Intel, ARM, …);
# only the offload targets are AMD gfx* ISAs (see %%{rocm_gpu_targets}).
# Build HIP on all arches; opt out with --without rocm only if a builder
# truly lacks the stack.
%bcond_without rocm

%define libname %{mklibname ggml}
%define devname %{mklibname -d ggml}

Summary:		Tensor library for machine learning
Name:			ggml
Version:		0.20.0
Release:		1
License:		MIT
Group:			System/Libraries
%{!?rocm_llvm_maj_ver:%global rocm_llvm_maj_ver 23}
URL:			https://github.com/ggml-org/ggml
Source0:		https://github.com/ggml-org/ggml/archive/refs/tags/v%{version}/ggml-%{version}.tar.gz

# Backend DSO search path (baked into libggml via GGML_BACKEND_DIR)
%global backend_dir %{_libdir}/ggml-backends-%{version}

# Prefer -O3 over distro -Os for throughput-sensitive tensor kernels (all arches)
%global optflags %{optflags} -O3

# BLAS / Vulkan / OpenCL (optional at runtime; needed to *build* the plugins)
BuildRequires:	pkgconfig(openblas)
BuildRequires:	pkgconfig(vulkan)
BuildRequires:	cmake(glslang)
BuildRequires:	cmake(SPIRV-Headers)
BuildRequires:	pkgconfig(shaderc)
BuildRequires:	glslang
BuildRequires:	glslc
BuildRequires:	pkgconfig(OpenCL-Headers)
BuildRequires:	pkgconfig(OpenCL)
%if %{with rocm}
BuildRequires:	rocm-rpm-macros
BuildRequires:	hipcc
BuildRequires:	rocminfo
# clang-offload-bundler / amdgcn-link for HIP fat binaries
BuildRequires:	clang-tools
BuildRequires:	rocm-hip-devel
BuildRequires:	rocm-comgr-devel
BuildRequires:	rocm-runtime-devel
BuildRequires:	rocblas-devel
BuildRequires:	hipblas-devel
BuildRequires:	hipsolver-devel
BuildRequires:	clang >= %{rocm_llvm_maj_ver}
%endif

# ggml-config.cmake lists optional backends (CUDA, DNNL, …) as hard deps — drop them
%global __requires_exclude cmake\\((hip|roc|mkl|intelsycl|cudatoolkit|CUDAToolkit|dnnl|DNNL|openvino|OpenVINO|sycl|SYCL).*

BuildSystem:	cmake
BuildOption:	-DGGML_NATIVE:BOOL=OFF
# Use LTO (also via distro -flto in optflags). Re-disable only if a real
# failure is found, with a comment describing it.
BuildOption:	-DGGML_LTO:BOOL=ON
BuildOption:	-DGGML_BACKEND_DL:BOOL=ON
BuildOption:	-DGGML_BACKEND_DIR=%{backend_dir}
BuildOption:	-DGGML_CPU:BOOL=ON
BuildOption:	-DGGML_CPU_ALL_VARIANTS:BOOL=ON
BuildOption:	-DGGML_VULKAN:BOOL=ON
BuildOption:	-DGGML_OPENCL:BOOL=ON
BuildOption:	-DGGML_BLAS:BOOL=ON
BuildOption:	-DGGML_BLAS_VENDOR=OpenBLAS
# Tests are not installed; needed as offline PGO training binaries (see %pgo).
BuildOption:	-DGGML_BUILD_TESTS:BOOL=ON
BuildOption:	-DGGML_BUILD_EXAMPLES:BOOL=OFF
BuildOption:	-DGGML_CUDA:BOOL=OFF
BuildOption:	-DCMAKE_C_COMPILER=clang
# Host CPU baseline ISA (independent of GPU backend; ALL_VARIANTS still ships
# higher plugins for runtime dispatch). znver1 (Zen 1) has AVX2+FMA+F16C;
# no AVX-512 until Zen 4.
%ifarch znver1
BuildOption:	-DGGML_AVX:BOOL=ON
BuildOption:	-DGGML_AVX2:BOOL=ON
BuildOption:	-DGGML_FMA:BOOL=ON
BuildOption:	-DGGML_F16C:BOOL=ON
BuildOption:	-DGGML_AVX512:BOOL=OFF
%else
BuildOption:	-DGGML_AVX:BOOL=OFF
BuildOption:	-DGGML_AVX2:BOOL=OFF
BuildOption:	-DGGML_FMA:BOOL=OFF
BuildOption:	-DGGML_F16C:BOOL=OFF
BuildOption:	-DGGML_AVX512:BOOL=OFF
%endif
%ifarch %{aarch64}
BuildOption:	-DGGML_CPU_AARCH64:BOOL=ON
%else
BuildOption:	-DGGML_CPU_AARCH64:BOOL=OFF
BuildOption:	-DGGML_OPENCL_USE_ADRENO_KERNELS:BOOL=OFF
%endif
%if %{with rocm}
# Modern HIP: host CXX and HIP language both use clang++ (not the hipcc
# wrapper). CMake rejects CMAKE_HIP_COMPILER=hipcc. Using CXX=hipcc is
# legacy and also breaks aarch64 multi-ISA CPU plugins (SVE host objects
# get -march=…+sve but hipcc still errors "target without sve").
# hipcc remains a BuildRequires for the ROCm driver stack / hipconfig.
BuildOption:	-DCMAKE_CXX_COMPILER=clang++
BuildOption:	-DCMAKE_HIP_COMPILER=clang++
BuildOption:	-DGGML_HIP:BOOL=ON
BuildOption:	-DGGML_HIP_GRAPHS:BOOL=OFF
BuildOption:	-DGGML_HIP_RCCL:BOOL=OFF
# Quote targets: ';' must not split the conf-script for-loop / shell words
BuildOption:	-DGPU_TARGETS="%{rocm_gpu_targets}"
BuildOption:	-DAMDGPU_TARGETS="%{rocm_gpu_targets}"
BuildOption:	-DCMAKE_HIP_ARCHITECTURES="%{rocm_gpu_targets}"
# Ensure device bitcode is found during enable_language(HIP) try-compile
BuildOption:	-DCMAKE_HIP_FLAGS="--rocm-path=%{_prefix} --rocm-device-lib-path=%{_libdir}/amdgcn/bitcode"
BuildOption:	-DCMAKE_PREFIX_PATH=%{_prefix}
%else
BuildOption:	-DCMAKE_CXX_COMPILER=clang++
BuildOption:	-DGGML_HIP:BOOL=OFF
%endif

# 0001: LLVM 23 amdgcn bf16 WMMA/MFMA builtins take short vectors, not __bf16
# 0002: test-quantize-{perf,fns} with GGML_BACKEND_DL via get_proc_address
# 0003: [i/N] progress; apply -o before counting so N is cases that run
# 0004: GGML_MAX_NAME 160 (sd.cpp Flux/SDXL tensor names) + ComfyUI
#      int8_tensorwise/convrot ops. CPU + Vulkan native; HIP uses
#      hipblasGemmEx I8→I32 with a naive kernel fallback (Vega etc.).
#      Must not revert 0002's get_proc_address quantize exports
#      (ABF 649598–600 %pgo: "CPU backend does not export quantize API").
# Keep after all preamble tags: %patchlist is a section-like directive.
%patchlist
0001-llvm23-bf16-wmma-short-vectors.patch
0002-backend-dl-quantize-tests.patch
0003-test-backend-ops-case-counter.patch
0004-max-name-160-and-i8-convrot.patch

%description
ggml is a tensor library for machine learning with integer quantization
and multi-ISA CPU backends. Optional accelerators are separate packages:

* %{name}-backend-blas — OpenBLAS
* %{name}-backend-vulkan — Vulkan
* %{name}-backend-opencl — OpenCL
* %{name}-backend-hip — AMD ROCm/HIP (any host CPU with an AMD GPU)

Used system-wide by llama-cpp, whisper-cpp, stable-diffusion.cpp
and other consumers via find_package(ggml) / *_USE_SYSTEM_GGML.

Tensor names are 160 bytes (GGML_MAX_NAME) so Flux/SDXL graphs fit.
ComfyUI int8_tensorwise/convrot models use extra I8 kernels on CPU,
Vulkan (integerDotProduct) and HIP (hipBLAS I8 GEMM or a portable
fallback). Llama/whisper never emit those ops.

Runtime library: %{libname}. Development files: %{devname}.

%package -n %{libname}
Summary:	Shared libraries for %{name}
Group:		System/Libraries
# File hand-off from llama-cpp which previously shipped these libraries.
# Rebuild llama-cpp with -DLLAMA_USE_SYSTEM_GGML=ON in the same repo push.
Conflicts:	llama-cpp < b10107-11
# Soft deps: useful everywhere, no ROCm. HIP stays Suggests only.
Recommends:	%{name}-backend-blas%{?_isa} = %{EVRD}
Recommends:	%{name}-backend-vulkan%{?_isa} = %{EVRD}
Suggests:	%{name}-backend-opencl%{?_isa} = %{EVRD}
%if %{with rocm}
Suggests:	%{name}-backend-hip%{?_isa} = %{EVRD}
%endif

%description -n %{libname}
Shared libraries for ggml (libggml, libggml-base) and multi-ISA CPU
backend plugins. Optional GPU/BLAS plugins are separate packages.

%package -n %{devname}
Summary:	Development files for %{name}
Group:		Development/C++
Requires:	%{libname}%{?_isa} = %{EVRD}
Provides:	%{name}-devel = %{EVRD}

%description -n %{devname}
Headers, pkg-config and CMake package config for ggml.
Runtime backends are optional plugins; install the matching
%{name}-backend-* packages as needed.

%package backend-blas
Summary:	OpenBLAS backend plugin for %{name}
Group:		System/Libraries
Requires:	%{libname}%{?_isa} = %{EVRD}

%description backend-blas
ggml backend plugin using OpenBLAS for accelerated matrix ops on CPU.

%package backend-vulkan
Summary:	Vulkan backend plugin for %{name}
Group:		System/Libraries
Requires:	%{libname}%{?_isa} = %{EVRD}

%description backend-vulkan
ggml backend plugin using the Vulkan API (cross-vendor GPU support).

%package backend-opencl
Summary:	OpenCL backend plugin for %{name}
Group:		System/Libraries
Requires:	%{libname}%{?_isa} = %{EVRD}

%description backend-opencl
ggml backend plugin using OpenCL (cross-vendor GPU / accelerator support).

%if %{with rocm}
%package backend-hip
Summary:	AMD ROCm/HIP backend plugin for %{name}
Group:		System/Libraries
Requires:	%{libname}%{?_isa} = %{EVRD}

%description backend-hip
ggml backend plugin for AMD GPUs via ROCm/HIP (multi-arch fat binary).
%endif

# Prefer clang; for HIP, strip host-only -m* flags that break device compiles.
# Preserve any pre-set CFLAGS/CXXFLAGS/LDFLAGS (rpm injects -fprofile-* for %pgo
# via the process environment before %conf; do not rebuild from bare %%{optflags}).
%conf -p
export HIP_DEVICE_LIB_PATH=%{_libdir}/amdgcn/bitcode
export ROCM_PATH=%{_prefix}
export HIP_PATH=%{_prefix}
export CC=clang
_cbase=${CFLAGS:-%{optflags}}
_lbase=${LDFLAGS:-%{?__global_ldflags}}
_cflags=$(printf '%s' "$_cbase" | sed -E \
	's/-mfpmath=[^ ]+//g; s/ -m[a-z0-9+.=]+//g; s/-fstack-protector-(strong|all)/-Xarch_host -fstack-protector-\1/g; s/-fcf-protection[^ ]*//g')
_ldflags=$(printf '%s' "$_lbase" | sed -E \
	's/-mfpmath=[^ ]+//g; s/ -m[a-z0-9+.=]+//g; s/-fcf-protection[^ ]*//g')
# PGO instrumentation (from rpm %%pgo):
# 1) Clang defaults to x86_64-pc-linux-gnu but OMV ships libclang_rt.profile.a
#    under *-openmandriva-linux-gnu — force that triple when profiling.
# 2) hipcc must not pass -fprofile-* to the device side (undefined
#    __llvm_profile_*_gpu). Scope PGO to host via -Xarch_host on CXXFLAGS.
if printf '%s' "$_cflags" | grep -q 'fprofile-'; then
	# Clang defaults to *-pc-linux-gnu but OMV ships libclang_rt.profile under
	# the distro triple. Use the rpm target triple so this works for every
	# supported arch (x86_64/znver*, aarch64, riscv64, loongarch64, …) and
	# for alternate libcs (gnu vs musl via %%{_gnu}).
	# Equivalent to %%{_target} on normal OMV builds.
	_pgo_tgt=%{_arch}-%{_target_vendor}-%{_target_os}%{_gnu}
	case " $_cflags " in
	*"--target="*) ;;
	*)
		_cflags="$_cflags --target=$_pgo_tgt"
		_ldflags="$_ldflags --target=$_pgo_tgt"
		;;
	esac
	# Default value-profile site budget is tiny; instrumented test-backend-ops
	# then floods stderr with "Running out of static counters" (tens of GB).
	# More counters per site cuts the spam and keeps more VP data for PGO.
	case " $_cflags " in
	*"-vp-counters-per-site="*) ;;
	*)
		_cflags="$_cflags -mllvm -vp-counters-per-site=16"
		;;
	esac
fi
export CFLAGS="$_cflags"
export LDFLAGS="$_ldflags"
# Host C++ is always clang++ (HIP device code uses CMAKE_HIP_COMPILER=hipcc).
export CXX=clang++
export CXXFLAGS="$_cflags"

# Nested ExternalProject (vulkan-shaders-gen) configures a *new* CMake at
# %build time and inherits process-environment CFLAGS/LDFLAGS — not the
# flags baked into the parent CMake cache. Those env flags still carry
# -fprofile-* from rpm PGO, but not our --target=*-openmandriva-linux-gnu
# fix, so the nested compiler self-test fails looking for
# libclang_rt.profile.a under the default *-pc-linux-gnu triple.
# vulkan-shaders-gen is a one-shot host tool (shader embedder); strip PGO
# from the environment so it configures cleanly. Main libraries already
# have PGO flags in the CMake cache from %conf above.
%build -p
_strip_pgo() {
	printf '%s' "$1" | sed -E 's/(^| )-fprofile-[^ ]+//g; s/  +/ /g; s/^ //; s/ $//'
}
if printf '%s' "${CFLAGS-}${CXXFLAGS-}${LDFLAGS-}" | grep -q 'fprofile-'; then
	export CFLAGS="$(_strip_pgo "${CFLAGS-}")"
	export CXXFLAGS="$(_strip_pgo "${CXXFLAGS-}")"
	export LDFLAGS="$(_strip_pgo "${LDFLAGS-}")"
	export FFLAGS="$(_strip_pgo "${FFLAGS-}")"
	export FCFLAGS="$(_strip_pgo "${FCFLAGS-}")"
fi

# Profile-guided optimization training (pass 1 instrumented binaries).
# Offline synthetic workloads only — no models, no network.
# Host/CPU focused: portable on ABF builders without a GPU; device kernels
# are not improved by host PGO anyway.
#
# Plugins land next to the test binary under _OMV_rpm_build/bin and are
# discovered via the executable directory; libggml*.so live under src/.
# test-quantize-* work with GGML_BACKEND_DL via 0002 patch.
#
# mock rpmbuild_timeout is 40h for the *whole* build (two compiles + training).
# ABF 648966 (aarch64) was SIGKILL'd at 144308s during test-backend-ops perf:
# LIGHTNING_INDEXER(kv=65536,nb=2048) is ~50 min/case at ~3 GFLOPS on Altra,
# plus an equal-cost warmup — ~31h for that op alone. znver1/x86_64 only just
# finished the full suite (~37.6h). There is no --exclude; -o is include-only.
%pgo
# Absolute paths: long training runs can outlive relative cwd assumptions, and
# bash reports missing .so as "No such file or directory" for the binary.
_bd="$PWD/_OMV_rpm_build"
_bin="$_bd/bin"
_src="$_bd/src"
export LD_LIBRARY_PATH="$_src:$_bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ ! -x "$_bin/test-backend-ops" ] || [ ! -x "$_bin/test-quantize-perf" ]; then
	echo "PGO: expected test binaries under $_bin — listing:"
	ls -la "$_bin" 2>/dev/null || true
	exit 1
fi

# 20h budget inside %%pgo leaves room for pass-2 rebuild before the 40h kill.
# date(1) rather than $SECONDS: rpm scriptlets are /bin/sh -e (not always bash).
_pgo_start=$(date +%s)
_pgo_budget=$((20 * 3600))

# Drop compiler-rt VP spam from training logs (still fixable at compile time
# via -vp-counters-per-site in %%conf; this is belt-and-suspenders for ABF logs).
# timeout is best-effort: SIGTERM mid-run may not dump that process's raw
# profiles, so keep LIGHTNING_INDEXER out rather than relying on the cap.
_pgo_run() {
	_left=$((_pgo_start + _pgo_budget - $(date +%s)))
	if [ "$_left" -le 30 ]; then
		echo "PGO: time budget exhausted, skipping: $*"
		return 0
	fi
	echo "PGO: [$((_left / 60)) min left] $*"
	_rc=0
	if command -v timeout >/dev/null 2>&1; then
		timeout --kill-after=30 "$_left" "$@" \
			2> >(grep -v --line-buffered 'LLVM Profile Warning' >&2) || _rc=$?
	else
		"$@" 2> >(grep -v --line-buffered 'LLVM Profile Warning' >&2) || _rc=$?
	fi
	if [ "$_rc" -eq 124 ] || [ "$_rc" -eq 137 ]; then
		echo "PGO: timed out, keeping profiles from earlier steps: $*"
		return 0
	fi
	return "$_rc"
}

# Quantize / dequant / vec_dot hot paths on synthetic rows (best CPU plugin)
_pgo_run "$_bin/test-quantize-perf"

# Op microbenchmarks; -b CPU required (CPU devices are skipped without a filter).
# Same make_test_cases_perf() ops as upstream, minus LIGHTNING_INDEXER.
_pgo_ops='ADD,ADD_ID,ACC,ARGMAX,ARGSORT,COL2IM_1D,CONV_2D,CONV_2D_DW,CONV_3D'
_pgo_ops="$_pgo_ops,CONV_TRANSPOSE_2D,CPY,CUMSUM,FLASH_ATTN_EXT,GATED_DELTA_NET"
_pgo_ops="$_pgo_ops,IM2COL,MEAN,MUL_MAT,MUL_MAT_HADAMARD,MUL_MAT_ID,MUL_MAT_ID_FUSION"
_pgo_ops="$_pgo_ops,PAD_REFLECT_1D,ROPE,ROPE_BACK,SNAKE_FUSE,SOFT_MAX,SOLVE_TRI"
_pgo_ops="$_pgo_ops,SSM_CONV,SSM_CONV_BIAS_SILU,SSM_SCAN,SUM,SUM_ROWS,TOP_K,TRI"
_pgo_run "$_bin/test-backend-ops" perf -b CPU -o "$_pgo_ops"

# Cheap DeepSeek indexer shapes only (kv=256). Full kv=65536 is the timeout.
# Non-fatal: optional extra coverage, not the hot path.
_pgo_run "$_bin/test-backend-ops" perf -b CPU -o LIGHTNING_INDEXER -p 'kv=256' \
	|| echo "PGO: LIGHTNING_INDEXER kv=256 perf failed (continuing)"

# Broader op/branch coverage (correctness suite on CPU). Non-fatal: after a
# successful multi-hour perf run we still want the raw profiles even if the
# binary/libs vanish or the correctness suite fails. Do not pass -j: each
# worker is already N_THREADS=nproc, and 160 workers on Altra can OOM.
if [ -x "$_bin/test-backend-ops" ]; then
	_pgo_run "$_bin/test-backend-ops" test -b CPU \
		|| echo "PGO: test-backend-ops test failed (profiles from perf still kept)"
else
	echo "PGO: test-backend-ops missing after perf; keeping profiles from earlier steps"
	ls -la "$_bin" "$_src" 2>/dev/null || true
fi

# Backend plugins land in the backend dir; drop accidental copies of main libs
%install -a
mkdir -p %{buildroot}%{backend_dir}
rm -f %{buildroot}%{backend_dir}/libggml-base.so* \
	%{buildroot}%{backend_dir}/libggml.so* 2>/dev/null || true

%files -n %{libname}
%license LICENSE
%{_libdir}/libggml.so.*
%{_libdir}/libggml-base.so.*
%dir %{backend_dir}
# CPU ISA variants only — no ROCm/Vulkan/OpenCL/BLAS hard deps
%{backend_dir}/libggml-cpu-*.so

%files -n %{devname}
%doc README.md
%{_includedir}/ggml.h
%{_includedir}/ggml-*.h
%{_includedir}/gguf.h
%{_libdir}/libggml.so
%{_libdir}/libggml-base.so
%{_libdir}/cmake/ggml/
%{_libdir}/pkgconfig/ggml.pc

%files backend-blas
%{backend_dir}/libggml-blas.so

%files backend-vulkan
%{backend_dir}/libggml-vulkan.so

%files backend-opencl
%{backend_dir}/libggml-opencl.so

%if %{with rocm}
%files backend-hip
%{backend_dir}/libggml-hip.so
%endif
