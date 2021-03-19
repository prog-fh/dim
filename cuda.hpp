//----------------------------------------------------------------------------

#ifndef DIM_CUDA_HPP
#define DIM_CUDA_HPP

#include <cuda.h>
#include <nvrtc.h>
#include <nvml.h>

#include "enumerate.hpp"

#include <string>
#include <memory>
#include <cstdint>
#include <tuple>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <iostream>

#define DIM_CUDA_THROW(function, error_code)            \
        do                                              \
        {                                               \
          const auto where=std::string{__FILE__}+':'+   \
            std::to_string(__LINE__)+':'+__func__+"()"; \
          throw std::runtime_error{                     \
            where+' '+function+"() failure --- "+       \
            CudaPlatform::error_message((error_code))}; \
        } while(0)

#define DIM_CUDA_CALL(function, args)             \
        do                                        \
        {                                         \
          const auto cu_result=function args;     \
          if(cu_result!=CUDA_SUCCESS)             \
          {                                       \
            DIM_CUDA_THROW(#function, cu_result); \
          }                                       \
        } while(0)

namespace dim {

class CudaDevice;

class CudaPlatform
{
public:

  inline // see the code below
  CudaPlatform();

  CudaPlatform(const CudaPlatform &) =delete;
  CudaPlatform & operator=(const CudaPlatform &) =delete;
  CudaPlatform(CudaPlatform &&rhs) =default;
  CudaPlatform & operator=(CudaPlatform &&rhs) =default;

  ~CudaPlatform()
  {
    nvmlShutdown();
  }

  int
  device_count() const
  {
    return devices_ ? device_count_ : 0;
  }

  const CudaDevice &
  device(int idx) const
  {
    return devices_[idx];
  }

  static
  std::string
  error_message(int error_code)
  {
    const char *error_msg;
    switch(error_code)
    {
#define DIM_CUDA_ERR_CODE_MSG(code) \
          case code:         \
          {                  \
            error_msg=#code; \
            break;           \
          }
      DIM_CUDA_ERR_CODE_MSG(CUDA_SUCCESS)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_INVALID_VALUE)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_OUT_OF_MEMORY)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_NOT_INITIALIZED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_DEINITIALIZED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_PROFILER_DISABLED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_PROFILER_NOT_INITIALIZED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_PROFILER_ALREADY_STARTED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_PROFILER_ALREADY_STOPPED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_NO_DEVICE)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_INVALID_DEVICE)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_INVALID_IMAGE)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_INVALID_CONTEXT)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_CONTEXT_ALREADY_CURRENT)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_MAP_FAILED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_UNMAP_FAILED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_ARRAY_IS_MAPPED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_ALREADY_MAPPED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_NO_BINARY_FOR_GPU)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_ALREADY_ACQUIRED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_NOT_MAPPED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_NOT_MAPPED_AS_ARRAY)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_NOT_MAPPED_AS_POINTER)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_ECC_UNCORRECTABLE)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_UNSUPPORTED_LIMIT)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_CONTEXT_ALREADY_IN_USE)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_INVALID_PTX)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_INVALID_GRAPHICS_CONTEXT)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_NVLINK_UNCORRECTABLE)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_JIT_COMPILER_NOT_FOUND)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_INVALID_SOURCE)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_FILE_NOT_FOUND)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_OPERATING_SYSTEM)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_INVALID_HANDLE)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_NOT_FOUND)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_NOT_READY)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_ILLEGAL_ADDRESS)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_LAUNCH_TIMEOUT)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_CONTEXT_IS_DESTROYED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_ASSERT)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_TOO_MANY_PEERS)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_HARDWARE_STACK_ERROR)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_ILLEGAL_INSTRUCTION)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_MISALIGNED_ADDRESS)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_INVALID_ADDRESS_SPACE)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_INVALID_PC)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_LAUNCH_FAILED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_NOT_PERMITTED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_NOT_SUPPORTED)
      DIM_CUDA_ERR_CODE_MSG(CUDA_ERROR_UNKNOWN)
#undef DIM_CUDA_ERR_CODE_MSG
      default:
      {
        error_msg="unknown Cuda error";
        break;
      }
    }
    return std::string{error_msg}+" ("+std::to_string(error_code)+')';
  }

private:

  static
  std::tuple<void *,      // host pointer
             CUdeviceptr> // device pointer
  alloc_locked_mem_(bool write_only,
                    std::intptr_t size)
  {
    // no need to select a specific context with CU_MEMHOSTALLOC_PORTABLE 
    unsigned int flags=CU_MEMHOSTALLOC_PORTABLE|CU_MEMHOSTALLOC_DEVICEMAP;
    if(write_only)
    {
      flags|=CU_MEMHOSTALLOC_WRITECOMBINED;
    }
    void *host_ptr;
    DIM_CUDA_CALL(cuMemHostAlloc, (&host_ptr, size, flags));
    // FIXME: is this really not specific to any device?
    CUdeviceptr dev_ptr;
    DIM_CUDA_CALL(cuMemHostGetDevicePointer,(&dev_ptr, host_ptr, 0));
    return {host_ptr, dev_ptr};
  }

  static
  void
  free_locked_mem_(void *host_ptr) noexcept
  {
    cuMemFreeHost(host_ptr);
  }

  friend class CudaDevice;
  template<typename T> friend class CudaLockedMem;

  int device_count_{};
  std::unique_ptr<CudaDevice[]> devices_{};
};

//----------------------------------------------------------------------------

class CudaDevice
{
public:

  int
  id() const
  {
    return id_;
  }

  const std::string &
  name() const
  {
    return name_;
  }

  struct Properties
  {
    std::int64_t total_memory;
    int max_threads_per_block;
    int max_block_dim_x;
    int max_block_dim_y;
    int max_block_dim_z;
    int max_grid_dim_x;
    int max_grid_dim_y;
    int max_grid_dim_z;
    int max_shared_memory_per_block;
    int total_constant_memory;
    int warp_size;
    int max_pitch;
    int max_registers_per_block;
    int clock_rate_kHz;
    int texture_alignment;
    int gpu_overlap;
    int multiprocessor_count;
    int kernel_exec_timeout;
    int integrated;
    int can_map_host_memory;
    int compute_mode;
    int maximum_texture1d_width;
    int maximum_texture2d_width;
    int maximum_texture2d_height;
    int maximum_texture3d_width;
    int maximum_texture3d_height;
    int maximum_texture3d_depth;
    int maximum_texture2d_layered_width;
    int maximum_texture2d_layered_height;
    int maximum_texture2d_layered_layers;
    int surface_alignment;
    int concurrent_kernels;
    int ecc_enabled;
    int pci_bus_id;
    int pci_device_id;
    int tcc_driver;
    int memory_clock_rate_kHz;
    int global_memory_bus_width;
    int l2_cache_size;
    int max_threads_per_multiprocessor;
    int async_engine_count;
    int unified_addressing;
    int maximum_texture1d_layered_width;
    int maximum_texture1d_layered_layers;
    int maximum_texture2d_gather_width;
    int maximum_texture2d_gather_height;
    int maximum_texture3d_width_alternate;
    int maximum_texture3d_height_alternate;
    int maximum_texture3d_depth_alternate;
    int pci_domain_id;
    int texture_pitch_alignment;
    int maximum_texturecubemap_width;
    int maximum_texturecubemap_layered_width;
    int maximum_texturecubemap_layered_layers;
    int maximum_surface1d_width;
    int maximum_surface2d_width;
    int maximum_surface2d_height;
    int maximum_surface3d_width;
    int maximum_surface3d_height;
    int maximum_surface3d_depth;
    int maximum_surface1d_layered_width;
    int maximum_surface1d_layered_layers;
    int maximum_surface2d_layered_width;
    int maximum_surface2d_layered_height;
    int maximum_surface2d_layered_layers;
    int maximum_surfacecubemap_width;
    int maximum_surfacecubemap_layered_width;
    int maximum_surfacecubemap_layered_layers;
    int maximum_texture1d_linear_width;
    int maximum_texture2d_linear_width;
    int maximum_texture2d_linear_height;
    int maximum_texture2d_linear_pitch;
    int maximum_texture2d_mipmapped_width;
    int maximum_texture2d_mipmapped_height;
    int compute_capability_major;
    int compute_capability_minor;
    int maximum_texture1d_mipmapped_width;
    int stream_priorities_supported;
    int global_l1_cache_supported;
    int local_l1_cache_supported;
    int max_shared_memory_per_multiprocessor;
    int max_registers_per_multiprocessor;
    int managed_memory;
    int multi_gpu_board;
    int multi_gpu_board_group_id;
    int host_native_atomic_supported;
    int single_to_double_precision_perf_ratio;
    int pageable_memory_access;
    int concurrent_managed_access;
    int compute_preemption_supported;
    int can_use_host_pointer_for_registered_mem;
    int can_use_stream_mem_ops;
    int can_use_64_bit_stream_mem_ops;
    int can_use_stream_wait_value_nor;
    int cooperative_launch;
    int cooperative_multi_device_launch;
    int max_shared_memory_per_block_optin;
    int can_flush_remote_writes;
    int host_register_supported;
    int pageable_memory_access_uses_host_page_tables;
    int direct_managed_mem_access_from_host;
    int cores_per_multiprocessor;
    int core_count;
  };

  const Properties &
  properties() const
  {
    return properties_;
  }

  std::int64_t
  free_memory() const
  {
    make_current_();
    std::size_t free_mem, total_mem;
    DIM_CUDA_CALL(cuMemGetInfo, (&free_mem, &total_mem));
    return std::int64_t(free_mem);
  }

  double // electric power consumed by GPU device, in Watts
  power() const
  {
    unsigned int milliwatts=0;
    if(nvml_dev_)
    {
      nvmlDeviceGetPowerUsage(nvml_dev_, &milliwatts);
    }
    return double(milliwatts)*1e-3;
  }

  ~CudaDevice()
  {
    if(current_==this)
    {
      current_=nullptr;
      cuCtxSetCurrent(nullptr);
    }
    if(context_)
    {
      cuCtxDestroy(context_);
    }
  }

private:

  CudaDevice() =default;
  CudaDevice(const CudaDevice &) =delete;
  CudaDevice & operator=(const CudaDevice &) =delete;

  CudaDevice(CudaDevice &&rhs) noexcept
  : CudaDevice{}
  {
    *this=std::move(rhs);
  }

  CudaDevice & operator=(CudaDevice &&rhs) noexcept
  {
    if(this!=&rhs)
    {
      std::swap(id_, rhs.id_);
      std::swap(context_, rhs.context_);
      std::swap(peer_mask_, rhs.peer_mask_);
      std::swap(name_, rhs.name_);
      std::swap(properties_, rhs.properties_);
      std::swap(nvml_dev_, rhs.nvml_dev_);
    }
    return *this;
  }

  void
  make_current_() const
  {
    if(current_!=this)
    {
      current_=this;
      DIM_CUDA_CALL(cuCtxSetCurrent, (context_));
    }
  }

  void
  make_current_unchecked_() const noexcept
  {
    if(current_!=this)
    {
      current_=this;
      cuCtxSetCurrent(context_);
    }
  }

  CUdeviceptr
  alloc_buffer_(std::intptr_t size) const
  {
    make_current_();
    CUdeviceptr dev_ptr;
    DIM_CUDA_CALL(cuMemAlloc, (&dev_ptr, size));
    return dev_ptr;
  }

  void
  free_buffer_(CUdeviceptr dev_ptr) const noexcept
  {
    make_current_unchecked_();
    cuMemFree(dev_ptr);
  }

  void
  host_to_device_(CUstream stream,
                  CUdeviceptr dev_dst,
                  const void *host_src,
                  std::intptr_t size,
                  std::intptr_t dst_offset,
                  std::intptr_t src_offset) const
  {
    make_current_();
    DIM_CUDA_CALL(cuMemcpyHtoDAsync, (dst_offset+dev_dst,
                                      src_offset+(const char *)host_src,
                                      size, stream));
  }

  void
  device_to_host_(CUstream stream,
                  void *host_dst,
                  CUdeviceptr dev_src,
                  std::intptr_t size,
                  std::intptr_t dst_offset,
                  std::intptr_t src_offset) const
  {
    make_current_();
    DIM_CUDA_CALL(cuMemcpyDtoHAsync, (dst_offset+(char *)host_dst,
                                      src_offset+dev_src,
                                      size, stream));
  }

  void
  device_to_device_(CUstream stream,
                    CUcontext dst_ctx,
                    CUcontext src_ctx,
                    CUdeviceptr dev_dst,
                    CUdeviceptr dev_src,
                    std::intptr_t size,
                    std::intptr_t dst_offset,
                    std::intptr_t src_offset) const
  {
    make_current_();
    const auto dst=dst_offset+dev_dst;
    const auto src=src_offset+dev_src;
    if(dst_ctx==src_ctx)
    {
      DIM_CUDA_CALL(cuMemcpyDtoDAsync, (dst, src, size, stream));
    }
    else
    {
      DIM_CUDA_CALL(cuMemcpyPeerAsync, (dst, dst_ctx, src, src_ctx,
                                        size, stream));
    }
  }

  friend class CudaPlatform;
  friend class CudaStream;
  friend class CudaMarker;
  friend class CudaProgram;
  template<typename T> friend class CudaBuffer;

  inline static const CudaDevice *current_=nullptr;

  int id_{-1};
  CUcontext context_{};
  std::uint64_t peer_mask_{};
  std::string name_{};
  Properties properties_{};
  nvmlDevice_t nvml_dev_{};
};

inline
int // maximal size supported by a 1D block
max_block_size(const CudaDevice &device)
{
  const auto &prop=device.properties();
  return std::min(prop.max_threads_per_block,
                  prop.max_block_dim_x);
}

inline
int // maximal power-of-two size supported by a 1D block
max_power_of_two_block_size(const CudaDevice &device)
{
  const auto max_sz=max_block_size(device);
  auto sz=device.properties().warp_size;
  while((sz<<1)<=max_sz)
  {
    sz<<=1;
  }
  return sz;
}

inline
int // a generaly suitable block size
choose_block_size(const CudaDevice &device,
                  bool power_of_two=true)
{
  (void)power_of_two; // avoid ``unused parameter'' warning
  // A power-of-two block size is not mandatory but seems to be faster
  // FIXME: this hardcoded setting gives good average performances for some
  //        experiments on the actual device used during development.
  //        The performances may vary for other devices and/or algorithms that
  //        would require specific settings.
  return std::max(device.properties().warp_size,
                  max_power_of_two_block_size(device)/4);
}

inline
int // a generaly suitable block count
choose_block_count(const CudaDevice &device)
{
  // FIXME: this hardcoded setting gives good average performances for some
  //        experiments on the actual device used during development.
  //        The performances may vary for other devices and/or algorithms that
  //        would require specific settings.
  return 8*device.properties().multiprocessor_count;
}

inline
std::tuple<int, // a generaly suitable block size
           int> // a generaly suitable block count
choose_layout(const CudaDevice& device,
              bool power_of_two_block_size=false)
{
  return {choose_block_size(device, power_of_two_block_size),
          choose_block_count(device)};
}

inline
std::string
to_string(const CudaDevice &device)
{
  const auto &prop=device.properties();
  auto result=std::string{};
  result+="CUDA device "+std::to_string(device.id())+": "+device.name()+'\n';
#define DIM_CUDA_PROPERTY(pname) \
        result+="  "#pname": "+std::to_string(prop.pname)+'\n';
  result+="  compute_capability: "+
          std::to_string(prop.compute_capability_major)+'.'+
          std::to_string(prop.compute_capability_minor)+'\n';
  DIM_CUDA_PROPERTY(total_memory);
  result+="  free_memory: "+
          std::to_string(device.free_memory())+'\n';
  DIM_CUDA_PROPERTY(max_threads_per_block);
  result+="  max_block_dim: "+
          std::to_string(prop.max_block_dim_x)+' '+
          std::to_string(prop.max_block_dim_y)+' '+
          std::to_string(prop.max_block_dim_z)+'\n';
  result+="  max_grid_dim: "+
          std::to_string(prop.max_grid_dim_x)+' '+
          std::to_string(prop.max_grid_dim_y)+' '+
          std::to_string(prop.max_grid_dim_z)+'\n';
  DIM_CUDA_PROPERTY(max_shared_memory_per_block);
  DIM_CUDA_PROPERTY(total_constant_memory);
  DIM_CUDA_PROPERTY(warp_size);
  DIM_CUDA_PROPERTY(max_registers_per_block);
  DIM_CUDA_PROPERTY(clock_rate_kHz);
  DIM_CUDA_PROPERTY(gpu_overlap);
  DIM_CUDA_PROPERTY(multiprocessor_count);
  DIM_CUDA_PROPERTY(cores_per_multiprocessor);
  DIM_CUDA_PROPERTY(core_count);
  DIM_CUDA_PROPERTY(kernel_exec_timeout);
  DIM_CUDA_PROPERTY(integrated);
  DIM_CUDA_PROPERTY(can_map_host_memory);
  DIM_CUDA_PROPERTY(compute_mode);
  DIM_CUDA_PROPERTY(concurrent_kernels);
  DIM_CUDA_PROPERTY(ecc_enabled);
  DIM_CUDA_PROPERTY(pci_bus_id);
  DIM_CUDA_PROPERTY(pci_device_id);
  DIM_CUDA_PROPERTY(tcc_driver);
  DIM_CUDA_PROPERTY(memory_clock_rate_kHz);
  DIM_CUDA_PROPERTY(global_memory_bus_width);
  DIM_CUDA_PROPERTY(l2_cache_size);
  DIM_CUDA_PROPERTY(max_threads_per_multiprocessor);
  DIM_CUDA_PROPERTY(async_engine_count);
  DIM_CUDA_PROPERTY(unified_addressing);
  DIM_CUDA_PROPERTY(pci_domain_id);
  DIM_CUDA_PROPERTY(stream_priorities_supported);
  DIM_CUDA_PROPERTY(global_l1_cache_supported);
  DIM_CUDA_PROPERTY(local_l1_cache_supported);
  DIM_CUDA_PROPERTY(max_shared_memory_per_multiprocessor);
  DIM_CUDA_PROPERTY(max_registers_per_multiprocessor);
  DIM_CUDA_PROPERTY(managed_memory);
  DIM_CUDA_PROPERTY(multi_gpu_board);
  DIM_CUDA_PROPERTY(multi_gpu_board_group_id);
  DIM_CUDA_PROPERTY(host_native_atomic_supported);
  DIM_CUDA_PROPERTY(single_to_double_precision_perf_ratio);
  DIM_CUDA_PROPERTY(pageable_memory_access);
  DIM_CUDA_PROPERTY(concurrent_managed_access);
  DIM_CUDA_PROPERTY(compute_preemption_supported);
  DIM_CUDA_PROPERTY(can_use_host_pointer_for_registered_mem);
  DIM_CUDA_PROPERTY(can_use_stream_mem_ops);
  DIM_CUDA_PROPERTY(can_use_64_bit_stream_mem_ops);
  DIM_CUDA_PROPERTY(can_use_stream_wait_value_nor);
  DIM_CUDA_PROPERTY(cooperative_launch);
  DIM_CUDA_PROPERTY(cooperative_multi_device_launch);
  DIM_CUDA_PROPERTY(max_shared_memory_per_block_optin);
  DIM_CUDA_PROPERTY(can_flush_remote_writes);
  DIM_CUDA_PROPERTY(host_register_supported);
  DIM_CUDA_PROPERTY(pageable_memory_access_uses_host_page_tables);
  DIM_CUDA_PROPERTY(direct_managed_mem_access_from_host);
  DIM_CUDA_PROPERTY(cores_per_multiprocessor);
  DIM_CUDA_PROPERTY(core_count);
  return result;
}

//----------------------------------------------------------------------------

class CudaStream
{
public:

  CudaStream() =default;

  CudaStream(const CudaDevice &device)
  : device_{&device}
  , stream_{}
  {
    device_->make_current_();
    DIM_CUDA_CALL(cuStreamCreate, (&stream_, CU_STREAM_NON_BLOCKING));
  }

  CudaStream(const CudaStream &) =delete;
  CudaStream & operator=(const CudaStream &) =delete;

  CudaStream(CudaStream &&rhs) noexcept
  : CudaStream{}
  {
    *this=std::move(rhs);
  }

  CudaStream & operator=(CudaStream &&rhs) noexcept
  {
    if(this!=&rhs)
    {
      std::swap(device_, rhs.device_);
      std::swap(stream_, rhs.stream_);
    }
    return *this;
  }

  ~CudaStream()
  {
    if(stream_)
    {
      device_->make_current_unchecked_();
      cuStreamDestroy(stream_);
    }
  }

  const CudaDevice &
  device() const
  {
    return *device_;
  }

  void
  host_sync()
  {
    device_->make_current_();
    DIM_CUDA_CALL(cuStreamSynchronize, (stream_));
  }

private:
  friend class CudaMarker;
  friend class CudaProgram;
  template<typename T> friend class CudaBuffer;

  const CudaDevice *device_{};
  CUstream stream_{};
};

//----------------------------------------------------------------------------

class CudaMarker
{
public:

  CudaMarker() = default;

  CudaMarker(const CudaDevice &device)
  : device_{&device}
  , event_{}
  {
    device_->make_current_();
    DIM_CUDA_CALL(cuEventCreate, (&event_, CU_EVENT_DEFAULT));
  }

  CudaMarker(const CudaMarker &) =delete;
  CudaMarker & operator=(const CudaMarker &) =delete;

  CudaMarker(CudaMarker &&rhs) noexcept
  : CudaMarker{}
  {
    *this=std::move(rhs);
  }

  CudaMarker & operator=(CudaMarker &&rhs) noexcept
  {
    if(this!=&rhs)
    {
      std::swap(device_, rhs.device_);
      std::swap(event_, rhs.event_);
    }
    return *this;
  }

  ~CudaMarker()
  {
    if(event_)
    {
      device_->make_current_unchecked_();
      cuEventDestroy(event_);
    }
  }

  const CudaDevice &
  device() const
  {
    return *device_;
  }

  void
  set(CudaStream &stream)
  {
    device_->make_current_();
    DIM_CUDA_CALL(cuEventRecord, (event_, stream.stream_));
  }

  void
  device_sync(CudaStream &stream)
  {
    device_->make_current_(); // FIXME: really necessary?
    DIM_CUDA_CALL(cuStreamWaitEvent, (stream.stream_, event_, 0));
  }

  void
  host_sync()
  {
    // device_->makeCurrent_(); // not necessary
    DIM_CUDA_CALL(cuEventSynchronize, (event_));
  }

  bool // previous work is done
  test() const
  {
    // device_->makeCurrent_(); // not necessary
    const auto cu_result=cuEventQuery(event_);
    switch(cu_result)
    {
      case CUDA_SUCCESS:
      {
        return true;
      }
      case CUDA_ERROR_NOT_READY:
      {
        return false;
      }
      default:
      {
        DIM_CUDA_THROW("cuEventQuery", cu_result);
        return false;
      }
    }
  }

  std::int64_t // microseconds
  duration(const CudaMarker &previous) const
  {
    // device_->makeCurrent_(); // not necessary
    float milliseconds;
    DIM_CUDA_CALL(cuEventElapsedTime,
                  (&milliseconds, previous.event_, event_));
    return std::int64_t(1.0e3*double(milliseconds));
  }

private:
  const CudaDevice *device_{};
  CUevent event_{};
};

//----------------------------------------------------------------------------

class CudaProgram
{
public:

  CudaProgram() =default;

  CudaProgram(const CudaDevice &device,
              std::string name,
              std::string source_code,
              std::string options={},
              bool prefers_cache_to_shared=true)
  : CudaProgram{device, std::move(name),
                std::move(source_code), std::move(options), {},
                prefers_cache_to_shared}
  {
    // nothing more to be done
  }

  CudaProgram(const CudaDevice &device,
              std::string name,
              std::vector<std::uint8_t> binary_code,
              bool prefers_cache_to_shared=true)
  : CudaProgram{device, std::move(name),
                {}, {}, std::move(binary_code),
                prefers_cache_to_shared}
  {
    // nothing more to be done
  }

  CudaProgram(const CudaProgram &) =delete;
  CudaProgram & operator=(const CudaProgram &) =delete;

  CudaProgram(CudaProgram &&rhs) noexcept
  : CudaProgram{}
  {
    *this=std::move(rhs);
  }

  CudaProgram & operator=(CudaProgram &&rhs) noexcept
  {
    if(this!=&rhs)
    {
      std::swap(device_, rhs.device_);
      std::swap(source_code_, rhs.source_code_);
      std::swap(options_, rhs.options_);
      std::swap(binary_code_, rhs.binary_code_);
      std::swap(prefers_cache_to_shared_, rhs.prefers_cache_to_shared_);
      std::swap(build_log_, rhs.build_log_);
      std::swap(module_, rhs.module_);
      std::swap(kernel_, rhs.kernel_);
      std::swap(properties_, rhs.properties_);
    }
    return *this;
  }

  ~CudaProgram()
  {
    if(module_)
    {
      device_->make_current_unchecked_();
      cuModuleUnload(module_);
    }
  }

  struct Properties
  {
    int max_threads_per_block;
    int shared_size_bytes;
    int const_size_bytes;
    int local_size_bytes;
    int num_regs;
    int ptx_version;
    int binary_version;
    int cache_mode_ca;
    int max_dynamic_shared_size_bytes;
    int preferred_shared_memory_carveout;
  };

  const Properties &
  properties() const
  {
    return properties_;
  }

  const CudaDevice &
  device() const
  {
    return *device_;
  }

  const std::string &
  name() const
  {
    return name_;
  }

  const std::string &
  options() const
  {
    return options_;
  }

  const std::string &
  source_code() const
  {
    return source_code_;
  }

  const std::vector<std::uint8_t> &
  binary_code() const
  {
    return binary_code_;
  }

  bool
  prefers_cache_to_shared() const
  {
    return prefers_cache_to_shared_;
  }

  bool
  build_failure() const
  {
    return !kernel_;
  }

  const std::string &
  build_log() const
  {
    return build_log_;
  }

  void
  launch(CudaStream &stream,
         int x_block_count,
         int y_block_count,
         int z_block_count,
         int x_block_size,
         int y_block_size,
         int z_block_size,
         int shared_memory_size,
         const void * const *args) const
  {
    device_->make_current_();
    DIM_CUDA_CALL(cuLaunchKernel,
                  (kernel_,
                   x_block_count, y_block_count, z_block_count,
                   x_block_size, y_block_size, z_block_size,
                   shared_memory_size,
                   stream.stream_,
                   (void **)args, nullptr));
  }

  void
  launch(CudaStream &stream,
         int x_block_count,
         int y_block_count,
         int x_block_size,
         int y_block_size,
         int shared_memory_size,
         const void * const *args) const
  {
    launch(stream,
           x_block_count, y_block_count, 1,
           x_block_size, y_block_size, 1,
           shared_memory_size, args);
  }

  void
  launch(CudaStream &stream,
         int block_count,
         int block_size,
         int shared_memory_size,
         const void * const *args) const
  {
    launch(stream,
           block_count, 1, 1,
           block_size, 1, 1,
           shared_memory_size, args);
  }

private:

  CudaProgram(const CudaDevice &device,
              std::string name,
              std::string source_code,
              std::string options,
              std::vector<std::uint8_t> binary_code,
              bool prefers_cache_to_shared)
  : device_{&device}
  , name_{std::move(name)}
  , source_code_{std::move(source_code)}
  , options_{std::move(options)}
  , binary_code_{std::move(binary_code)}
  , prefers_cache_to_shared_{prefers_cache_to_shared}
  , build_log_{}
  , module_{}
  , kernel_{}
  , properties_{}
  {
    device_->make_current_();
    CUresult cu_result;
    if(!empty(source_code_))
    {
      auto prog=nvrtcProgram{};
      auto res=nvrtcCreateProgram(&prog, data(source_code_) , data(name_),
                                  0, nullptr, nullptr);
      if(res!=NVRTC_SUCCESS)
      {
        build_log_+="nvrtcCreateProgram("+name_+") failure: "+
                    nvrtcGetErrorString(res)+'\n';
      }
      else
      {
        auto option_words=std::vector<std::string>{std::string{}};
        for(const auto &c: options_)
        {
          if(std::isspace(c))
          {
            if(!empty(option_words.back()))
            {
              option_words.emplace_back();
            }
          }
          else
          {
            option_words.back()+=c;
          }
        }
        if(empty(option_words.back()))
        {
          option_words.pop_back();
        }
        const auto &prop=device_->properties();
        option_words.emplace_back("-std=c++17");
        option_words.emplace_back(
          "-arch=compute_"+
          std::to_string(prop.compute_capability_major)+
          std::to_string(prop.compute_capability_minor));
        option_words.emplace_back("-default-device");
#if defined NDEBUG
        option_words.emplace_back("-use_fast_math");
        option_words.emplace_back("-extra-device-vectorization");
        option_words.emplace_back("-restrict");
#endif
        options_.clear();
        for(const auto &w: option_words)
        {
          if(!empty(options_))
          {
            options_+=' ';
          }
          options_+=w;
        }
        auto raw_options=std::vector<const char *>{};
        raw_options.reserve(size(option_words));
        std::transform(
          cbegin(option_words), cend(option_words),
          back_inserter(raw_options),
          [&](const auto &elem)
          {
            return data(elem);
          });
        const auto comp_res=nvrtcCompileProgram(prog,
                                                int(size(raw_options)),
                                                data(raw_options));
        auto log_size=std::size_t{};
        res=nvrtcGetProgramLogSize(prog, &log_size);
        if(res!=NVRTC_SUCCESS)
        {
          build_log_+="nvrtcGetProgramLogSize("+name_+") failure: "+
                      nvrtcGetErrorString(res)+'\n';
        }
        if(log_size>0)
        {
          auto log=std::string{};
          log.resize(log_size);
          res=nvrtcGetProgramLog(prog, data(log));
          if(res!=NVRTC_SUCCESS)
          {
            build_log_+="nvrtcGetProgramLog("+name_+") failure: "+
                        nvrtcGetErrorString(res)+'\n';
          }
          else
          {
            while(!empty(log)&&
                  ((log.back()=='\0')||std::isspace(log.back())))
            {
              log.pop_back();
            }
            if(!empty(log))
            {
              log+='\n';
              build_log_+=log;
            }
          }
        }
        if(comp_res!=NVRTC_SUCCESS)
        {
          build_log_+="nvrtcCompileProgram("+name_+") failure: "+
                      nvrtcGetErrorString(comp_res)+'\n';
        }
        else
        {
          auto ptx_size=std::size_t{};
          res=nvrtcGetPTXSize(prog, &ptx_size);
          if(res!=NVRTC_SUCCESS)
          {
            build_log_+="nvrtcGetPTXSize("+name_+") failure: "+
                        nvrtcGetErrorString(res)+'\n';
          }
          else
          {
            auto ptx_code=std::vector<std::uint8_t>{};
            ptx_code.resize(ptx_size);
            res=nvrtcGetPTX(prog, (char *)data(ptx_code));
            if(res!=NVRTC_SUCCESS)
            {
              build_log_+="nvrtcGetPTX("+name_+") failure: "+
                          nvrtcGetErrorString(res)+'\n';
            }
            else
            {
              binary_code_=std::move(ptx_code);
            }
          }
        }
        res=nvrtcDestroyProgram(&prog);
        if(res!=NVRTC_SUCCESS)
        {
          build_log_+="nvrtcDestroyProgram("+name_+") failure: "+
                      nvrtcGetErrorString(res)+'\n';
        }
      }
    }
    if(!empty(binary_code_))
    {
      cu_result=cuModuleLoadData(&module_, data(binary_code_));
      if(cu_result!=CUDA_SUCCESS)
      {
        build_log_+="cuModuleLoadData() failure: "+
                    CudaPlatform::error_message(cu_result)+'\n';
      }
    }
    if(module_)
    {
      cu_result=cuModuleGetFunction(&kernel_, module_, data(name_));
      if(cu_result!=CUDA_SUCCESS)
      {
        build_log_+="cuModuleGetFunction() failure: "+
                    CudaPlatform::error_message(cu_result)+'\n';
      }
    }
    if(kernel_)
    {
      DIM_CUDA_CALL(cuFuncSetCacheConfig, (kernel_,
                                          prefers_cache_to_shared_
                                          ? CU_FUNC_CACHE_PREFER_L1
                                          : CU_FUNC_CACHE_PREFER_SHARED));
      Properties &prop=properties_;
#define DIM_CUDA_FUNC_ATTR(value, attrib) \
        DIM_CUDA_CALL(cuFuncGetAttribute, (&prop.value, attrib , kernel_))
      DIM_CUDA_FUNC_ATTR(max_threads_per_block,
                         CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
      DIM_CUDA_FUNC_ATTR(shared_size_bytes,
                         CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
      DIM_CUDA_FUNC_ATTR(const_size_bytes,
                         CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES);
      DIM_CUDA_FUNC_ATTR(local_size_bytes,
                         CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);
      DIM_CUDA_FUNC_ATTR(num_regs,
                         CU_FUNC_ATTRIBUTE_NUM_REGS);
      DIM_CUDA_FUNC_ATTR(ptx_version,
                         CU_FUNC_ATTRIBUTE_PTX_VERSION);
      DIM_CUDA_FUNC_ATTR(binary_version,
                         CU_FUNC_ATTRIBUTE_BINARY_VERSION);
      DIM_CUDA_FUNC_ATTR(cache_mode_ca,
                         CU_FUNC_ATTRIBUTE_CACHE_MODE_CA);
      DIM_CUDA_FUNC_ATTR(max_dynamic_shared_size_bytes,
                         CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
      DIM_CUDA_FUNC_ATTR(preferred_shared_memory_carveout,
                         CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT);
#undef DIM_CUDA_FUNC_ATTR
    }
  }

  const CudaDevice *device_{};
  std::string name_{};
  std::string source_code_{};
  std::string options_{};
  std::vector<std::uint8_t> binary_code_{};
  bool prefers_cache_to_shared_{};
  std::string build_log_{};
  CUmodule module_{};
  CUfunction kernel_{};
  Properties properties_{};
};

inline
std::string
to_string(const CudaProgram &program)
{
  const auto &prop=program.properties();
  auto result=std::string{};
  result+="CUDA program: "+program.name()+'\n';
  result+="  options: "+program.options()+'\n';
  DIM_CUDA_PROPERTY(max_threads_per_block);
  DIM_CUDA_PROPERTY(shared_size_bytes);
  DIM_CUDA_PROPERTY(const_size_bytes);
  DIM_CUDA_PROPERTY(local_size_bytes);
  DIM_CUDA_PROPERTY(num_regs);
  DIM_CUDA_PROPERTY(ptx_version);
  DIM_CUDA_PROPERTY(binary_version);
  DIM_CUDA_PROPERTY(cache_mode_ca);
  DIM_CUDA_PROPERTY(max_dynamic_shared_size_bytes);
  DIM_CUDA_PROPERTY(preferred_shared_memory_carveout);
#undef DIM_CUDA_PROPERTY
  return result;
}

//----------------------------------------------------------------------------

template<typename T>
class CudaBuffer
{
public:

  static_assert(std::is_standard_layout_v<T>&&std::is_trivial_v<T>,
                "plain-old-data type expected");

  CudaBuffer() =default;

  CudaBuffer(const CudaDevice &device,
             std::intptr_t size)
  : device_{&device}
  , dev_ptr_{}
  , size_{size}
  {
    dev_ptr_=device_->alloc_buffer_(size_*sizeof(T));
  }

  CudaBuffer(const CudaBuffer &) =delete;
  CudaBuffer & operator=(const CudaBuffer &) =delete;

  CudaBuffer(CudaBuffer &&rhs) noexcept
  : CudaBuffer{}
  {
    *this=std::move(rhs);
  }

  CudaBuffer & operator=(CudaBuffer &&rhs) noexcept
  {
    if(this!=&rhs)
    {
      std::swap(device_, rhs.device_);
      std::swap(dev_ptr_, rhs.dev_ptr_);
      std::swap(size_, rhs.size_);
    }
    return *this;
  }

  ~CudaBuffer()
  {
    if(dev_ptr_)
    {
      device_->free_buffer_(dev_ptr_);
    }
  }

  const CudaDevice &
  device() const
  {
    return *device_;
  }

  const void * // buffer as program argument
  program_arg() const
  {
    return static_cast<const void *>(&dev_ptr_);
  }

  static
  const void * // null buffer as program argument
  null_program_arg()
  {
    static const auto null_arg=CUdeviceptr{};
    return static_cast<const void *>(&null_arg);
  }

  std::intptr_t
  size() const
  {
    return size_;
  }

  void
  from_host(CudaStream &stream,
            const T *host_src,
            std::intptr_t size=0,
            std::intptr_t dst_offset=0,
            std::intptr_t src_offset=0)
  {
    device_->host_to_device_(stream.stream_,
                             dev_ptr_,
                             host_src,
                             (size ? size : size_-dst_offset)*sizeof(T),
                             dst_offset*sizeof(T),
                             src_offset*sizeof(T));
  }

  void
  to_host(CudaStream &stream,
          T *host_dst,
          std::intptr_t size=0,
          std::intptr_t dst_offset=0,
          std::intptr_t src_offset=0) const
  {
    device_->device_to_host_(stream.stream_,
                             host_dst,
                             dev_ptr_,
                             (size ? size : size_-src_offset)*sizeof(T),
                             dst_offset*sizeof(T),
                             src_offset*sizeof(T));
  }

  bool // copy to dst_buffer will not involve host
  direct_copy_available(const CudaBuffer &dst_buffer) const
  {
    return (dst_buffer.device_==device_)||
           (dst_buffer.device_->peer_mask_&
            ((std::uint64_t(1)<<device_->id_)));
  }

  void
  to_buffer(CudaStream &stream,
            CudaBuffer &dst_buffer,
            std::intptr_t size=0,
            std::intptr_t dst_offset=0,
            std::intptr_t src_offset=0) const
  {
    if(!size)
    {
      size=std::min(dst_buffer.size_-dst_offset, size_-src_offset);
    }
    device_->device_to_device_(stream.stream_,
                               dst_buffer.device_->context_,
                               device_->context_,
                               dst_buffer.dev_ptr_,
                               dev_ptr_,
                               size*sizeof(T),
                               dst_offset*sizeof(T),
                               src_offset*sizeof(T));
  }

private:
  const CudaDevice *device_{};
  CUdeviceptr dev_ptr_{};
  std::intptr_t size_{};
};

//----------------------------------------------------------------------------

template<typename T>
class CudaLockedMem
{
public:

  static_assert(std::is_standard_layout_v<T>&&std::is_trivial_v<T>,
                "plain-old-data type expected");

  CudaLockedMem() =default;

  CudaLockedMem(const CudaPlatform &platform,
                bool write_only,
                std::intptr_t size)
  : host_ptr_{}
  , dev_ptr_{}
  , size_{size}
  , write_only_{write_only}
  {
    const auto [host_ptr, dev_ptr]=
      platform.alloc_locked_mem_(write_only_, size_*sizeof(T));
    host_ptr_=static_cast<T *>(host_ptr);
    dev_ptr_=dev_ptr;
  }

  CudaLockedMem(const CudaLockedMem &) =delete;
  CudaLockedMem & operator=(const CudaLockedMem &) =delete;

  CudaLockedMem(CudaLockedMem &&rhs) noexcept
  : CudaLockedMem{}
  {
    *this=std::move(rhs);
  }

  CudaLockedMem & operator=(CudaLockedMem &&rhs) noexcept
  {
    if(this!=&rhs)
    {
      std::swap(host_ptr_, rhs.host_ptr_);
      std::swap(dev_ptr_, rhs.dev_ptr_);
      std::swap(size_, rhs.size_);
      std::swap(write_only_, rhs.write_only_);
    }
    return *this;
  }

  ~CudaLockedMem()
  {
    if(host_ptr_)
    {
      CudaPlatform::free_locked_mem_(host_ptr_);
    }
  }

  // standard array-like member functions
  auto data()         { return host_ptr_;      }
  auto data()   const { return host_ptr_;      }
  auto cdata()  const { return host_ptr_;      }
  auto size()   const { return size_;          }
  auto empty()  const { return !size();        }
  auto begin()        { return data();         }
  auto begin()  const { return data();         }
  auto cbegin() const { return data();         }
  auto end()          { return data()+size();  }
  auto end()    const { return data()+size();  }
  auto cend()   const { return data()+size();  }

  bool
  write_only() const
  {
    return write_only_;
  }

  const void * // host memory as zero-copy buffer program argument
  program_arg() const
  {
    return static_cast<const void *>(&dev_ptr_);
  }

private:
  T *host_ptr_{};
  CUdeviceptr dev_ptr_{};
  std::intptr_t size_{};
  bool write_only_{};
};

// standard array-like non-member functions

template<typename T>
auto data(        CudaLockedMem<T> &m) { return m.data();   }

template<typename T>
auto data(  const CudaLockedMem<T> &m) { return m.data();   }

template<typename T>
auto cdata( const CudaLockedMem<T> &m) { return m.cdata();  }

template<typename T>
auto size(  const CudaLockedMem<T> &m) { return m.size();   }

template<typename T>
auto empty( const CudaLockedMem<T> &m) { return m.empty();  }

template<typename T>
auto begin(       CudaLockedMem<T> &m) { return m.begin();  }

template<typename T>
auto begin( const CudaLockedMem<T> &m) { return m.begin();  }

template<typename T>
auto cbegin(const CudaLockedMem<T> &m) { return m.cbegin(); }

template<typename T>
auto end(         CudaLockedMem<T> &m) { return m.end();    }

template<typename T>
auto end(   const CudaLockedMem<T> &m) { return m.end();    }

template<typename T>
auto cend(  const CudaLockedMem<T> &m) { return m.cend();   }

//----------------------------------------------------------------------------

CudaPlatform::CudaPlatform()
: device_count_{}
, devices_{}
{
  const auto use_nvml=nvmlInit()==NVML_SUCCESS;
  DIM_CUDA_CALL(cuInit, (0));
  DIM_CUDA_CALL(cuDeviceGetCount, (&device_count_));
  // private ctor/dtor --> std::make_unique() unusable
  devices_=std::unique_ptr<CudaDevice[]>(new CudaDevice[device_count_]);
  for(const auto &i: dim::enumerate(device_count_))
  {
    auto &dev=devices_[i];
    DIM_CUDA_CALL(cuDeviceGet, (&dev.id_, i));
    DIM_CUDA_CALL(cuCtxCreate, (&dev.context_,
                                CU_CTX_SCHED_SPIN|CU_CTX_MAP_HOST,
                                dev.id_));
    dev.make_current_();
    if(dev.id_>=int(8*sizeof(dev.peer_mask_)))
    {
      throw std::runtime_error{
        "insufficient width for CudaDevice.peer_mask_"};
    }
    char name[0x80]="";
    DIM_CUDA_CALL(cuDeviceGetName, (name, sizeof(name), dev.id_));
    dev.name_=name;
    auto &prop=dev.properties_;
    std::size_t total_memory;
    DIM_CUDA_CALL(cuDeviceTotalMem, (&total_memory, dev.id_));
    prop.total_memory=std::int64_t(total_memory);
#define DIM_CUDA_DEV_ATTR(value, attrib) \
        DIM_CUDA_CALL(cuDeviceGetAttribute, (&prop.value, attrib, dev.id_))
    DIM_CUDA_DEV_ATTR(max_threads_per_block,
      CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    DIM_CUDA_DEV_ATTR(max_block_dim_x,
      CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
    DIM_CUDA_DEV_ATTR(max_block_dim_y,
      CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
    DIM_CUDA_DEV_ATTR(max_block_dim_z,
      CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
    DIM_CUDA_DEV_ATTR(max_grid_dim_x,
      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
    DIM_CUDA_DEV_ATTR(max_grid_dim_y,
      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
    DIM_CUDA_DEV_ATTR(max_grid_dim_z,
      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
    DIM_CUDA_DEV_ATTR(max_shared_memory_per_block,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
    DIM_CUDA_DEV_ATTR(total_constant_memory,
      CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY);
    DIM_CUDA_DEV_ATTR(warp_size,
      CU_DEVICE_ATTRIBUTE_WARP_SIZE);
    DIM_CUDA_DEV_ATTR(max_pitch,
      CU_DEVICE_ATTRIBUTE_MAX_PITCH);
    DIM_CUDA_DEV_ATTR(max_registers_per_block,
      CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
    DIM_CUDA_DEV_ATTR(clock_rate_kHz,
      CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
    DIM_CUDA_DEV_ATTR(texture_alignment,
      CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT);
    DIM_CUDA_DEV_ATTR(gpu_overlap,
      CU_DEVICE_ATTRIBUTE_GPU_OVERLAP);
    DIM_CUDA_DEV_ATTR(multiprocessor_count,
      CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    DIM_CUDA_DEV_ATTR(kernel_exec_timeout,
      CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT);
    DIM_CUDA_DEV_ATTR(integrated,
      CU_DEVICE_ATTRIBUTE_INTEGRATED);
    DIM_CUDA_DEV_ATTR(can_map_host_memory,
      CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY);
    DIM_CUDA_DEV_ATTR(compute_mode,
      CU_DEVICE_ATTRIBUTE_COMPUTE_MODE);
    DIM_CUDA_DEV_ATTR(maximum_texture1d_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_texture2d_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_texture2d_height,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT);
    DIM_CUDA_DEV_ATTR(maximum_texture3d_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_texture3d_height,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT);
    DIM_CUDA_DEV_ATTR(maximum_texture3d_depth,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH);
    DIM_CUDA_DEV_ATTR(maximum_texture2d_layered_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_texture2d_layered_height,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT);
    DIM_CUDA_DEV_ATTR(maximum_texture2d_layered_layers,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS);
    DIM_CUDA_DEV_ATTR(surface_alignment,
      CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT);
    DIM_CUDA_DEV_ATTR(concurrent_kernels,
      CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS);
    DIM_CUDA_DEV_ATTR(ecc_enabled,
      CU_DEVICE_ATTRIBUTE_ECC_ENABLED);
    DIM_CUDA_DEV_ATTR(pci_bus_id,
      CU_DEVICE_ATTRIBUTE_PCI_BUS_ID);
    DIM_CUDA_DEV_ATTR(pci_device_id,
      CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID);
    DIM_CUDA_DEV_ATTR(tcc_driver,
      CU_DEVICE_ATTRIBUTE_TCC_DRIVER);
    DIM_CUDA_DEV_ATTR(memory_clock_rate_kHz,
      CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE);
    DIM_CUDA_DEV_ATTR(global_memory_bus_width,
      CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH);
    DIM_CUDA_DEV_ATTR(l2_cache_size,
      CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
    DIM_CUDA_DEV_ATTR(max_threads_per_multiprocessor,
      CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
    DIM_CUDA_DEV_ATTR(async_engine_count,
      CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT);
    DIM_CUDA_DEV_ATTR(unified_addressing,
      CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING);
    DIM_CUDA_DEV_ATTR(maximum_texture1d_layered_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_texture1d_layered_layers,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS);
    DIM_CUDA_DEV_ATTR(maximum_texture2d_gather_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_texture2d_gather_height,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT);
    DIM_CUDA_DEV_ATTR(maximum_texture3d_width_alternate,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE);
    DIM_CUDA_DEV_ATTR(maximum_texture3d_height_alternate,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE);
    DIM_CUDA_DEV_ATTR(maximum_texture3d_depth_alternate,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE);
    DIM_CUDA_DEV_ATTR(pci_domain_id,
      CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID);
    DIM_CUDA_DEV_ATTR(texture_pitch_alignment,
      CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT);
    DIM_CUDA_DEV_ATTR(maximum_texturecubemap_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_texturecubemap_layered_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_texturecubemap_layered_layers,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS);
    DIM_CUDA_DEV_ATTR(maximum_surface1d_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_surface2d_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_surface2d_height,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT);
    DIM_CUDA_DEV_ATTR(maximum_surface3d_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_surface3d_height,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT);
    DIM_CUDA_DEV_ATTR(maximum_surface3d_depth,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH);
    DIM_CUDA_DEV_ATTR(maximum_surface1d_layered_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_surface1d_layered_layers,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS);
    DIM_CUDA_DEV_ATTR(maximum_surface2d_layered_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_surface2d_layered_height,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT);
    DIM_CUDA_DEV_ATTR(maximum_surface2d_layered_layers,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS);
    DIM_CUDA_DEV_ATTR(maximum_surfacecubemap_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_surfacecubemap_layered_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_surfacecubemap_layered_layers,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS);
    DIM_CUDA_DEV_ATTR(maximum_texture1d_linear_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_texture2d_linear_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_texture2d_linear_height,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT);
    DIM_CUDA_DEV_ATTR(maximum_texture2d_linear_pitch,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH);
    DIM_CUDA_DEV_ATTR(maximum_texture2d_mipmapped_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH);
    DIM_CUDA_DEV_ATTR(maximum_texture2d_mipmapped_height,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT);
    DIM_CUDA_DEV_ATTR(compute_capability_major,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    DIM_CUDA_DEV_ATTR(compute_capability_minor,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    DIM_CUDA_DEV_ATTR(maximum_texture1d_mipmapped_width,
      CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH);
    DIM_CUDA_DEV_ATTR(stream_priorities_supported,
      CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED);
    DIM_CUDA_DEV_ATTR(global_l1_cache_supported,
      CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED);
    DIM_CUDA_DEV_ATTR(local_l1_cache_supported,
      CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED);
    DIM_CUDA_DEV_ATTR(max_shared_memory_per_multiprocessor,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
    DIM_CUDA_DEV_ATTR(max_registers_per_multiprocessor,
      CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR);
    DIM_CUDA_DEV_ATTR(managed_memory,
      CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY);
    DIM_CUDA_DEV_ATTR(multi_gpu_board,
      CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD);
    DIM_CUDA_DEV_ATTR(multi_gpu_board_group_id,
      CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID);
    DIM_CUDA_DEV_ATTR(host_native_atomic_supported,
      CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED);
    DIM_CUDA_DEV_ATTR(single_to_double_precision_perf_ratio,
      CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO);
    DIM_CUDA_DEV_ATTR(pageable_memory_access,
      CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS);
    DIM_CUDA_DEV_ATTR(concurrent_managed_access,
      CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS);
    DIM_CUDA_DEV_ATTR(compute_preemption_supported,
      CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED);
    DIM_CUDA_DEV_ATTR(can_use_host_pointer_for_registered_mem,
      CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM);
    DIM_CUDA_DEV_ATTR(can_use_stream_mem_ops,
      CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS);
    DIM_CUDA_DEV_ATTR(can_use_64_bit_stream_mem_ops,
      CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS);
    DIM_CUDA_DEV_ATTR(can_use_stream_wait_value_nor,
      CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR);
    DIM_CUDA_DEV_ATTR(cooperative_launch,
      CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH);
    DIM_CUDA_DEV_ATTR(cooperative_multi_device_launch,
      CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH);
    DIM_CUDA_DEV_ATTR(max_shared_memory_per_block_optin,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
    DIM_CUDA_DEV_ATTR(can_flush_remote_writes,
      CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES);
    DIM_CUDA_DEV_ATTR(host_register_supported,
      CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED);
    DIM_CUDA_DEV_ATTR(pageable_memory_access_uses_host_page_tables,
      CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES);
    DIM_CUDA_DEV_ATTR(direct_managed_mem_access_from_host,
      CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST);
#undef DIM_CUDA_DEV_ATTR
    // https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications
    const int core_counts[][3]={{1, 0,   8}, // Tesla
                                {1, 1,   8},
                                {1, 2,   8},
                                {1, 3,   8},
                                {2, 0,  32}, // Fermi
                                {2, 1,  48},
                                {3, 0, 192}, // Kepler
                                {3, 2, 192},
                                {3, 5, 192},
                                {3, 7, 192},
                                {5, 0, 128}, // Maxwell
                                {5, 2, 128},
                                {5, 3, 128},
                                {6, 0,  64}, // Pascal
                                {6, 1, 128},
                                {6, 2, 128},
                                {7, 0,  64}, // Volta
                                {7, 1,  64},
                                {7, 2,  64},
                                {7, 5,  64}, // Turing
                                {8, 0,  64}, // Ampere
                                {8, 6,  64},
                                {0, 0,   0}};
    auto last_index=0;
    for(auto c=0; core_counts[c][0]; ++c)
    {
      if((core_counts[c][0]>prop.compute_capability_major)||
         ((core_counts[c][0]==prop.compute_capability_major)&&
          (core_counts[c][1]>prop.compute_capability_minor)))
      {
        break;
      }
      last_index=c;
    }
    if((core_counts[last_index][0]!=prop.compute_capability_major)||
       (core_counts[last_index][1]!=prop.compute_capability_minor))
    {
      std::cerr << "warning: unknown compute capability "
                << prop.compute_capability_major << '.'
                << prop.compute_capability_minor
                << " for GPU device " << dev.name_
                << ", assuming "
                << core_counts[last_index][0] << '.'
                << core_counts[last_index][1] << '\n';
    }
    prop.cores_per_multiprocessor=core_counts[last_index][2];
    prop.core_count=prop.multiprocessor_count*
                    prop.cores_per_multiprocessor;
    if(use_nvml)
    {
      char pci_str[0x20]="";
      std::snprintf(pci_str, sizeof(pci_str)-1, "%.8x:%.2x:%.2x.0",
                    prop.pci_domain_id,
                    prop.pci_bus_id,
                    prop.pci_device_id);
      nvmlDeviceGetHandleByPciBusId(pci_str, &dev.nvml_dev_);
    }
  }
  // sort devices to ease peer access
  for(const auto &i: dim::enumerate(device_count_))
  {
    CudaDevice *dev=&devices_[i];
    CudaDevice *best_dev=dev;
    for(const auto &j: dim::enumerate(i+1, device_count_))
    {
      CudaDevice *other=&devices_[j];
#define DIM_CUDA_COMPARE_FIELD(f)                             \
        if(other->properties_.f>best_dev->properties_.f) \
        {                                                \
          best_dev=other;                                \
          continue;                                      \
        }                                                \
        if(other->properties_.f<best_dev->properties_.f) \
        {                                                \
          continue;                                      \
        }
      DIM_CUDA_COMPARE_FIELD(compute_capability_major)
      DIM_CUDA_COMPARE_FIELD(compute_capability_minor)
      DIM_CUDA_COMPARE_FIELD(multiprocessor_count)
      DIM_CUDA_COMPARE_FIELD(clock_rate_kHz)
      DIM_CUDA_COMPARE_FIELD(total_memory)
      DIM_CUDA_COMPARE_FIELD(memory_clock_rate_kHz)
#undef DIM_CUDA_COMPARE_FIELD
    }
    if(best_dev!=dev)
    {
      // private move-operations --> std::swap() unusable
      CudaDevice tmp{std::move(*dev)};
      *dev=std::move(*best_dev);
      *best_dev=std::move(tmp);
    }
  }
  for(const auto &i: dim::enumerate(device_count_))
  {
    CudaDevice &dev=devices_[i];
    for(const auto &j: dim::enumerate(i+1, device_count_))
    {
      CudaDevice &other=devices_[j];
      auto can_access=0;
      DIM_CUDA_CALL(cuDeviceCanAccessPeer, (&can_access, dev.id_, other.id_));
      if(can_access)
      {
        dev.peer_mask_|=std::uint64_t(1)<<other.id_;
        dev.make_current_();
        DIM_CUDA_CALL(cuCtxEnablePeerAccess, (other.context_, 0));
      }
      DIM_CUDA_CALL(cuDeviceCanAccessPeer, (&can_access, other.id_, dev.id_));
      if(can_access)
      {
        other.peer_mask_|=std::uint64_t(1)<<dev.id_;
        other.make_current_();
        DIM_CUDA_CALL(cuCtxEnablePeerAccess, (dev.context_, 0));
      }
    }
  }
}

} // namespace dim

#endif // DIM_CUDA_HPP

//----------------------------------------------------------------------------
