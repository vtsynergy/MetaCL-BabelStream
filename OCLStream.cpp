
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "OCLStream.h"
#include "metamorph.h"
#include "metamorph_opencl.h"
#include "metacl_module.h"

a_dim3 globalWorkSize = {1, 1, 1};
a_dim3 localwg = {0, 0, 0};

// Cache list of devices
bool cached = false;
void getDeviceList(void);


template <class T>
OCLStream<T>::OCLStream(const unsigned int ARRAY_SIZE, const int device_index)
{
  cl::Platform platformlist;
  cl_int errNum;

  globalWorkSize[0] = ARRAY_SIZE;
  meta_set_acc(device_index, metaModePreferOpenCL); //Must be set to OpenCL, pass in device_index directly
  meta_get_state_OpenCL(&platformlist(), &device(), &context(), &queue());

  errNum=clRetainCommandQueue(queue());
  errNum=clRetainContext(context());
  errNum=clRetainDevice(device());
  std::string platName = cl::Platform(device.getInfo<CL_DEVICE_PLATFORM>()).getInfo<CL_PLATFORM_NAME>();

  // Determine sensible dot kernel NDRange configuration
  if (device.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_CPU)
  {
    dot_num_groups = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    dot_wgsize     = device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>() * 2;
  }
  else if (device.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_ACCELERATOR && (platName.find("Intel (R) FPGA")!=std::string::npos || platName.find("Altera")!=std::string::npos))
  {
    dot_num_groups = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * 4;
    dot_wgsize=64;
  }
  else
  {
    dot_num_groups = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * 4;
    dot_wgsize     = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  }

  // Print out device information
  std::string name, driver;
  device.getInfo(CL_DEVICE_NAME, &name);
  device.getInfo(CL_DRIVER_VERSION, &driver);
  std::cout << "Using OpenCL device " << name << std::endl;
  std::cout << "Driver: " << driver << std::endl;
  std::cout << "Reduction kernel config: " << dot_num_groups << " groups of size " << dot_wgsize << std::endl;

  std::ostringstream args;
  args << "-DstartScalar=" << startScalar << " ";
  if (sizeof(T) == sizeof(double))
  {
    args << "-DTYPE=double";
    // Check device can do double
    if (!device.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>())
      throw std::runtime_error("Device does not support double precision, please use --float");
  }
  else if (sizeof(T) == sizeof(float))
  {
    args << "-DTYPE=float";
  }
  __metacl_babelstream_custom_args= args.str().c_str();
  meta_register_module(&metacl_metacl_module_registry);

  array_size = ARRAY_SIZE;

  // Check buffers fit on the device
  cl_ulong totalmem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
  cl_ulong maxbuffer = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  if (maxbuffer < sizeof(T)*ARRAY_SIZE)
    throw std::runtime_error("Device cannot allocate a buffer big enough");
  if (totalmem < 3*sizeof(T)*ARRAY_SIZE)
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create buffers
  d_a = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * ARRAY_SIZE);
  d_b = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * ARRAY_SIZE);
  d_c = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * ARRAY_SIZE);
  d_sum = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(T) * dot_num_groups);

  sums = std::vector<T>(dot_num_groups);
}

template <class T>
OCLStream<T>::~OCLStream()
{
  meta_deregister_module(&metacl_metacl_module_registry);
}

template <class T>
void OCLStream<T>::copy()
{
  metacl_babelstream_copy(queue(), &globalWorkSize, &localwg, NULL, 0, NULL, &d_a(), &d_c());
}

template <class T>
void OCLStream<T>::mul()
{
  metacl_babelstream_mul(queue(), &globalWorkSize, &localwg, NULL, 0, NULL, &d_b(), &d_c());
}

template <class T>
void OCLStream<T>::add()
{
  metacl_babelstream_add(queue(), &globalWorkSize, &localwg, NULL, 0, NULL, &d_a(), &d_b(), &d_c());
}

template <class T>
void OCLStream<T>::triad()
{
  metacl_babelstream_triad(queue(), &globalWorkSize, &localwg, NULL, 0, NULL, &d_a(), &d_b(), &d_c());
}

template <class T>
T OCLStream<T>::dot()
{
  a_dim3 global = {dot_num_groups*dot_wgsize,1,1};
  a_dim3 local  = {dot_wgsize,1,1};

  metacl_babelstream_stream_dot(queue(), &global, &local, NULL, 0, NULL, &d_a(), &d_b(), &d_sum(), (size_t) dot_wgsize, array_size);
  cl::copy(queue, d_sum, sums.begin(), sums.end());
  T sum = 0.0;
  for (T val : sums)
    sum += val;

  return sum;
}

template <class T>
void OCLStream<T>::init_arrays(T initA, T initB, T initC)
{
  metacl_babelstream_init(queue(), &globalWorkSize , &localwg, NULL, 0, NULL, &d_a(), &d_b(), &d_c(), initA, initB, initC);
}

template <class T>
void OCLStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  cl::copy(queue, d_a, a.begin(), a.end());
  cl::copy(queue, d_b, b.begin(), b.end());
  cl::copy(queue, d_c, c.begin(), c.end());
}

template class OCLStream<float>;
template class OCLStream<double>;
