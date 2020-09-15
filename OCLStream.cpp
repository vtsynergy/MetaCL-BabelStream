
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "OCLStream.h"
#include "metamorph.h"
#include "metamorph_opencl.h"
#include "metacl_module.h"

#define BILLION 1E9
a_dim3 globalWorkSize = {1, 1, 1};
a_dim3 localwg = {0, 0, 0};

// Cache list of devices
bool cached = false;
void getDeviceList(void);
cl::Event exec_event;
cl_ulong start_time,end_time;size_t return_bytes;
struct timespec start, end;


template <class T>
void OCLStream<T>::print_res()
{
  it_monitor++;

  for (int i=0;i<=4;i++)
  {
    ker_launch_over_rec[i].push_back(ker_launch_over[i]);
    ker_exec_time_rec[i].push_back(ker_exec_time[i]);
  }
}
template <class T>
OCLStream<T>::OCLStream(const unsigned int ARRAY_SIZE, const int device_index):ker_launch_over(6, 0.0), ker_exec_time(6, 0.0), ker_exec_time_rec(6), ker_launch_over_rec(6)
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
  else if (device.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_ACCELERATOR && (platName.find("Intel(R) FPGA")!=std::string::npos || platName.find("Altera")!=std::string::npos))
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
  printf("Kernel_Init_array_NDRange : %lf\nKernel_Init_array_Event_Based Time: %lf\nKernel_Init_array_Launch_Overhead: %lf\n",ker_launch_over_rec[5][0],ker_exec_time_rec[5][0],ker_launch_over_rec[5][0] - ker_exec_time_rec[5][0]);
  for(int i=0; i<it_monitor;i++)
  {
    printf("****iteration %d *******\n",i+1);
    printf("Kernel_Copy_NDRange : %lf\nKernel_Copy_Event_Based : %lf\nKernel_Copy_Launch_Overhead:  %lf\n",ker_launch_over_rec[0][i],ker_exec_time_rec[0][i],ker_launch_over_rec[0][i] - ker_exec_time_rec[0][i]);
    printf("Kernel_Mul_NDRange : %lf\nKernel_Mul_Event_Based : %lf\nKernel_Mul_Launch_Overhead: %lf\n",ker_launch_over_rec[1][i],ker_exec_time_rec[1][i],ker_launch_over_rec[1][i] - ker_exec_time_rec[1][i]);
    printf("Kernel_Add_NDRange : %lf\nKernel_Add_Event_Based : %lf\nKernel_Add_Launch_Overhead: %lf\n",ker_launch_over_rec[2][i],ker_exec_time_rec[2][i],ker_launch_over_rec[2][i] - ker_exec_time_rec[2][i]);
    printf("Kernel_Triad_NDRange : %lf\nKernel_Triad_Event_Based : %lf\nKernel_Triad_Launch_Overhead: %lf\n",ker_launch_over_rec[3][i],ker_exec_time_rec[3][i],ker_launch_over_rec[3][i] - ker_exec_time_rec[3][i]);
    printf("Kernel_Dot_NDRange : %lf\nKernel_Dot_Event_Based : %lf\nKernel_Dot_Launch_Overhead: %lf\n",ker_launch_over_rec[4][i],ker_exec_time_rec[4][i],ker_launch_over_rec[4][i] - ker_exec_time_rec[4][i]);
    printf("************************\n\n");
  }
  meta_deregister_module(&metacl_metacl_module_registry);
}

template <class T>
void OCLStream<T>::copy()
{
  clock_gettime(CLOCK_REALTIME, &start);
  metacl_babelstream_copy(queue(), &globalWorkSize, &localwg, NULL, 0, &exec_event(), &d_a(), &d_c());
  clock_gettime(CLOCK_REALTIME, &end);
  ker_launch_over[0]=( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec )/ BILLION;
  exec_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
  exec_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
  ker_exec_time[0]=static_cast<double>(end_time-start_time)/BILLION;
  exec_event=NULL;
}

template <class T>
void OCLStream<T>::mul()
{
  clock_gettime(CLOCK_REALTIME, &start);
  metacl_babelstream_mul(queue(), &globalWorkSize, &localwg, NULL, 0, &exec_event(), &d_b(), &d_c());
  clock_gettime(CLOCK_REALTIME, &end);
  ker_launch_over[1]=( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec )/ BILLION;
  exec_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
  exec_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
  ker_exec_time[1]=static_cast<double>(end_time-start_time)/BILLION;
  exec_event=NULL;
}

template <class T>
void OCLStream<T>::add()
{
  clock_gettime(CLOCK_REALTIME, &start);
  metacl_babelstream_add(queue(), &globalWorkSize, &localwg, NULL, 0, &exec_event(), &d_a(), &d_b(), &d_c());
  clock_gettime(CLOCK_REALTIME, &end);
  ker_launch_over[2]=( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec )/ BILLION;
  exec_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
  exec_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
  ker_exec_time[2]=static_cast<double>(end_time-start_time)/BILLION;
  exec_event=NULL;
}

template <class T>
void OCLStream<T>::triad()
{
  clock_gettime(CLOCK_REALTIME, &start);
  metacl_babelstream_triad(queue(), &globalWorkSize, &localwg, NULL, 0, &exec_event(), &d_a(), &d_b(), &d_c());
  clock_gettime(CLOCK_REALTIME, &end);
  ker_launch_over[3]=( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec )/ BILLION;
  exec_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
  exec_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
  ker_exec_time[3]=static_cast<double>(end_time-start_time)/BILLION;
  exec_event=NULL;
}

template <class T>
T OCLStream<T>::dot()
{
  a_dim3 global = {dot_num_groups*dot_wgsize,1,1};
  a_dim3 local  = {dot_wgsize,1,1};

  clock_gettime(CLOCK_REALTIME, &start);
  metacl_babelstream_stream_dot(queue(), &global, &local, NULL, 0, &exec_event(), &d_a(), &d_b(), &d_sum(), (size_t) dot_wgsize, array_size);
  clock_gettime(CLOCK_REALTIME, &end);
  ker_launch_over[4]=( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec )/ BILLION;
  exec_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
  exec_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
  ker_exec_time[4]=static_cast<double>(end_time-start_time)/BILLION;
  exec_event=NULL;

  cl::copy(queue, d_sum, sums.begin(), sums.end());
  T sum = 0.0;
  for (T val : sums)
    sum += val;

  return sum;
}

template <class T>
void OCLStream<T>::init_arrays(T initA, T initB, T initC)
{
  clock_gettime(CLOCK_REALTIME, &start);
  metacl_babelstream_init(queue(), &globalWorkSize , &localwg, NULL, 0, &exec_event(), &d_a(), &d_b(), &d_c(), initA, initB, initC);
  clock_gettime(CLOCK_REALTIME, &end);
  ker_launch_over[5]=( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec )/ BILLION;
  ker_launch_over_rec[5].push_back( ker_launch_over[5]);
  exec_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
  exec_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
  ker_exec_time[5]=static_cast<double>(end_time-start_time)/BILLION;
  exec_event=NULL;
  ker_exec_time_rec[5].push_back(ker_exec_time[5]);
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
