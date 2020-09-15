
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "OCLStream.h"
#define BILLION 1E9

// Cache list of devices
bool cached = false;
std::vector<cl::Device> devices;
void getDeviceList(void);
cl::Event exec_event;
cl_ulong start_time,end_time;size_t return_bytes;
struct timespec start, end;

std::string kernels{R"CLC(

  constant TYPE scalar = startScalar;

  kernel void init(
    global TYPE * restrict a,
    global TYPE * restrict b,
    global TYPE * restrict c,
    TYPE initA, TYPE initB, TYPE initC)
  {
    const size_t i = get_global_id(0);
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }

  kernel void copy(
    global const TYPE * restrict a,
    global TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    c[i] = a[i];
  }

  kernel void mul(
    global TYPE * restrict b,
    global const TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    b[i] = scalar * c[i];
  }

  kernel void add(
    global const TYPE * restrict a,
    global const TYPE * restrict b,
    global TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    c[i] = a[i] + b[i];
  }

  kernel void triad(
    global TYPE * restrict a,
    global const TYPE * restrict b,
    global const TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    a[i] = b[i] + scalar * c[i];
  }

  kernel void stream_dot(
    global const TYPE * restrict a,
    global const TYPE * restrict b,
    global TYPE * restrict sum,
    local TYPE * restrict wg_sum,
    int array_size)
  {
    size_t i = get_global_id(0);
    const size_t local_i = get_local_id(0);
    wg_sum[local_i] = 0.0;
    for (; i < array_size; i += get_global_size(0))
      wg_sum[local_i] += a[i] * b[i];

    for (int offset = get_local_size(0) / 2; offset > 0; offset /= 2)
    {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (local_i < offset)
      {
        wg_sum[local_i] += wg_sum[local_i+offset];
      }
    }

    if (local_i == 0)
      sum[get_group_id(0)] = wg_sum[local_i];
  }

)CLC"};


template <class T>
OCLStream<T>::OCLStream(const unsigned int ARRAY_SIZE, const int device_index):ker_launch_over(6, 0.0), ker_exec_time(6, 0.0), ker_exec_time_rec(6), ker_launch_over_rec(6)
{
  if (!cached)
    getDeviceList();

  // Setup default OpenCL GPU
  if (device_index >= devices.size())
    throw std::runtime_error("Invalid device index");
  device = devices[device_index];
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
  std::cout << "Using OpenCL device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;
  std::cout << "Reduction kernel config: " << dot_num_groups << " groups of size " << dot_wgsize << std::endl;

  context = cl::Context(device);
  queue = cl::CommandQueue(context,CL_QUEUE_PROFILING_ENABLE);

  // Create program
  cl::Program program;
  if (device.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_ACCELERATOR && (platName.find("Intel(R) FPGA")!=std::string::npos || platName.find("Altera")!=std::string::npos)) {
    #define clBinaryProg(name) \
    cl_program name; { \
       printf("Loading "#name".aocx\n"); \
FILE * f = fopen(#name".aocx", "r"); \
       fseek(f, 0, SEEK_END); \
       size_t len = (size_t) ftell(f); \
       const unsigned char * progSrc = (const unsigned char *) malloc(sizeof(char) * len); \
       rewind(f); \
       fread((void *) progSrc, len, 1, f); \
       fclose(f); \
       cl_int err2; \
       name = clCreateProgramWithBinary(context(), 1, &device(), &len, &progSrc, NULL, &err2);}
    clBinaryProg(babelstream);
    program = cl::Program(babelstream);
  } else {
    program = cl::Program(context, kernels);
  }
  std::ostringstream args;
  args << "-DstartScalar=" << startScalar << " ";
  if (sizeof(T) == sizeof(double))
  {
    args << "-DTYPE=double";
    // Check device can do double
    if (!device.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>())
      throw std::runtime_error("Device does not support double precision, please use --float");
    try
    {
      program.build(args.str().c_str());
    }
    catch (cl::Error& err)
    {
      if (err.err() == CL_BUILD_PROGRAM_FAILURE)
      {
        std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>()[0].second << std::endl;
        throw err;
      }
    }
  }
  else if (sizeof(T) == sizeof(float))
  {
    args << "-DTYPE=float";
    program.build(args.str().c_str());
  }

  // Create kernels
  init_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, T, T, T>(program, "init");
  copy_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "copy");
  mul_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "mul");
  add_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(program, "add");
  triad_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(program, "triad");
  dot_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl_int>(program, "stream_dot");

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
  delete init_kernel;
  delete copy_kernel;
  delete mul_kernel;
  delete add_kernel;
  delete triad_kernel;

  devices.clear();
}

template <class T>
void OCLStream<T>::copy()
{
  clock_gettime(CLOCK_REALTIME, &start);
  exec_event=(*copy_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_a, d_c
  );
  exec_event.wait();
  clock_gettime(CLOCK_REALTIME, &end);
  ker_launch_over[0]=( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec )/ BILLION;
  exec_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
  exec_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
  ker_exec_time[0]=static_cast<double>(end_time-start_time)/BILLION;
  exec_event=NULL;
}
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
void OCLStream<T>::mul()
{
  clock_gettime(CLOCK_REALTIME, &start);
  exec_event=(*mul_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_b, d_c
  );
  exec_event.wait();
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
  exec_event=(*add_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_a, d_b, d_c
  );
  exec_event.wait();
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
  exec_event=(*triad_kernel)(
    cl::EnqueueArgs(queue,cl::NDRange(array_size)),
    d_a, d_b, d_c
  );
  exec_event.wait();
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
  clock_gettime(CLOCK_REALTIME, &start);
  exec_event=(*dot_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(dot_num_groups*dot_wgsize), cl::NDRange(dot_wgsize)),
    d_a, d_b, d_sum, cl::Local(sizeof(T) * dot_wgsize), array_size
  );
  exec_event.wait();
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
  exec_event=(*init_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_a, d_b, d_c, initA, initB, initC
  );
  exec_event.wait();
  clock_gettime(CLOCK_REALTIME, &end);
  ker_launch_over[5]=( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec )/ BILLION;
  ker_launch_over_rec[5].push_back(( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec )/ BILLION);
  exec_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
  exec_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
  ker_exec_time[5]=static_cast<double>(end_time-start_time)/BILLION;
  ker_exec_time_rec[5].push_back(ker_exec_time[5]);
  exec_event=NULL;
}

template <class T>
void OCLStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  cl::copy(queue, d_a, a.begin(), a.end());
  cl::copy(queue, d_b, b.begin(), b.end());
  cl::copy(queue, d_c, c.begin(), c.end());
}

void getDeviceList(void)
{
  // Get list of platforms
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  // Enumerate devices
  for (unsigned i = 0; i < platforms.size(); i++)
  {
    std::vector<cl::Device> plat_devices;
    platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &plat_devices);
    devices.insert(devices.end(), plat_devices.begin(), plat_devices.end());
  }
  cached = true;
}

void listDevices(void)
{
  getDeviceList();

  // Print device names
  if (devices.size() == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < devices.size(); i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }


}

std::string getDeviceName(const int device)
{
  if (!cached)
    getDeviceList();

  std::string name;
  cl_device_info info = CL_DEVICE_NAME;

  if (device < devices.size())
  {
    devices[device].getInfo(info, &name);
  }
  else
  {
    throw std::runtime_error("Error asking for name for non-existant device");
  }

  return name;

}

std::string getDeviceDriver(const int device)
{
  if (!cached)
    getDeviceList();

  std::string driver;

  if (device < devices.size())
  {
    devices[device].getInfo(CL_DRIVER_VERSION, &driver);
  }
  else
  {
    throw std::runtime_error("Error asking for driver for non-existant device");
  }

  return driver;
}


template class OCLStream<float>;
template class OCLStream<double>;
