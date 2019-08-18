
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "OCLStream.h"
#include "metamorph.h"
#include "metacl_module.h"
#include <time.h>
#define BILLION 1E9
a_dim3 globalWorkSize = {33554432,1,1};
a_dim3 localwg = {0,0,0};
cl_event exec_event;
cl_ulong start_time,end_time;size_t return_bytes;
struct timespec start, end;


// Cache list of devices
bool cached = false;
//std::vector<cl::Device> devices;
void getDeviceList(void);

template <class T>
OCLStream<T>::OCLStream(const unsigned int ARRAY_SIZE, const int device_index)
{
 
 for (int i=0;i<6;i++){
	ker_launch_over.push_back(0);
	ker_exec_time.push_back(0);
  }
  
  cl::Platform platformlist; 
  cl_int errNum;
  //cl_device_id device1;

  meta_set_acc(device_index, metaModePreferOpenCL); //Must be set to OpenCL, don't need a device since we will override
  meta_get_state_OpenCL(&platformlist(), &device(), &context(), &queue());
  
  /*
  if (!cached)
    getDeviceList();
  
  // Setup default OpenCL GPU
  if (device_index >= devices.size())
    throw std::runtime_error("Invalid device index");
  device = devices[device_index];
  */
  errNum=clRetainCommandQueue(queue());
  errNum=clRetainContext(context());
  errNum=clRetainDevice(device());
  
  // Determine sensible dot kernel NDRange configuration
  //cl::Device device(device1,true);
  
  if (device.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_CPU)
  {
    dot_num_groups = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    dot_wgsize     = device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>() * 2;
  }
  else
  {
    dot_num_groups = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * 4;
    dot_wgsize     = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  }
  
  std::string driver;
  device.getInfo(CL_DRIVER_VERSION, &driver);
  
  // Print out device information
  //std::cout << "Using OpenCL device " << getDeviceName(device_index) << std::endl;
  
  std::cout << "Driver: " << driver << std::endl;
  std::cout << "Reduction kernel config: " << dot_num_groups << " groups of size " << dot_wgsize << std::endl;
  
  //context = cl::Context(device);
  //queue = cl::CommandQueue(context);
  
  // Create program
  //cl::Program program(context, kernels);
  
  
  std::ostringstream args;

  args << "-DstartScalar=" << startScalar << " ";
  if (sizeof(T) == sizeof(double))
  {
    args << "-DTYPE=double";
    // Check device can do double
    
    if (!device.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>())
      printf("Device does not support double precision, please use --float");
   /*
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
   */
  }
  else if (sizeof(T) == sizeof(float))
  {
    args << "-DTYPE=float"; 
    //program.build(args.str().c_str());
  }


   std::string c_args = args.str();
   __meta_gen_opencl_babelstream_custom_args= c_args.c_str();
   meta_register_module(&meta_gen_opencl_metacl_module_registry);
  // Create kernels
  
  
  //init_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, T, T, T>(program, "init");
  //copy_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "copy");
  //mul_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "mul");
  //add_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(program, "add");
  //triad_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(program, "triad");
  //dot_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl_int>(program, "stream_dot");
  
  array_size = ARRAY_SIZE;
  //dot_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl_int>(program, "stream_dot");
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
  //devices.clear();
     meta_deregister_module(&meta_gen_opencl_metacl_module_registry);

     printf("Kernel: Copy, NDRange time: %lf, Event Based Time: %lf, Launch Overhead: %lf\n",ker_launch_over[0],ker_exec_time[0],ker_launch_over[0]-ker_exec_time[0]);
     printf("Kernel: Mul, NDRange time: %lf, Event Based Time: %lf, Launch Overhead: %lf\n",ker_launch_over[1],ker_exec_time[1],ker_launch_over[1]-ker_exec_time[1]);
     printf("Kernel: Add, NDRange time: %lf, Event Based Time: %lf, Launch Overhead: %lf\n",ker_launch_over[2],ker_exec_time[2],ker_launch_over[2]-ker_exec_time[2]);
     printf("Kernel: Triad, NDRange time: %lf, Event Based Time: %lf, Launch Overhead: %lf\n",ker_launch_over[3],ker_exec_time[3],ker_launch_over[3]-ker_exec_time[3]);
     printf("Kernel: Dot, NDRange time: %lf, Event Based Time: %lf, Launch Overhead: %lf\n",ker_launch_over[4],ker_exec_time[4],ker_launch_over[4]-ker_exec_time[4]);
     printf("Kernel: Init_array, NDRange time: %lf, Event Based Time: %lf, Launch Overhead: %lf\n",ker_launch_over[5],ker_exec_time[5],ker_launch_over[5]-ker_exec_time[5]);
     //printf("execution time for kernel %d is %lf seconds \n", i, ker_exec_time[i]);

  
}

template <class T>
void OCLStream<T>::copy()
{
    
  //(*copy_kernel)(cl::EnqueueArgs(queue, cl::NDRange(array_size)),d_a, d_c);
  clock_gettime(CLOCK_REALTIME, &start);	
  cl_int err =meta_gen_opencl_babelstream_copy(queue(), &globalWorkSize , &localwg, &d_a(), &d_c(), 0, &exec_event);
  clock_gettime(CLOCK_REALTIME, &end);
  ker_launch_over[0]+=( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec )/ BILLION;
  err = clGetEventProfilingInfo(exec_event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),  &start_time,&return_bytes);
  err = clGetEventProfilingInfo(exec_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end_time,&return_bytes);
  exec_event=NULL;
  ker_exec_time[0]+=(double)(end_time-start_time)/BILLION;
  //queue.finish();
  
}

template <class T>
void OCLStream<T>::mul()
{
  
  //(*mul_kernel)(cl::EnqueueArgs(queue, cl::NDRange(array_size)),d_b, d_c);
  clock_gettime(CLOCK_REALTIME, &start);
  cl_int err =meta_gen_opencl_babelstream_mul(queue(), &globalWorkSize , &localwg, &d_b(), &d_c(), 0, &exec_event);
  clock_gettime(CLOCK_REALTIME, &end);
  ker_launch_over[1]+= ( end.tv_sec - start.tv_sec )+ ( end.tv_nsec - start.tv_nsec )/ BILLION;
  err = clGetEventProfilingInfo(exec_event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),  &start_time,&return_bytes);
  err = clGetEventProfilingInfo(exec_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end_time,&return_bytes);
  exec_event=NULL;
  ker_exec_time[1]+=(double)(end_time-start_time)/BILLION;
//printf( "%lf\n", accum );
 //queue.finish();
}

template <class T>
void OCLStream<T>::add()
{
  //(*add_kernel)(cl::EnqueueArgs(queue, cl::NDRange(array_size)),d_a, d_b, d_c);
  clock_gettime(CLOCK_REALTIME, &start);
  cl_int err =meta_gen_opencl_babelstream_add(queue(), &globalWorkSize , &localwg, &d_a(),&d_b(), &d_c(), 0, &exec_event);
  clock_gettime(CLOCK_REALTIME, &end);
  ker_launch_over[2]+=  ( end.tv_sec - start.tv_sec )+ ( end.tv_nsec - start.tv_nsec )/ BILLION;
  err = clGetEventProfilingInfo(exec_event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),  &start_time,&return_bytes);
  err = clGetEventProfilingInfo(exec_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end_time,&return_bytes);
  exec_event=NULL;
  ker_exec_time[2]+=(double)(end_time-start_time)/BILLION;
//printf( "%lf\n", accum );
  //queue.finish();
}

template <class T>
void OCLStream<T>::triad()
{
  //(*triad_kernel)(cl::EnqueueArgs(queue, cl::NDRange(array_size)),d_a, d_b, d_c);
  clock_gettime(CLOCK_REALTIME, &start);
  cl_int err =meta_gen_opencl_babelstream_triad(queue(), &globalWorkSize,&localwg, &d_a(), &d_b(), &d_c(), 0, &exec_event);
  clock_gettime(CLOCK_REALTIME, &end);
  ker_launch_over[3]+=  ( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec )/ BILLION;
  err = clGetEventProfilingInfo(exec_event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),  &start_time,&return_bytes);
  err = clGetEventProfilingInfo(exec_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end_time,&return_bytes);
  exec_event=NULL;
  ker_exec_time[3]+=(double)(end_time-start_time)/BILLION;
//printf( "%lf\n", accum );
  //queue.finish();
}

template <class T>
T OCLStream<T>::dot()
{
  //(*dot_kernel)(cl::EnqueueArgs(queue, cl::NDRange(dot_num_groups*dot_wgsize), cl::NDRange(dot_wgsize)),d_a, d_b, d_sum, cl::Local(sizeof(T) * dot_wgsize), array_size);
  
  a_dim3 global = {dot_num_groups,1,1};
  a_dim3 local  = {dot_wgsize,1,1};
  clock_gettime(CLOCK_REALTIME, &start);
  cl_int err =meta_gen_opencl_babelstream_stream_dot(queue(), &global, &local, &d_a(), &d_b(), &d_sum(), (size_t) dot_wgsize, array_size,0, &exec_event);
  clock_gettime(CLOCK_REALTIME, &end);
  ker_launch_over[4]+=  ( end.tv_sec - start.tv_sec )+ ( end.tv_nsec - start.tv_nsec)/ BILLION;
//printf( "%lf\n", accum );
  err = clGetEventProfilingInfo(exec_event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),  &start_time,&return_bytes);
  err = clGetEventProfilingInfo(exec_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end_time,&return_bytes);
  exec_event=NULL;
  ker_exec_time[4]+=(double)(end_time-start_time)/BILLION;
  cl::copy(queue, d_sum, sums.begin(), sums.end());
  
 
  T sum = 0.0;
  for (T val : sums)
    sum += val;
 return sum;

}

template <class T>
void OCLStream<T>::init_arrays(T initA, T initB, T initC)
{
  //(*init_kernel)(cl::EnqueueArgs(queue, cl::NDRange(array_size)),d_a, d_b, d_c, initA, initB, initC );
  //a_dim3 global = { array_size,1,1};
  clock_gettime(CLOCK_REALTIME, &start);
  cl_int err= meta_gen_opencl_babelstream_init(queue(), &globalWorkSize, &localwg, &d_a(), &d_b(), &d_c(), initA,  initB,  initC, 0, &exec_event);
  clock_gettime(CLOCK_REALTIME, &end);
  ker_launch_over[5]+=  ( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec )/ BILLION;
  err = clGetEventProfilingInfo(exec_event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),  &start_time,&return_bytes);
  err = clGetEventProfilingInfo(exec_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end_time,&return_bytes);
  exec_event=NULL;
  ker_exec_time[5]+=(double)(end_time-start_time)/BILLION;
//printf( "%lf\n", accum );
  //queue.finish();
}

template <class T>
void OCLStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  cl::copy(queue, d_a, a.begin(), a.end());
  cl::copy(queue, d_b, b.begin(), b.end());
  cl::copy(queue, d_c, c.begin(), c.end());
}
/*
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
*/

template class OCLStream<float>;
template class OCLStream<double>;
