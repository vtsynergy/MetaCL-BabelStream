//Force MetaMorph to include the OpenCL code
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include "metamorph.h"
#include "mm_opencl_backend.h"
#include "metacl_module.h"
extern cl_context meta_context;
extern cl_command_queue meta_queue;
extern cl_device_id meta_device;
//TODO: Expose this with a function (with safety checks) rather than a variable
const char * __metacl_babelstream_custom_args = NULL;
struct __metacl_metacl_module_frame * __metacl_metacl_module_current_frame = NULL;

struct __metacl_metacl_module_frame * __metacl_metacl_module_lookup_frame(cl_command_queue queue) {
  struct __metacl_metacl_module_frame * current = __metacl_metacl_module_current_frame;
  while (current != NULL) {
    if (current->queue == queue) break;
    current = current->next_frame;
  }
  return current;
}
a_module_record * metacl_metacl_module_registration = NULL;
a_module_record * metacl_metacl_module_registry(a_module_record * record) {
  if (record == NULL) return metacl_metacl_module_registration;
  a_module_record * old_registration = metacl_metacl_module_registration;
  if (old_registration == NULL) {
    record->implements = module_implements_opencl;
    record->module_init = &metacl_metacl_module_init;
    record->module_deinit = &metacl_metacl_module_deinit;
    record->module_registry_func = &metacl_metacl_module_registry;
    metacl_metacl_module_registration = record;
  }
  if (old_registration != NULL && old_registration != record) return record;
  if (old_registration == record) metacl_metacl_module_registration = NULL;
  return old_registration;
}
void metacl_metacl_module_init() {
  cl_int buildError, createError;
  if (metacl_metacl_module_registration == NULL) {
    meta_register_module(&metacl_metacl_module_registry);
    return;
  }
  if (meta_context == NULL) metaOpenCLFallback();
  struct __metacl_metacl_module_frame * new_frame = (struct __metacl_metacl_module_frame *) calloc(1, sizeof(struct __metacl_metacl_module_frame));
  new_frame->next_frame = __metacl_metacl_module_current_frame;
  new_frame->device = meta_device;
  new_frame->queue = meta_queue;
  new_frame->context = meta_context;
  __metacl_metacl_module_current_frame = new_frame;
  meta_cl_device_vendor vendor = metaOpenCLDetectDevice(new_frame->device);
  if ((vendor & meta_cl_device_is_accel) && ((vendor & meta_cl_device_vendor_mask) == meta_cl_device_vendor_intelfpga)) {
    __metacl_metacl_module_current_frame->babelstream_progLen = metaOpenCLLoadProgramSource("babelstream.aocx", &__metacl_metacl_module_current_frame->babelstream_progSrc);
    if (__metacl_metacl_module_current_frame->babelstream_progLen != -1)
      __metacl_metacl_module_current_frame->babelstream_prog = clCreateProgramWithBinary(meta_context, 1, &meta_device, &__metacl_metacl_module_current_frame->babelstream_progLen, (const unsigned char **)&__metacl_metacl_module_current_frame->babelstream_progSrc, NULL, &buildError);
  } else {
    __metacl_metacl_module_current_frame->babelstream_progLen = metaOpenCLLoadProgramSource("babelstream.cl", &__metacl_metacl_module_current_frame->babelstream_progSrc);
    if (__metacl_metacl_module_current_frame->babelstream_progLen != -1)
      __metacl_metacl_module_current_frame->babelstream_prog = clCreateProgramWithSource(meta_context, 1, &__metacl_metacl_module_current_frame->babelstream_progSrc, &__metacl_metacl_module_current_frame->babelstream_progLen, &buildError);
  }
  if (__metacl_metacl_module_current_frame->babelstream_progLen != -1) {
    if (buildError != CL_SUCCESS) fprintf(stderr, "OpenCL program creation error %d at %s:%d\n", buildError, __FILE__, __LINE__);
    buildError = clBuildProgram(__metacl_metacl_module_current_frame->babelstream_prog, 1, &meta_device, __metacl_babelstream_custom_args ? __metacl_babelstream_custom_args : "", NULL, NULL);
    if (buildError != CL_SUCCESS) {
      size_t logsize = 0;
      clGetProgramBuildInfo(__metacl_metacl_module_current_frame->babelstream_prog, meta_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
      char * buildLog = (char *) malloc(sizeof(char) * (logsize + 1));
      clGetProgramBuildInfo(__metacl_metacl_module_current_frame->babelstream_prog, meta_device, CL_PROGRAM_BUILD_LOG, logsize, buildLog, NULL);
      if (buildError != CL_SUCCESS) fprintf(stderr, "OpenCL program build error %d at %s:%d\n", buildError, __FILE__, __LINE__);
      fprintf(stderr, "Build Log:\n%s\n", buildLog);
      free(buildLog);
    } else {
      __metacl_metacl_module_current_frame->babelstream_init = 1;
    }
    __metacl_metacl_module_current_frame->babelstream_init_kernel = clCreateKernel(__metacl_metacl_module_current_frame->babelstream_prog, "init", &createError);
    if (createError != CL_SUCCESS) fprintf(stderr, "OpenCL kernel creation error %d at %s:%d\n", createError, __FILE__, __LINE__);
    __metacl_metacl_module_current_frame->babelstream_copy_kernel = clCreateKernel(__metacl_metacl_module_current_frame->babelstream_prog, "copy", &createError);
    if (createError != CL_SUCCESS) fprintf(stderr, "OpenCL kernel creation error %d at %s:%d\n", createError, __FILE__, __LINE__);
    __metacl_metacl_module_current_frame->babelstream_mul_kernel = clCreateKernel(__metacl_metacl_module_current_frame->babelstream_prog, "mul", &createError);
    if (createError != CL_SUCCESS) fprintf(stderr, "OpenCL kernel creation error %d at %s:%d\n", createError, __FILE__, __LINE__);
    __metacl_metacl_module_current_frame->babelstream_add_kernel = clCreateKernel(__metacl_metacl_module_current_frame->babelstream_prog, "add", &createError);
    if (createError != CL_SUCCESS) fprintf(stderr, "OpenCL kernel creation error %d at %s:%d\n", createError, __FILE__, __LINE__);
    __metacl_metacl_module_current_frame->babelstream_triad_kernel = clCreateKernel(__metacl_metacl_module_current_frame->babelstream_prog, "triad", &createError);
    if (createError != CL_SUCCESS) fprintf(stderr, "OpenCL kernel creation error %d at %s:%d\n", createError, __FILE__, __LINE__);
    __metacl_metacl_module_current_frame->babelstream_stream_dot_kernel = clCreateKernel(__metacl_metacl_module_current_frame->babelstream_prog, "stream_dot", &createError);
    if (createError != CL_SUCCESS) fprintf(stderr, "OpenCL kernel creation error %d at %s:%d\n", createError, __FILE__, __LINE__);
  }
  metacl_metacl_module_registration->initialized = 1;
}

void metacl_metacl_module_deinit() {
  cl_int releaseError;
  if (__metacl_metacl_module_current_frame != NULL) {
    if (__metacl_metacl_module_current_frame->babelstream_progLen != -1) {
      releaseError = clReleaseKernel(__metacl_metacl_module_current_frame->babelstream_init_kernel);
      if (releaseError != CL_SUCCESS) fprintf(stderr, "OpenCL kernel release error %d at %s:%d\n", releaseError, __FILE__, __LINE__);
      releaseError = clReleaseKernel(__metacl_metacl_module_current_frame->babelstream_copy_kernel);
      if (releaseError != CL_SUCCESS) fprintf(stderr, "OpenCL kernel release error %d at %s:%d\n", releaseError, __FILE__, __LINE__);
      releaseError = clReleaseKernel(__metacl_metacl_module_current_frame->babelstream_mul_kernel);
      if (releaseError != CL_SUCCESS) fprintf(stderr, "OpenCL kernel release error %d at %s:%d\n", releaseError, __FILE__, __LINE__);
      releaseError = clReleaseKernel(__metacl_metacl_module_current_frame->babelstream_add_kernel);
      if (releaseError != CL_SUCCESS) fprintf(stderr, "OpenCL kernel release error %d at %s:%d\n", releaseError, __FILE__, __LINE__);
      releaseError = clReleaseKernel(__metacl_metacl_module_current_frame->babelstream_triad_kernel);
      if (releaseError != CL_SUCCESS) fprintf(stderr, "OpenCL kernel release error %d at %s:%d\n", releaseError, __FILE__, __LINE__);
      releaseError = clReleaseKernel(__metacl_metacl_module_current_frame->babelstream_stream_dot_kernel);
      if (releaseError != CL_SUCCESS) fprintf(stderr, "OpenCL kernel release error %d at %s:%d\n", releaseError, __FILE__, __LINE__);
      releaseError = clReleaseProgram(__metacl_metacl_module_current_frame->babelstream_prog);
      if (releaseError != CL_SUCCESS) fprintf(stderr, "OpenCL program release error %d at %s:%d\n", releaseError, __FILE__, __LINE__);
      free((char *)__metacl_metacl_module_current_frame->babelstream_progSrc);
      __metacl_metacl_module_current_frame->babelstream_progLen = 0;
      __metacl_metacl_module_current_frame->babelstream_init = 0;
    }
    struct __metacl_metacl_module_frame * next_frame = __metacl_metacl_module_current_frame->next_frame;
    free(__metacl_metacl_module_current_frame);
    __metacl_metacl_module_current_frame = next_frame;
    if (__metacl_metacl_module_current_frame == NULL && metacl_metacl_module_registration != NULL) {
      metacl_metacl_module_registration->initialized = 0;
    }
  }
}

cl_int metacl_babelstream_init(cl_command_queue queue, size_t (*grid_size)[3], size_t (*block_size)[3], size_t (*meta_offset)[3], int async, cl_event * event, cl_mem * a, cl_mem * b, cl_mem * c, cl_double initA, cl_double initB, cl_double initC) {
  if (metacl_metacl_module_registration == NULL) meta_register_module(&metacl_metacl_module_registry);
  struct __metacl_metacl_module_frame * frame = __metacl_metacl_module_current_frame;
  if (queue != NULL) frame = __metacl_metacl_module_lookup_frame(queue);
  //If the user requests a queue this module doesn't know about, or a NULL queue and there is no current frame
  if (frame == NULL) return CL_INVALID_COMMAND_QUEUE;
  if (frame->babelstream_init != 1) return CL_INVALID_PROGRAM;
  cl_int retCode = CL_SUCCESS;
  a_bool nullBlock = (block_size != NULL && (*block_size)[0] == 0 && (*block_size)[1] == 0 && (*block_size)[2] == 0);
  size_t _global_size[3];
  size_t _local_size[3] = METAMORPH_OCL_DEFAULT_BLOCK_3D;
  size_t _temp_offset[3];
  if (meta_offset == NULL) { _temp_offset[0] = 0, _temp_offset[1] = 0, _temp_offset[2] = 0;}
  else { _temp_offset[0] = (*meta_offset)[0], _temp_offset[1] = (*meta_offset)[1], _temp_offset[2] = (*meta_offset)[2]; }
  const size_t _meta_offset[3] = {_temp_offset[0], _temp_offset[1], _temp_offset[2]};
  int iters;

  //Default runs a single workgroup
  if (grid_size == NULL || block_size == NULL) {
    _global_size[0] = _local_size[0];
    _global_size[1] = _local_size[1];
    _global_size[2] = _local_size[2];
    iters = 1;
  } else {
    _global_size[0] = (*grid_size)[0] * (nullBlock ? 1 : (*block_size)[0]);
    _global_size[1] = (*grid_size)[1] * (nullBlock ? 1 : (*block_size)[1]);
    _global_size[2] = (*grid_size)[2] * (nullBlock ? 1 : (*block_size)[2]);
    _local_size[0] = (*block_size)[0];
    _local_size[1] = (*block_size)[1];
    _local_size[2] = (*block_size)[2];
  }
  retCode = clSetKernelArg(frame->babelstream_init_kernel, 0, sizeof(cl_mem), a);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"a\", host wrapper: \"metacl_babelstream_init\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clSetKernelArg(frame->babelstream_init_kernel, 1, sizeof(cl_mem), b);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"b\", host wrapper: \"metacl_babelstream_init\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clSetKernelArg(frame->babelstream_init_kernel, 2, sizeof(cl_mem), c);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"c\", host wrapper: \"metacl_babelstream_init\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clSetKernelArg(frame->babelstream_init_kernel, 3, sizeof(cl_double), &initA);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"initA\", host wrapper: \"metacl_babelstream_init\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clSetKernelArg(frame->babelstream_init_kernel, 4, sizeof(cl_double), &initB);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"initB\", host wrapper: \"metacl_babelstream_init\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clSetKernelArg(frame->babelstream_init_kernel, 5, sizeof(cl_double), &initC);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"initC\", host wrapper: \"metacl_babelstream_init\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clEnqueueNDRangeKernel(frame->queue, frame->babelstream_init_kernel, 3, &_meta_offset[0], _global_size, (nullBlock ? NULL : _local_size), 0, NULL, event);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel enqueue error (host wrapper: \"metacl_babelstream_init\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  if (!async) {
    retCode = clFinish(frame->queue);
    if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel execution error (host wrapper: \"metacl_babelstream_init\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  }
  return retCode;
}

cl_int metacl_babelstream_copy(cl_command_queue queue, size_t (*grid_size)[3], size_t (*block_size)[3], size_t (*meta_offset)[3], int async, cl_event * event, cl_mem * a, cl_mem * c) {
  if (metacl_metacl_module_registration == NULL) meta_register_module(&metacl_metacl_module_registry);
  struct __metacl_metacl_module_frame * frame = __metacl_metacl_module_current_frame;
  if (queue != NULL) frame = __metacl_metacl_module_lookup_frame(queue);
  //If the user requests a queue this module doesn't know about, or a NULL queue and there is no current frame
  if (frame == NULL) return CL_INVALID_COMMAND_QUEUE;
  if (frame->babelstream_init != 1) return CL_INVALID_PROGRAM;
  cl_int retCode = CL_SUCCESS;
  a_bool nullBlock = (block_size != NULL && (*block_size)[0] == 0 && (*block_size)[1] == 0 && (*block_size)[2] == 0);
  size_t _global_size[3];
  size_t _local_size[3] = METAMORPH_OCL_DEFAULT_BLOCK_3D;
  size_t _temp_offset[3];
  if (meta_offset == NULL) { _temp_offset[0] = 0, _temp_offset[1] = 0, _temp_offset[2] = 0;}
  else { _temp_offset[0] = (*meta_offset)[0], _temp_offset[1] = (*meta_offset)[1], _temp_offset[2] = (*meta_offset)[2]; }
  const size_t _meta_offset[3] = {_temp_offset[0], _temp_offset[1], _temp_offset[2]};
  int iters;

  //Default runs a single workgroup
  if (grid_size == NULL || block_size == NULL) {
    _global_size[0] = _local_size[0];
    _global_size[1] = _local_size[1];
    _global_size[2] = _local_size[2];
    iters = 1;
  } else {
    _global_size[0] = (*grid_size)[0] * (nullBlock ? 1 : (*block_size)[0]);
    _global_size[1] = (*grid_size)[1] * (nullBlock ? 1 : (*block_size)[1]);
    _global_size[2] = (*grid_size)[2] * (nullBlock ? 1 : (*block_size)[2]);
    _local_size[0] = (*block_size)[0];
    _local_size[1] = (*block_size)[1];
    _local_size[2] = (*block_size)[2];
  }
  retCode = clSetKernelArg(frame->babelstream_copy_kernel, 0, sizeof(cl_mem), a);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"a\", host wrapper: \"metacl_babelstream_copy\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clSetKernelArg(frame->babelstream_copy_kernel, 1, sizeof(cl_mem), c);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"c\", host wrapper: \"metacl_babelstream_copy\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clEnqueueNDRangeKernel(frame->queue, frame->babelstream_copy_kernel, 3, &_meta_offset[0], _global_size, (nullBlock ? NULL : _local_size), 0, NULL, event);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel enqueue error (host wrapper: \"metacl_babelstream_copy\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  if (!async) {
    retCode = clFinish(frame->queue);
    if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel execution error (host wrapper: \"metacl_babelstream_copy\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  }
  return retCode;
}

cl_int metacl_babelstream_mul(cl_command_queue queue, size_t (*grid_size)[3], size_t (*block_size)[3], size_t (*meta_offset)[3], int async, cl_event * event, cl_mem * b, cl_mem * c) {
  if (metacl_metacl_module_registration == NULL) meta_register_module(&metacl_metacl_module_registry);
  struct __metacl_metacl_module_frame * frame = __metacl_metacl_module_current_frame;
  if (queue != NULL) frame = __metacl_metacl_module_lookup_frame(queue);
  //If the user requests a queue this module doesn't know about, or a NULL queue and there is no current frame
  if (frame == NULL) return CL_INVALID_COMMAND_QUEUE;
  if (frame->babelstream_init != 1) return CL_INVALID_PROGRAM;
  cl_int retCode = CL_SUCCESS;
  a_bool nullBlock = (block_size != NULL && (*block_size)[0] == 0 && (*block_size)[1] == 0 && (*block_size)[2] == 0);
  size_t _global_size[3];
  size_t _local_size[3] = METAMORPH_OCL_DEFAULT_BLOCK_3D;
  size_t _temp_offset[3];
  if (meta_offset == NULL) { _temp_offset[0] = 0, _temp_offset[1] = 0, _temp_offset[2] = 0;}
  else { _temp_offset[0] = (*meta_offset)[0], _temp_offset[1] = (*meta_offset)[1], _temp_offset[2] = (*meta_offset)[2]; }
  const size_t _meta_offset[3] = {_temp_offset[0], _temp_offset[1], _temp_offset[2]};
  int iters;

  //Default runs a single workgroup
  if (grid_size == NULL || block_size == NULL) {
    _global_size[0] = _local_size[0];
    _global_size[1] = _local_size[1];
    _global_size[2] = _local_size[2];
    iters = 1;
  } else {
    _global_size[0] = (*grid_size)[0] * (nullBlock ? 1 : (*block_size)[0]);
    _global_size[1] = (*grid_size)[1] * (nullBlock ? 1 : (*block_size)[1]);
    _global_size[2] = (*grid_size)[2] * (nullBlock ? 1 : (*block_size)[2]);
    _local_size[0] = (*block_size)[0];
    _local_size[1] = (*block_size)[1];
    _local_size[2] = (*block_size)[2];
  }
  retCode = clSetKernelArg(frame->babelstream_mul_kernel, 0, sizeof(cl_mem), b);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"b\", host wrapper: \"metacl_babelstream_mul\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clSetKernelArg(frame->babelstream_mul_kernel, 1, sizeof(cl_mem), c);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"c\", host wrapper: \"metacl_babelstream_mul\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clEnqueueNDRangeKernel(frame->queue, frame->babelstream_mul_kernel, 3, &_meta_offset[0], _global_size, (nullBlock ? NULL : _local_size), 0, NULL, event);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel enqueue error (host wrapper: \"metacl_babelstream_mul\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  if (!async) {
    retCode = clFinish(frame->queue);
    if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel execution error (host wrapper: \"metacl_babelstream_mul\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  }
  return retCode;
}

cl_int metacl_babelstream_add(cl_command_queue queue, size_t (*grid_size)[3], size_t (*block_size)[3], size_t (*meta_offset)[3], int async, cl_event * event, cl_mem * a, cl_mem * b, cl_mem * c) {
  if (metacl_metacl_module_registration == NULL) meta_register_module(&metacl_metacl_module_registry);
  struct __metacl_metacl_module_frame * frame = __metacl_metacl_module_current_frame;
  if (queue != NULL) frame = __metacl_metacl_module_lookup_frame(queue);
  //If the user requests a queue this module doesn't know about, or a NULL queue and there is no current frame
  if (frame == NULL) return CL_INVALID_COMMAND_QUEUE;
  if (frame->babelstream_init != 1) return CL_INVALID_PROGRAM;
  cl_int retCode = CL_SUCCESS;
  a_bool nullBlock = (block_size != NULL && (*block_size)[0] == 0 && (*block_size)[1] == 0 && (*block_size)[2] == 0);
  size_t _global_size[3];
  size_t _local_size[3] = METAMORPH_OCL_DEFAULT_BLOCK_3D;
  size_t _temp_offset[3];
  if (meta_offset == NULL) { _temp_offset[0] = 0, _temp_offset[1] = 0, _temp_offset[2] = 0;}
  else { _temp_offset[0] = (*meta_offset)[0], _temp_offset[1] = (*meta_offset)[1], _temp_offset[2] = (*meta_offset)[2]; }
  const size_t _meta_offset[3] = {_temp_offset[0], _temp_offset[1], _temp_offset[2]};
  int iters;

  //Default runs a single workgroup
  if (grid_size == NULL || block_size == NULL) {
    _global_size[0] = _local_size[0];
    _global_size[1] = _local_size[1];
    _global_size[2] = _local_size[2];
    iters = 1;
  } else {
    _global_size[0] = (*grid_size)[0] * (nullBlock ? 1 : (*block_size)[0]);
    _global_size[1] = (*grid_size)[1] * (nullBlock ? 1 : (*block_size)[1]);
    _global_size[2] = (*grid_size)[2] * (nullBlock ? 1 : (*block_size)[2]);
    _local_size[0] = (*block_size)[0];
    _local_size[1] = (*block_size)[1];
    _local_size[2] = (*block_size)[2];
  }
  retCode = clSetKernelArg(frame->babelstream_add_kernel, 0, sizeof(cl_mem), a);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"a\", host wrapper: \"metacl_babelstream_add\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clSetKernelArg(frame->babelstream_add_kernel, 1, sizeof(cl_mem), b);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"b\", host wrapper: \"metacl_babelstream_add\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clSetKernelArg(frame->babelstream_add_kernel, 2, sizeof(cl_mem), c);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"c\", host wrapper: \"metacl_babelstream_add\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clEnqueueNDRangeKernel(frame->queue, frame->babelstream_add_kernel, 3, &_meta_offset[0], _global_size, (nullBlock ? NULL : _local_size), 0, NULL, event);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel enqueue error (host wrapper: \"metacl_babelstream_add\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  if (!async) {
    retCode = clFinish(frame->queue);
    if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel execution error (host wrapper: \"metacl_babelstream_add\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  }
  return retCode;
}

cl_int metacl_babelstream_triad(cl_command_queue queue, size_t (*grid_size)[3], size_t (*block_size)[3], size_t (*meta_offset)[3], int async, cl_event * event, cl_mem * a, cl_mem * b, cl_mem * c) {
  if (metacl_metacl_module_registration == NULL) meta_register_module(&metacl_metacl_module_registry);
  struct __metacl_metacl_module_frame * frame = __metacl_metacl_module_current_frame;
  if (queue != NULL) frame = __metacl_metacl_module_lookup_frame(queue);
  //If the user requests a queue this module doesn't know about, or a NULL queue and there is no current frame
  if (frame == NULL) return CL_INVALID_COMMAND_QUEUE;
  if (frame->babelstream_init != 1) return CL_INVALID_PROGRAM;
  cl_int retCode = CL_SUCCESS;
  a_bool nullBlock = (block_size != NULL && (*block_size)[0] == 0 && (*block_size)[1] == 0 && (*block_size)[2] == 0);
  size_t _global_size[3];
  size_t _local_size[3] = METAMORPH_OCL_DEFAULT_BLOCK_3D;
  size_t _temp_offset[3];
  if (meta_offset == NULL) { _temp_offset[0] = 0, _temp_offset[1] = 0, _temp_offset[2] = 0;}
  else { _temp_offset[0] = (*meta_offset)[0], _temp_offset[1] = (*meta_offset)[1], _temp_offset[2] = (*meta_offset)[2]; }
  const size_t _meta_offset[3] = {_temp_offset[0], _temp_offset[1], _temp_offset[2]};
  int iters;

  //Default runs a single workgroup
  if (grid_size == NULL || block_size == NULL) {
    _global_size[0] = _local_size[0];
    _global_size[1] = _local_size[1];
    _global_size[2] = _local_size[2];
    iters = 1;
  } else {
    _global_size[0] = (*grid_size)[0] * (nullBlock ? 1 : (*block_size)[0]);
    _global_size[1] = (*grid_size)[1] * (nullBlock ? 1 : (*block_size)[1]);
    _global_size[2] = (*grid_size)[2] * (nullBlock ? 1 : (*block_size)[2]);
    _local_size[0] = (*block_size)[0];
    _local_size[1] = (*block_size)[1];
    _local_size[2] = (*block_size)[2];
  }
  retCode = clSetKernelArg(frame->babelstream_triad_kernel, 0, sizeof(cl_mem), a);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"a\", host wrapper: \"metacl_babelstream_triad\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clSetKernelArg(frame->babelstream_triad_kernel, 1, sizeof(cl_mem), b);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"b\", host wrapper: \"metacl_babelstream_triad\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clSetKernelArg(frame->babelstream_triad_kernel, 2, sizeof(cl_mem), c);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"c\", host wrapper: \"metacl_babelstream_triad\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clEnqueueNDRangeKernel(frame->queue, frame->babelstream_triad_kernel, 3, &_meta_offset[0], _global_size, (nullBlock ? NULL : _local_size), 0, NULL, event);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel enqueue error (host wrapper: \"metacl_babelstream_triad\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  if (!async) {
    retCode = clFinish(frame->queue);
    if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel execution error (host wrapper: \"metacl_babelstream_triad\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  }
  return retCode;
}

cl_int metacl_babelstream_stream_dot(cl_command_queue queue, size_t (*grid_size)[3], size_t (*block_size)[3], size_t (*meta_offset)[3], int async, cl_event * event, cl_mem * a, cl_mem * b, cl_mem * sum, size_t wg_sum_num_local_elems, cl_int array_size) {
  if (metacl_metacl_module_registration == NULL) meta_register_module(&metacl_metacl_module_registry);
  struct __metacl_metacl_module_frame * frame = __metacl_metacl_module_current_frame;
  if (queue != NULL) frame = __metacl_metacl_module_lookup_frame(queue);
  //If the user requests a queue this module doesn't know about, or a NULL queue and there is no current frame
  if (frame == NULL) return CL_INVALID_COMMAND_QUEUE;
  if (frame->babelstream_init != 1) return CL_INVALID_PROGRAM;
  cl_int retCode = CL_SUCCESS;
  a_bool nullBlock = (block_size != NULL && (*block_size)[0] == 0 && (*block_size)[1] == 0 && (*block_size)[2] == 0);
  size_t _global_size[3];
  size_t _local_size[3] = METAMORPH_OCL_DEFAULT_BLOCK_3D;
  size_t _temp_offset[3];
  if (meta_offset == NULL) { _temp_offset[0] = 0, _temp_offset[1] = 0, _temp_offset[2] = 0;}
  else { _temp_offset[0] = (*meta_offset)[0], _temp_offset[1] = (*meta_offset)[1], _temp_offset[2] = (*meta_offset)[2]; }
  const size_t _meta_offset[3] = {_temp_offset[0], _temp_offset[1], _temp_offset[2]};
  int iters;

  //Default runs a single workgroup
  if (grid_size == NULL || block_size == NULL) {
    _global_size[0] = _local_size[0];
    _global_size[1] = _local_size[1];
    _global_size[2] = _local_size[2];
    iters = 1;
  } else {
    _global_size[0] = (*grid_size)[0] * (nullBlock ? 1 : (*block_size)[0]);
    _global_size[1] = (*grid_size)[1] * (nullBlock ? 1 : (*block_size)[1]);
    _global_size[2] = (*grid_size)[2] * (nullBlock ? 1 : (*block_size)[2]);
    _local_size[0] = (*block_size)[0];
    _local_size[1] = (*block_size)[1];
    _local_size[2] = (*block_size)[2];
  }
  retCode = clSetKernelArg(frame->babelstream_stream_dot_kernel, 0, sizeof(cl_mem), a);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"a\", host wrapper: \"metacl_babelstream_stream_dot\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clSetKernelArg(frame->babelstream_stream_dot_kernel, 1, sizeof(cl_mem), b);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"b\", host wrapper: \"metacl_babelstream_stream_dot\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clSetKernelArg(frame->babelstream_stream_dot_kernel, 2, sizeof(cl_mem), sum);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"sum\", host wrapper: \"metacl_babelstream_stream_dot\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clSetKernelArg(frame->babelstream_stream_dot_kernel, 3, sizeof(cl_double) * wg_sum_num_local_elems, NULL);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"wg_sum\", host wrapper: \"metacl_babelstream_stream_dot\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clSetKernelArg(frame->babelstream_stream_dot_kernel, 4, sizeof(cl_int), &array_size);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel argument assignment error (arg: \"array_size\", host wrapper: \"metacl_babelstream_stream_dot\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  retCode = clEnqueueNDRangeKernel(frame->queue, frame->babelstream_stream_dot_kernel, 3, &_meta_offset[0], _global_size, (nullBlock ? NULL : _local_size), 0, NULL, event);
  if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel enqueue error (host wrapper: \"metacl_babelstream_stream_dot\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  if (!async) {
    retCode = clFinish(frame->queue);
    if (retCode != CL_SUCCESS) fprintf(stderr, "OpenCL kernel execution error (host wrapper: \"metacl_babelstream_stream_dot\") %d at %s:%d\n", retCode, __FILE__, __LINE__);
  }
  return retCode;
}

