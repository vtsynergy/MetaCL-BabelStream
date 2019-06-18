extern const char * __meta_gen_opencl_babelstream_custom_args;
struct __meta_gen_opencl_metacl_module_frame;
struct __meta_gen_opencl_metacl_module_frame {
  struct __meta_gen_opencl_metacl_module_frame * next_frame;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  const char * babelstream_progSrc;
  size_t babelstream_progLen;
  cl_program babelstream_prog;
  cl_int babelstream_init;
  cl_kernel babelstream_init_kernel;
  cl_kernel babelstream_copy_kernel;
  cl_kernel babelstream_mul_kernel;
  cl_kernel babelstream_add_kernel;
  cl_kernel babelstream_triad_kernel;
  cl_kernel babelstream_stream_dot_kernel;
};
#ifdef __cplusplus
extern "C" {
#endif
a_module_record * meta_gen_opencl_metacl_module_registry(a_module_record * record);
void meta_gen_opencl_metacl_module_init();
void meta_gen_opencl_metacl_module_deinit();
cl_int meta_gen_opencl_babelstream_init(cl_command_queue queue, size_t (*grid_size)[3], size_t (*block_size)[3], cl_mem * a, cl_mem * b, cl_mem * c, cl_double initA, cl_double initB, cl_double initC, int async, cl_event * event);
cl_int meta_gen_opencl_babelstream_copy(cl_command_queue queue, size_t (*grid_size)[3], size_t (*block_size)[3], cl_mem * a, cl_mem * c, int async, cl_event * event);
cl_int meta_gen_opencl_babelstream_mul(cl_command_queue queue, size_t (*grid_size)[3], size_t (*block_size)[3], cl_mem * b, cl_mem * c, int async, cl_event * event);
cl_int meta_gen_opencl_babelstream_add(cl_command_queue queue, size_t (*grid_size)[3], size_t (*block_size)[3], cl_mem * a, cl_mem * b, cl_mem * c, int async, cl_event * event);
cl_int meta_gen_opencl_babelstream_triad(cl_command_queue queue, size_t (*grid_size)[3], size_t (*block_size)[3], cl_mem * a, cl_mem * b, cl_mem * c, int async, cl_event * event);
cl_int meta_gen_opencl_babelstream_stream_dot(cl_command_queue queue, size_t (*grid_size)[3], size_t (*block_size)[3], cl_mem * a, cl_mem * b, cl_mem * sum, size_t wg_sum_num_local_elems, cl_int array_size, int async, cl_event * event);
#ifdef __cplusplus
}
#endif
