extern const char * __metacl_babelstream_custom_args;
struct __metacl_metacl_module_frame;
struct __metacl_metacl_module_frame {
  struct __metacl_metacl_module_frame * next_frame;
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
struct __metacl_metacl_module_frame * __metacl_metacl_module_lookup_frame(cl_command_queue queue);
#ifdef __cplusplus
extern "C" {
#endif
a_module_record * metacl_metacl_module_registry(a_module_record * record);
void metacl_metacl_module_init();
void metacl_metacl_module_deinit();
/** Automatically-generated by MetaCL
\param queue the cl_command_queue the kernel is being prepared to run on (to lookup and/or enqueue associated cl_kernel)
\param global_size a size_t[3] providing the global number of workitems in the X, Y, Z dimensions\param local_size a size_t[3] providing the workgroup size in the X, Y, Z dimensions
\param meta_offset the NDRange offset, NULL if none
\param async whether the kernel should run asynchronously
\param event returns the cl_event corresponding to the kernel launch if run asynchronously
\param a a cl_mem buffer, must internally store cl_double types
\param b a cl_mem buffer, must internally store cl_double types
\param c a cl_mem buffer, must internally store cl_double types
\param initA scalar parameter of type "cl_double"
\param initB scalar parameter of type "cl_double"
\param initC scalar parameter of type "cl_double"
 */
cl_int metacl_babelstream_init(cl_command_queue queue, size_t (*global_size)[3], size_t (*local_size)[3], size_t (*meta_offset)[3], int async, cl_event * event, cl_mem * a, cl_mem * b, cl_mem * c, cl_double initA, cl_double initB, cl_double initC);
/** Automatically-generated by MetaCL
\param queue the cl_command_queue the kernel is being prepared to run on (to lookup and/or enqueue associated cl_kernel)
\param global_size a size_t[3] providing the global number of workitems in the X, Y, Z dimensions\param local_size a size_t[3] providing the workgroup size in the X, Y, Z dimensions
\param meta_offset the NDRange offset, NULL if none
\param async whether the kernel should run asynchronously
\param event returns the cl_event corresponding to the kernel launch if run asynchronously
\param a a cl_mem buffer, must internally store cl_double types
\param c a cl_mem buffer, must internally store cl_double types
 */
cl_int metacl_babelstream_copy(cl_command_queue queue, size_t (*global_size)[3], size_t (*local_size)[3], size_t (*meta_offset)[3], int async, cl_event * event, cl_mem * a, cl_mem * c);
/** Automatically-generated by MetaCL
\param queue the cl_command_queue the kernel is being prepared to run on (to lookup and/or enqueue associated cl_kernel)
\param global_size a size_t[3] providing the global number of workitems in the X, Y, Z dimensions\param local_size a size_t[3] providing the workgroup size in the X, Y, Z dimensions
\param meta_offset the NDRange offset, NULL if none
\param async whether the kernel should run asynchronously
\param event returns the cl_event corresponding to the kernel launch if run asynchronously
\param b a cl_mem buffer, must internally store cl_double types
\param c a cl_mem buffer, must internally store cl_double types
 */
cl_int metacl_babelstream_mul(cl_command_queue queue, size_t (*global_size)[3], size_t (*local_size)[3], size_t (*meta_offset)[3], int async, cl_event * event, cl_mem * b, cl_mem * c);
/** Automatically-generated by MetaCL
\param queue the cl_command_queue the kernel is being prepared to run on (to lookup and/or enqueue associated cl_kernel)
\param global_size a size_t[3] providing the global number of workitems in the X, Y, Z dimensions\param local_size a size_t[3] providing the workgroup size in the X, Y, Z dimensions
\param meta_offset the NDRange offset, NULL if none
\param async whether the kernel should run asynchronously
\param event returns the cl_event corresponding to the kernel launch if run asynchronously
\param a a cl_mem buffer, must internally store cl_double types
\param b a cl_mem buffer, must internally store cl_double types
\param c a cl_mem buffer, must internally store cl_double types
 */
cl_int metacl_babelstream_add(cl_command_queue queue, size_t (*global_size)[3], size_t (*local_size)[3], size_t (*meta_offset)[3], int async, cl_event * event, cl_mem * a, cl_mem * b, cl_mem * c);
/** Automatically-generated by MetaCL
\param queue the cl_command_queue the kernel is being prepared to run on (to lookup and/or enqueue associated cl_kernel)
\param global_size a size_t[3] providing the global number of workitems in the X, Y, Z dimensions\param local_size a size_t[3] providing the workgroup size in the X, Y, Z dimensions
\param meta_offset the NDRange offset, NULL if none
\param async whether the kernel should run asynchronously
\param event returns the cl_event corresponding to the kernel launch if run asynchronously
\param a a cl_mem buffer, must internally store cl_double types
\param b a cl_mem buffer, must internally store cl_double types
\param c a cl_mem buffer, must internally store cl_double types
 */
cl_int metacl_babelstream_triad(cl_command_queue queue, size_t (*global_size)[3], size_t (*local_size)[3], size_t (*meta_offset)[3], int async, cl_event * event, cl_mem * a, cl_mem * b, cl_mem * c);
/** Automatically-generated by MetaCL
\param queue the cl_command_queue the kernel is being prepared to run on (to lookup and/or enqueue associated cl_kernel)
\param global_size a size_t[3] providing the global number of workitems in the X, Y, Z dimensions\param local_size a size_t[3] providing the workgroup size in the X, Y, Z dimensions
\param meta_offset the NDRange offset, NULL if none
\param async whether the kernel should run asynchronously
\param event returns the cl_event corresponding to the kernel launch if run asynchronously
\param a a cl_mem buffer, must internally store cl_double types
\param b a cl_mem buffer, must internally store cl_double types
\param sum a cl_mem buffer, must internally store cl_double types
\param wg_sum_num_local_elems allocate __local memory space for this many cl_double elements
\param array_size scalar parameter of type "cl_int"
 */
cl_int metacl_babelstream_stream_dot(cl_command_queue queue, size_t (*global_size)[3], size_t (*local_size)[3], size_t (*meta_offset)[3], int async, cl_event * event, cl_mem * a, cl_mem * b, cl_mem * sum, size_t wg_sum_num_local_elems, cl_int array_size);
#ifdef __cplusplus
}
#endif