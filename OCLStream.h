
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include "CL/cl2.hpp"

#include "Stream.h"

#define IMPLEMENTATION_STRING "OpenCL"

template <class T>
class OCLStream : public Stream<T>
{
  protected:
    // Size of arrays
    unsigned int array_size;

    // Host array for partial sums for dot kernel
    std::vector<T> sums;

    std::vector<double> ker_launch_over;
    std::vector<double> ker_exec_time;
    std::vector<std::vector<double>> ker_exec_time_rec;
    std::vector<std::vector<double>> ker_launch_over_rec;

    // Device side pointers to arrays
    cl::Buffer d_a;
    cl::Buffer d_b;
    cl::Buffer d_c;
    cl::Buffer d_sum;

    int it_monitor=0;
    // OpenCL objects
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;

    // NDRange configuration for the dot kernel
    size_t dot_num_groups;
    size_t dot_wgsize;

  public:

    OCLStream(const unsigned int, const int);
    ~OCLStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual T dot() override;

    virtual void print_res() override;
    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};

// Populate the devices list
void getDeviceList(void);
