#!/bin/bash
./namd3 +ppn 64 +devices 0 +setcpuaffinity +idlepoll apoa1_nve_cuda_soa.namd
