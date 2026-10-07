// Bench-only EXP-045. Generic kernel included unchanged from pinned MIOpen MIT source.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <stdint.h>
#define MIOPEN_USE_BFP16 1
#include "MIOpenIm3d2Col.cpp"

// Same scalar short-copy indexing algorithm, with only fixed geometry and
// zero backend padding/unit stride/dilation specialized at compile time.
template<unsigned C, unsigned D> __global__ void Im3d2ColFixed(
    short* const __restrict im, short* __restrict col)
{
    constexpr unsigned out_d_size=D-2, out_h_size=128, out_w_size=128;
    constexpr unsigned wei_d_size=3, wei_h_size=3, wei_w_size=3;
    constexpr unsigned im_d_size=D, im_h_size=130, im_w_size=130;
    constexpr unsigned col_size=out_d_size*out_h_size*out_w_size*27*C;
    unsigned gtid=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned global_size=blockDim.x*gridDim.x;
    for(unsigned tid=gtid;tid<col_size;tid+=global_size) {
        unsigned col_i=tid/(out_d_size*out_h_size*out_w_size);
        unsigned col_j=tid-col_i*(out_d_size*out_h_size*out_w_size);
        unsigned out_d=col_j/(out_h_size*out_w_size);
        unsigned tmp=col_j-out_d*(out_h_size*out_w_size);
        unsigned out_h=tmp/out_w_size;
        unsigned out_w=tmp-out_h*out_w_size;
        unsigned wei_c=col_i/(wei_d_size*wei_h_size*wei_w_size);
        tmp=col_i-wei_c*(wei_d_size*wei_h_size*wei_w_size);
        unsigned wei_d=tmp/(wei_h_size*wei_w_size);
        tmp-=wei_d*(wei_h_size*wei_w_size);
        unsigned wei_h=tmp/wei_w_size;
        unsigned wei_w=tmp-wei_h*wei_w_size;
        int im_d=(int)(out_d+wei_d);
        int im_h=(int)(out_h+wei_h);
        int im_w=(int)(out_w+wei_w);
        // Bounds provably hold: out + filter <= input - 1 for these shapes.
        short value=im[wei_c*(im_d_size*im_h_size*im_w_size)+
                       im_d*(im_h_size*im_w_size)+im_h*im_w_size+im_w];
        col[tid]=value;
    }
}

extern "C" int lowering(void* input, void* col, int channels, int depth,
                         int specialized, void* stream_pointer) {
    if((channels!=256 && channels!=512)||(depth!=3 && depth!=6))return -1;
    hipStream_t stream=(hipStream_t)stream_pointer;
    if(!specialized) {
        hipLaunchKernelGGL(Im3d2Col, dim3(1024), dim3(256), 0, stream,
            (short*)input,0u,(unsigned)channels,(unsigned)depth,130u,130u,
            3u,3u,3u,(unsigned)(depth-2),128u,128u,0u,0u,0u,
            1u,1u,1u,1u,1u,1u,(short*)col);
    } else if(channels==256 && depth==3) {
        hipLaunchKernelGGL((Im3d2ColFixed<256,3>),dim3(1024),dim3(256),0,stream,(short*)input,(short*)col);
    } else if(channels==256 && depth==6) {
        hipLaunchKernelGGL((Im3d2ColFixed<256,6>),dim3(1024),dim3(256),0,stream,(short*)input,(short*)col);
    } else if(channels==512 && depth==3) {
        hipLaunchKernelGGL((Im3d2ColFixed<512,3>),dim3(1024),dim3(256),0,stream,(short*)input,(short*)col);
    } else {
        hipLaunchKernelGGL((Im3d2ColFixed<512,6>),dim3(1024),dim3(256),0,stream,(short*)input,(short*)col);
    }
    return (int)hipGetLastError();
}
extern "C" int create_handle(void** pointer,void* stream,int atomics_allowed) {
    rocblas_handle handle=nullptr;
    rocblas_status status=rocblas_create_handle(&handle);
    if(status!=rocblas_status_success)return (int)status;
    status=rocblas_set_stream(handle,(hipStream_t)stream);
    if(status==rocblas_status_success)status=rocblas_set_pointer_mode(handle,rocblas_pointer_mode_host);
    if(status==rocblas_status_success)status=rocblas_set_atomics_mode(handle,
        atomics_allowed?rocblas_atomics_allowed:rocblas_atomics_not_allowed);
    if(status!=rocblas_status_success){rocblas_destroy_handle(handle);return (int)status;}
    *pointer=(void*)handle;
    return 0;
}
extern "C" int handle_policy(void* pointer,int* atomics,int* pointer_mode,void** stream) {
    rocblas_atomics_mode am;
    rocblas_pointer_mode pm;
    hipStream_t hs;
    rocblas_handle h=(rocblas_handle)pointer;
    auto s=rocblas_get_atomics_mode(h,&am);if(s!=rocblas_status_success)return (int)s;
    s=rocblas_get_pointer_mode(h,&pm);if(s!=rocblas_status_success)return (int)s;
    s=rocblas_get_stream(h,&hs);if(s!=rocblas_status_success)return (int)s;
    *atomics=(int)am;*pointer_mode=(int)pm;*stream=(void*)hs;return 0;
}
extern "C" int destroy_handle(void* pointer){return (int)rocblas_destroy_handle((rocblas_handle)pointer);}
extern "C" int gemm(void* pointer,void* col,void* weight,void* output,int channels,int depth) {
    if((channels!=256 && channels!=512)||(depth!=3 && depth!=6))return -1;
    int m=(depth-2)*128*128,n=256,k=channels*27;
    float alpha=1.0f,beta=0.0f;
    return (int)rocblas_gemm_ex((rocblas_handle)pointer,
        rocblas_operation_none,rocblas_operation_none,m,n,k,&alpha,
        col,rocblas_datatype_bf16_r,m,weight,rocblas_datatype_bf16_r,k,
        &beta,output,rocblas_datatype_bf16_r,m,output,rocblas_datatype_bf16_r,m,
        rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0);
}
