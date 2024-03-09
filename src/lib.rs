#![allow(dead_code)]

// Что-бы в модулях был доступен этот крейт.
extern crate libc;

use libc::{c_int, c_uchar, size_t};

#[repr(C)]
enum CudaMemcpyKind {
    CudaMemcpyHostToHost          =   0, // Host   -> Host
    CudaMemcpyHostToDevice        =   1, // Host   -> Device
    CudaMemcpyDeviceToHost        =   2, // Device -> Host
    CudaMemcpyDeviceToDevice      =   3, // Device -> Device
    CudaMemcpyDefault             =   4  // Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing.
}

extern "C" {
    fn cudaMalloc(ptr: &mut *mut c_uchar, count: size_t);
    fn cudaFree(ptr: *mut c_uchar);

    fn julia(pixels: *mut c_uchar, bound_x: size_t, bound_y: size_t);

    fn cudaMemcpy(dst: *mut c_uchar, src: *const c_uchar, count: size_t, kind: c_int);
}

pub fn gen_julia(pixels: &mut[u8], bounds: (usize, usize, usize)) {

    // Выделим память в GPU.
    let mut gpu_pixels: *mut c_uchar = std::ptr::null_mut();

    unsafe {
        cudaMalloc(&mut gpu_pixels, bounds.0 * bounds.1 * bounds.2);
    }

    // Вызов кода на GPU.
    unsafe {
        julia(gpu_pixels, bounds.0, bounds.1);
    }

    // Скопировать код из памяти GPU в ОЗУ.
    unsafe {
        cudaMemcpy(
            pixels.as_mut_ptr(),
            gpu_pixels,
            bounds.0 * bounds.1 * bounds.2,
            2
            //CudaMemcpyKind::CudaMemcpyDeviceToHost as i32
        );
    }

    unsafe {
        cudaFree(gpu_pixels);
    }
}