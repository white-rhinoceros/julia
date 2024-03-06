
extern crate libc;
extern crate rand;

use libc::{c_float, c_uchar, size_t};
use rand::Rng;

const SIZE: usize = 5_000;


extern "C" {
    fn cudaMalloc(ptr: &mut *mut c_float, len: size_t);

    fn dot(source: *mut c_float, dev: *mut c_float, count: size_t);
    fn dot2();

    fn dot__(dev: *mut c_float);
}

fn main() {
    let mut v: Vec<f32> = Vec::with_capacity(SIZE);

    let mut rng = rand::thread_rng();
    for _ in 0..SIZE {
        //v.push(rng.gen_range(0. ..1.));
        v.push(0.0);
    }

    let s = std::mem::size_of::<c_float>();
    println!("{}", s);

    // // Размер изображения
    // let bounds = (1000, 1000);
    // // Изображение в памяти компьютера
    // let mut pixels = vec![0, bounds.0 * bounds.1];
    // // Указатель на тип данных uchar в памяти GPU
    // let gpu_bitmap: *mut c_uchar;

    let mut gpu_float: *mut c_float = std::ptr::null_mut();
    unsafe {
        cudaMalloc(&mut gpu_float, s * SIZE);

        dot2();

        dot(v.as_mut_ptr(), gpu_float, SIZE);

        dot__(gpu_float);
    }
}