extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .cpp(true)
        .flag("-cudart=shared")
        .flag("-allow-unsupported-compiler")
        .files(&[
            "./cuda/julia.cu",
            //"./cuda/sph.cu",
        ])
        .compile("library.a");

    println!("cargo:rustc-link-search=native=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3\\lib");
    println!("cargo:rustc-link-search=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3\\lib");
    println!("cargo:rustc-env=LD_LIBRARY_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3\\lib");
    println!("cargo:rustc-link-lib=dylib=cudart");
}