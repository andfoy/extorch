fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("src/wrapper.cc")
        .flag_if_supported("-std=c++14")
        .compile("torch-wrapper");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/wrapper.cc");
    println!("cargo:rerun-if-changed=include/wrapper.h");
}