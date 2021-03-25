use std::path::Path;
use std::process::Command;
use std::str;
use cxx_build::CFG;
// use tera::Tera;


fn command_ok(cmd: &mut Command) -> bool {
    cmd.status().ok().map_or(false, |s| s.success())
}

fn command_output(cmd: &mut Command) -> String {
    str::from_utf8(&cmd.output().unwrap().stdout)
        .unwrap()
        .trim()
        .to_string()
}

fn main() {
    // let mut torch_include_path = env::current_dir().unwrap();
    // let mut inner_torch_include_path = env::current_dir().unwrap();
    // let mut torch_lib = env::current_dir().unwrap();

    if command_ok(Command::new("python").arg("--version")) {
        let torch_location =
            command_output(Command::new("python").args(&["-c", "import torch; print(torch.__file__)"]));
        let torch_path = Path::new(&torch_location).parent().unwrap();

        let torch_include_path = torch_path.join("include");
        let inner_torch_include_path = torch_path.join("include/torch/csrc/api/include");

        let torch_lib = torch_path.join("lib");
        println!(
            "cargo:rustc-link-search=native={}",
            torch_lib.to_str().unwrap()
        );

        CFG.exported_header_dirs.push(&torch_include_path);
        CFG.exported_header_dirs.push(&inner_torch_include_path);

    }

    cxx_build::bridge("src/native.rs")
        .file("src/csrc/wrapper.cc")
        .file("src/csrc/utils.cc")
        .file("src/csrc/creation.cc")
        .file("src/csrc/info.cc")
        // .flag_if_supported("-std=c++17")
        .flag_if_supported("-std=gnu++14")
        .define("_GLIBCXX_USE_CXX11_ABI", "0")
        .warnings(false)
        .extra_warnings(false)
        .compile("torch-wrapper");

    // TODO: See why Cargo is not linking against libtorch directly
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/native.rs");
    println!("cargo:rerun-if-changed=src/dtypes.rs");
    println!("cargo:rerun-if-changed=src/conversion.rs");
    println!("cargo:rerun-if-changed=src/conversion/scalar_types.rs");
    println!("cargo:rerun-if-changed=src/wrapper.cc");
    println!("cargo:rerun-if-changed=include/wrapper.h");
}
