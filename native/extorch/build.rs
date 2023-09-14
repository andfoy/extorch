use cxx_build::CFG;
use glob::glob;
use glob::GlobError;
use std::env;
use std::fs;
use std::hash::Hasher;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str;

use rustc_hash::FxHasher;
use tera::{Context, Tera};

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
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/csrc");
    println!("cargo:rerun-if-changed=src/native");
    println!("cargo:rerun-if-changed=src/nifs.rs");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let template_path = Path::new(&manifest_dir)
        .join("src")
        .join("native")
        .join("**")
        .join("*.rs.in");

    let tera = match Tera::new(template_path.to_str().unwrap()) {
        Ok(t) => t,
        Err(e) => {
            println!("Parsing error(s): {}", e);
            ::std::process::exit(1);
        }
    };

    let context = Context::new();
    let mut rendering = tera.render("native.rs.in", &context).unwrap();
    let syntax_tree = syn::parse_file(&rendering).unwrap();
    rendering = prettyplease::unparse(&syntax_tree);

    let native_hash = Path::new(&manifest_dir)
        .join("src")
        .join("native")
        .join("native.rs.sum");

    let hash = match fs::read_to_string(native_hash.clone()) {
        Ok(hash) => hash.parse::<u64>().unwrap(),
        Err(_) => 0,
    };

    let mut hasher = FxHasher::default();
    hasher.write(rendering.as_bytes());
    let cur_hash = hasher.finish();

    if hash != cur_hash {
        let ref path = Path::new(&manifest_dir).join("src").join("native.rs");
        fs::write(path, rendering).unwrap();
        fs::write(native_hash, cur_hash.to_string()).unwrap();
    }

    let mut link_gpu = false;

    let ref libtorch_path = Path::new(&manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("priv")
        .join("native")
        .join("libtorch");
    if libtorch_path.exists() {
        let torch_include_path = libtorch_path.join("include");
        let inner_torch_include_path = libtorch_path.join("include/torch/csrc/api/include");
        let torch_lib = libtorch_path.join("lib");

        let glob_pattern = torch_lib.join("*torch_cuda*").to_str().unwrap().to_string();
        let glob_all = glob(&glob_pattern);
        match glob_all {
            Ok(paths) => {
                let valid_paths: Vec<Result<PathBuf, GlobError>> =
                    paths.into_iter().filter(|x| x.is_ok()).collect();
                link_gpu = valid_paths.len() > 0;
            }
            Err(_) => {
                link_gpu = false;
            }
        }

        println!(
            "cargo:rustc-link-search=native={}",
            torch_lib.to_str().unwrap()
        );

        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            torch_lib.to_str().unwrap()
        );

        CFG.exported_header_dirs.push(&torch_include_path);
        CFG.exported_header_dirs.push(&inner_torch_include_path);
    } else {
        if command_ok(Command::new("python").arg("--version")) {
            let torch_location = command_output(
                Command::new("python").args(&["-c", "import torch; print(torch.__file__)"]),
            );
            let torch_path = Path::new(&torch_location).parent().unwrap();

            let torch_include_path = torch_path.join("include");
            let inner_torch_include_path = torch_path.join("include/torch/csrc/api/include");

            let torch_lib = torch_path.join("lib");

            let glob_pattern = torch_lib.join("*torch_cuda*").to_str().unwrap().to_string();
            let glob_all = glob(&glob_pattern);
            match glob_all {
                Ok(paths) => {
                    let valid_paths: Vec<Result<PathBuf, GlobError>> =
                        paths.into_iter().filter(|x| x.is_ok()).collect();
                    link_gpu = valid_paths.len() > 0;
                }
                Err(_) => {
                    link_gpu = false;
                }
            }

            println!(
                "cargo:rustc-link-search=native={}",
                torch_lib.to_str().unwrap()
            );

            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                torch_lib.to_str().unwrap()
            );

            CFG.exported_header_dirs.push(&torch_include_path);
            CFG.exported_header_dirs.push(&inner_torch_include_path);
        }
    }

    cxx_build::bridge("src/native.rs")
        .file("src/csrc/wrapper.cc")
        .file("src/csrc/utils.cc")
        .file("src/csrc/creation.cc")
        .file("src/csrc/manipulation.cc")
        .file("src/csrc/reduction.cc")
        .file("src/csrc/pointwise.cc")
        .file("src/csrc/comparison.cc")
        .file("src/csrc/other.cc")
        .file("src/csrc/printing.cc")
        .file("src/csrc/info.cc")
        .flag_if_supported("-std=c++17")
        // .flag_if_supported("-std=gnu++14")
        .define("_GLIBCXX_USE_CXX11_ABI", "0")
        .warnings(false)
        .extra_warnings(false)
        .compile("torch-wrapper");

    // TODO: See why Cargo is not linking against libtorch directly
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");

    if link_gpu {
        println!("cargo:rustc-link-lib=dylib=torch_cuda");
    }
}
