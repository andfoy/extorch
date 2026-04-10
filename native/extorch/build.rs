use cxx_build::CFG;
use glob::glob;
use glob::GlobError;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::hash::Hasher;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str;

use rustc_hash::FxHasher;
use tera::{Context, Tera};

/// Extract function names from the rendered cxx bridge Rust source.
/// Matches lines like `fn foo_bar(` inside the `unsafe extern "C++"` block.
fn extract_bridge_fn_names(rendered_rs: &str) -> HashSet<String> {
    let mut names = HashSet::new();
    let mut in_extern_cpp = false;

    for line in rendered_rs.lines() {
        let trimmed = line.trim();
        if trimmed.contains("unsafe extern \"C++\"") {
            in_extern_cpp = true;
            continue;
        }
        if in_extern_cpp {
            // Track brace depth is overkill; the block ends at the module close.
            // Just look for `fn name(` patterns.
            if let Some(rest) = trimmed.strip_prefix("fn ") {
                if let Some(paren_pos) = rest.find('(') {
                    let name = rest[..paren_pos].trim().to_string();
                    names.insert(name);
                }
            }
        }
    }
    names
}

/// Extract function names declared in C++ header files.
/// Matches lines where a function name is followed by `(` and the line
/// doesn't start with common non-function patterns.
fn extract_header_fn_names(include_dir: &Path) -> HashSet<String> {
    let mut names = HashSet::new();
    let pattern = include_dir.join("*.h").to_str().unwrap().to_string();

    for entry in glob(&pattern).unwrap().flatten() {
        let content = fs::read_to_string(&entry).unwrap_or_default();
        for line in content.lines() {
            let trimmed = line.trim();
            // Skip preprocessor, includes, structs, using, extern, type aliases, comments
            if trimmed.starts_with('#')
                || trimmed.starts_with("//")
                || trimmed.starts_with("struct ")
                || trimmed.starts_with("using ")
                || trimmed.starts_with("extern ")
                || trimmed.starts_with("typedef ")
                || trimmed.starts_with("template")
                || trimmed.is_empty()
            {
                continue;
            }
            // Look for: `return_type name(` or `return_type name (` patterns
            // Function declarations have a `(` and a return type before the name
            if let Some(paren_pos) = trimmed.find('(') {
                let before_paren = trimmed[..paren_pos].trim();
                // The function name is the last word before `(`
                if let Some(name) = before_paren.split_whitespace().last() {
                    // Filter out obvious non-functions
                    let name = name.trim_start_matches('*').trim_start_matches('&');
                    if !name.is_empty()
                        && !name.contains("::")
                        && !name.contains('<')
                        && !name.contains('>')
                        && !name.starts_with('{')
                    {
                        names.insert(name.to_string());
                    }
                }
            }
        }
    }
    names
}

/// Check that every function declared in the cxx bridge has a matching
/// declaration in the C++ headers, and vice versa. Emits cargo warnings
/// for any mismatches.
fn validate_bridge_header_sync(rendered_rs: &str, include_dir: &Path) {
    let bridge_fns = extract_bridge_fn_names(rendered_rs);
    let header_fns = extract_header_fn_names(include_dir);

    // Functions in bridge but missing from headers
    let mut missing_from_headers: Vec<&str> = bridge_fns
        .iter()
        .filter(|name| !header_fns.contains(name.as_str()))
        .map(|s| s.as_str())
        .collect();
    missing_from_headers.sort();

    // Only check bridge → header direction.
    // Headers legitimately contain internal helpers not exposed through the bridge.
    for name in &missing_from_headers {
        println!(
            "cargo:warning=Bridge/header mismatch: '{}' declared in .rs.in bridge but NOT found in include/*.h",
            name
        );
    }

    if missing_from_headers.is_empty() {
        let matched = bridge_fns.intersection(&header_fns).count();
        println!(
            "cargo:warning=Bridge/header sync OK: all {} bridge functions found in headers ({} header-only helpers skipped)",
            matched,
            header_fns.len() - matched
        );
    }
}

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
        fs::write(path, &rendering).unwrap();
        fs::write(native_hash, cur_hash.to_string()).unwrap();
    }

    // Validate that bridge declarations and C++ headers are in sync
    let include_dir = Path::new(&manifest_dir).join("include");
    validate_bridge_header_sync(&rendering, &include_dir);

    let mut link_gpu = false;
    // Hoisted so the link-directives block below can check for individual
    // CUDA library presence (torch_cuda is required, c10_cuda and
    // torch_cuda_linalg may or may not exist depending on the build).
    let mut torch_lib_path: Option<PathBuf> = None;

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
        torch_lib_path = Some(torch_lib);
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
            torch_lib_path = Some(torch_lib);
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
        .file("src/csrc/jit.cc")
        .file("src/csrc/nn.cc")
        .file("src/csrc/aoti.cc")
        .flag_if_supported("-std=c++17")
        // .flag_if_supported("-std=gnu++14")
        .define("_GLIBCXX_USE_CXX11_ABI", "1")
        .warnings(false)
        .extra_warnings(false)
        .compile("torch-wrapper");

    // TODO: See why Cargo is not linking against libtorch directly
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");

    if link_gpu {
        // libtorch_cuda.so — CUDA aten ops (at::cuda::*, most conv/bn CUDA kernels)
        println!("cargo:rustc-link-lib=dylib=torch_cuda");

        // Extra CUDA support libs — only link them when the .so actually
        // exists in the libtorch lib directory. Older or stripped CUDA
        // builds may omit one or both.
        if let Some(ref lib_dir) = torch_lib_path {
            // libc10_cuda.so — low-level CUDA device / stream / allocator
            // utilities. Required for symbols declared in
            // <c10/cuda/CUDAFunctions.h> (e.g. c10::cuda::device_synchronize).
            if lib_dir.join("libc10_cuda.so").exists() {
                println!("cargo:rustc-link-lib=dylib=c10_cuda");
            }
            // libtorch_cuda_linalg.so — CUDA linear algebra kernels.
            // Optional; not present on every CUDA build.
            if lib_dir.join("libtorch_cuda_linalg.so").exists() {
                println!("cargo:rustc-link-lib=dylib=torch_cuda_linalg");
            }
        }
    }
}
