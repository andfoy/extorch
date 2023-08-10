extern crate rustler;

use rustler::resource::ResourceArc;
use rustler::NifStruct;
use rustler::Term;

use crate::native::torch;

pub struct Reference<'a> {
    pub reference: Option<Term<'a>>,
}

impl<'a> Reference<'a> {
    pub fn new() -> Self {
        Reference { reference: None }
    }
}

pub struct AtomString {
    pub name: String,
}

pub struct Size {
    pub size: Vec<i64>,
}

#[derive(NifStruct)]
#[module = "ExTorch.Tensor.Options"]
pub struct TensorOptions {
    pub dtype: AtomString,
    pub layout: AtomString,
    pub device: torch::Device,
    pub requires_grad: bool,
    pub pin_memory: bool,
    pub memory_format: AtomString,
}

#[derive(NifStruct)]
#[module = "ExTorch.Tensor"]
pub struct TensorStruct<'a> {
    pub resource: ResourceArc<torch::CrossTensorRef>,
    pub reference: Reference<'a>,
    pub size: Size,
    pub dtype: AtomString,
    pub device: torch::Device,
}

#[derive(NifStruct)]
#[module = "ExTorch.Utils.ListWrapper"]
pub struct ListWrapper<'a> {
    pub list: Vec<Term<'a>>,
    pub size: Term<'a>,
    pub dtype: Term<'a>,
}

pub struct TensorIndex {
    pub indices: Vec<torch::TorchIndex>
}

#[derive(NifStruct)]
#[module = "ExTorch.Utils.Indices.Slice"]
pub struct ExSlice {
    pub start: i64,
    pub stop: i64,
    pub step: i64,
    pub mask: u8
}
