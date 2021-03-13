
extern crate rustler;
use rustler::{Term};
use rustler::resource::ResourceArc;

use crate::native::{torch};


pub struct TensorOptions {
    pub dtype: String,
    pub layout: String,
    pub device: torch::Device,
    pub requires_grad: bool,
    pub pin_memory: bool,
    pub memory_format: String,
}

#[derive(NifStruct)]
#[module = "ExTorch.Tensor"]
pub struct TensorStruct<'a> {
    pub resource: ResourceArc<torch::CrossTensorRef>,
    pub reference: Term<'a>,
    pub size: Term<'a>,
    pub dtype: Term<'a>,
    pub device: Term<'a>,
}

#[derive(NifStruct)]
#[module = "ExTorch.Utils.ListWrapper"]
pub struct ListWrapper<'a> {
    pub list: Vec<Term<'a>>,
    pub size: Term<'a>,
    pub dtype: Term<'a>
}
