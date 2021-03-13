
use lazy_static::lazy_static;
use std::collections::HashMap;


lazy_static! {
    pub static ref ALL_TYPES: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert("uint8", "uint8");
        m.insert("int8", "int8");
        m.insert("int16", "int16");
        m.insert("int32", "int32");
        m.insert("int64", "int64");
        m.insert("float16", "float16");
        m.insert("float32", "float32");
        m.insert("float64", "float64");
        m.insert("bfloat16", "bfloat16");
        m.insert("byte", "uint8");
        m.insert("char", "int8");
        m.insert("short", "int16");
        m.insert("int", "int32");
        m.insert("long", "int64");
        m.insert("half", "float16");
        m.insert("float", "float32");
        m.insert("double", "float64");
        m.insert("bool", "bool");
        m
    };
}
