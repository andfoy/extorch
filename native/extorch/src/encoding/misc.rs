use rustler::{Decoder, Encoder, Env, NifResult, Term};

use rustler_sys::enif_make_ref;

use crate::shared_types::Reference;

impl<'a> Decoder<'a> for Reference<'a> {
    fn decode(term: Term<'a>) -> NifResult<Self> {
        Ok(Reference {
            reference: Some(term),
        })
    }
}

impl<'a> Encoder for Reference<'a> {
    fn encode<'b>(&self, env: Env<'b>) -> Term<'b> {
        match self.reference {
            None => unsafe { Term::new(env, enif_make_ref(env.as_c_arg())) },
            Some(r) => r.encode(env),
        }
    }
}
