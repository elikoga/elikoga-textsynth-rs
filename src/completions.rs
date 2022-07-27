//! Provides completion api

pub mod logprob;

use std::{collections::HashMap, fmt, marker::PhantomData};

use bytes::{Buf, BytesMut};
use futures::{stream, Stream, StreamExt};
use serde::{de, Deserialize, Deserializer, Serialize};
use serde_with::skip_serializing_none;
use thiserror::Error;

use crate::{IsEngine, TextSynthClient};

/// Enum for the different completion engines available for TextSynth
#[derive(strum::Display)]
pub enum Engine {
    /// GPT-J is a language model with 6 billion parameters trained on the Pile
    /// (825 GB of text data) published by EleutherAI. Its main language is
    /// English but it is also fluent in several other languages. It is also
    /// trained on several computer languages.
    #[strum(serialize = "gptj_6B")]
    GPTJ6B,
    /// Boris is a fine tuned version of GPT-J for the French language. Use this
    /// model is you want the best performance with the French language.
    #[strum(serialize = "boris_6B")]
    Boris6B,
    /// Fairseq GPT 13B is an English language model with 13 billion parameters.
    /// Its training corpus is less diverse than GPT-J but it has better
    /// performance at least on pure English language tasks.
    #[strum(serialize = "fairseq_gpt_13B")]
    FairseqGPT13B,
    /// GPT-NeoX-20B is the largest publically available English language model
    /// with 20 billion parameters. It was trained on the same corpus as GPT-J.
    #[strum(serialize = "gptneox_20B")]
    GPTNeoX20B,
}

impl IsEngine for Engine {
    fn is_completion(&self) -> bool {
        true
    }
}

/// Struct for a completion request
#[skip_serializing_none]
#[derive(Serialize, Builder)]
#[builder(setter(into))]
#[builder(build_fn(validate = "Self::validate"))]
pub struct Request {
    /// The input text to complete.
    ///
    /// NOTE: The prompt is not included in the output.
    prompt: String,
    /// Maximum number of tokens to generate. A token represents about 4
    /// characters for English texts. The total number of tokens (prompt +
    /// generated text) cannot exceed the model's maximum context length. It
    /// is of 2048 for GPT-J and 1024 for the other models.
    #[builder(setter(strip_option))]
    #[builder(default)]
    max_tokens: Option<u32>,
    /// If true, the output is streamed so that it is possible to display the
    /// result before the complete output is generated. Several JSON answers
    /// are output. Each answer is followed by two line feed characters.
    #[builder(setter(strip_option))]
    #[builder(default)]
    stream: Option<bool>,
    /// Stop the generation when the string(s) are encountered. The generated
    /// text does not contain the string. The length of the array is at most 5.
    #[builder(setter(strip_option))]
    #[builder(default)]
    stop: Option<Vec<String>>,
    /// Generate n completions from a single prompt.
    #[builder(setter(strip_option))]
    #[builder(default)]
    n: Option<u32>,
    /// Sampling temperature. A higher temperature means the model will select
    /// less common tokens leading to a larger diversity but potentially less
    /// relevant output. It is usually better to tune top_p or top_k.
    #[builder(setter(strip_option))]
    #[builder(default)]
    temperature: Option<f64>,
    /// Select the next output token among the top_k most likely ones. A higher
    /// top_k gives more diversity but a potentially less relevant output.
    #[builder(setter(strip_option))]
    #[builder(default)]
    top_k: Option<u32>,
    /// Select the next output token among the most probable ones so that their
    /// cumulative probability is larger than top_p. A higher top_p gives more
    /// diversity but a potentially less relevant output. top_p and top_k are
    /// combined, meaning that at most top_k tokens are selected. A value of 1
    /// disables this sampling.
    #[builder(setter(strip_option))]
    #[builder(default)]
    top_p: Option<f64>,
    // More advanced sampling parameters are available:
    /// Modify the likelihood of the specified tokens in the completion.
    /// The specified object is a map between the token indexes and the
    /// corresponding logit bias. A negative bias reduces the likelihood of the
    /// corresponding token. The bias must be between -100 and 100. Note that
    /// the token indexes are specific to the selected model. You can use the
    /// tokenize API endpoint to retrieve the token indexes of a given model.
    /// Example: if you want to ban the " unicorn" token for GPT-J, you can use:
    /// logit_bias: { "44986": -100 }
    #[builder(setter(strip_option))]
    #[builder(default)]
    logit_bias: Option<HashMap<String, f64>>,
    /// A positive value penalizes tokens which already appeared in the
    /// generated text. Hence it forces the model to have a more diverse output.
    #[builder(setter(strip_option))]
    #[builder(default)]
    presence_penalty: Option<f64>,
    /// A positive value penalizes tokens which already appeared in the
    /// generated text proportionaly to their frequency. Hence it forces the
    /// model to have a more diverse output.
    #[builder(setter(strip_option))]
    #[builder(default)]
    frequency_penalty: Option<f64>,
    /// Divide by repetition_penalty the logits corresponding to tokens which
    /// already appeared in the generated text. A value of 1 effectively
    /// disables it.
    #[builder(setter(strip_option))]
    #[builder(default)]
    repetition_penalty: Option<f64>,
    /// Alternative to top_p sampling: instead of selecting the tokens starting
    /// from the most probable one, start from the ones whose log likelihood is
    /// the closest to the symbol entropy. This is useful for models with a
    /// low top_p value.
    /// The value of 1 disables this sampling.
    #[builder(setter(strip_option))]
    #[builder(default)]
    typical_p: Option<f64>,
}

impl RequestBuilder {
    fn validate(&self) -> Result<(), String> {
        // n must be between 1 and 16
        match self.n {
            Some(Some(n)) if !(1..=16).contains(&n) => {
                return Err("n must be between 1 and 16".to_string());
            }
            _ => {}
        };
        // top_k must be between 1 and 1000
        match self.top_k {
            Some(Some(top_k)) if !(1..=1000).contains(&top_k) => {
                return Err("top_k must be between 1 and 1000".to_string());
            }
            _ => {}
        };
        // top_p must be between 0.0 and 1.0
        match self.top_p {
            Some(Some(top_p)) if !(0.0..=1.0).contains(&top_p) => {
                return Err("top_p must be between 0.0 and 1.0".to_string());
            }
            _ => {}
        };
        // presence_penalty must be between -2.0 and 2.0
        match self.presence_penalty {
            Some(Some(presence_penalty)) if !(-2.0..=2.0).contains(&presence_penalty) => {
                return Err("presence_penalty must be between -2.0 and 2.0".to_string());
            }
            _ => {}
        };
        // frequency_penalty must be between -2.0 and 2.0
        match self.frequency_penalty {
            Some(Some(frequency_penalty)) if !(-2.0..=2.0).contains(&frequency_penalty) => {
                return Err("frequency_penalty must be between -2.0 and 2.0".to_string());
            }
            _ => {}
        };
        // typical_p: must be > 0 and <= 1\
        match self.typical_p {
            Some(Some(typical_p)) if !(typical_p > 0.0 && typical_p <= 1.0) => {
                return Err("typical_p must be between 0.0 and 1.0".to_string());
            }
            _ => {}
        };
        Ok(())
    }
}

fn string_or_seq_string<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringOrVec(PhantomData<Vec<String>>);

    impl<'de> de::Visitor<'de> for StringOrVec {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("string or list of strings")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(vec![value.to_owned()])
        }

        fn visit_seq<S>(self, visitor: S) -> Result<Self::Value, S::Error>
        where
            S: de::SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(visitor))
        }
    }

    deserializer.deserialize_any(StringOrVec(PhantomData))
}

/// Struct for a completion answer
#[derive(Deserialize, Debug)]
pub struct ResponseChunk {
    /// The completed text.
    #[serde(deserialize_with = "string_or_seq_string")]
    pub text: Vec<String>,
    /// If true, indicate that it is the last answer.
    pub reached_end: bool,
    /// If true, indicate that the prompt was truncated because it was too large
    pub truncated_prompt: Option<bool>,
    /// Indicate the number of input tokens.
    pub input_tokens: Option<u32>,
    /// Indicate the total number of generated tokens.
    pub output_tokens: Option<u32>,
}

#[derive(Error, Debug)]
/// Error for a completion answer
pub enum Error {
    /// Serde error
    #[error("Serde error: {0}")]
    SerdeError(#[from] serde_json::Error),
    /// Error from Reqwest
    #[error("Reqwest error: {0}")]
    RequestError(#[from] reqwest::Error),
    /// Couldn't parse the response to completion
    #[error("Couldn't parse the response to completion")]
    ParseError(bytes::Bytes),
}

impl TextSynthClient {
    /// Perform a completion request
    pub async fn completions(
        &self,
        engine: &Engine,
        request: &Request,
    ) -> Result<impl Stream<Item = Result<ResponseChunk, Error>>, Error> {
        let request_json = serde_json::to_string(&request)?;
        let url = format!("{}/engines/{}/completions", self.base_url, engine);
        let response = self.client.post(&url).body(request_json).send().await?;

        struct StreamState<S> {
            inner: S,
            chunks: BytesMut,
        }
        let state = StreamState {
            inner: response.bytes_stream(),
            chunks: BytesMut::new(),
        };
        let response_stream = stream::unfold(state, |mut state| async move {
            loop {
                if let Some(chunk) = state.inner.next().await {
                    let chunk = match chunk {
                        Ok(chunk) => chunk,
                        Err(err) => break Some((Err(err.into()), state)),
                    };
                    state.chunks.extend_from_slice(&chunk);
                    // stream parse
                    let mut stream = serde_json::Deserializer::from_slice(&state.chunks)
                        .into_iter::<ResponseChunk>();
                    // get next chunk
                    let next = Iterator::next(&mut stream);
                    // println!("Next: {:?}", next);
                    if let Some(Ok(chunk)) = next {
                        // remove parsed chunk from buffer
                        state.chunks.advance(stream.byte_offset());
                        // remove leading whitespace from buffer
                        let mut i = 0;
                        while i < state.chunks.len() {
                            if state.chunks[i].is_ascii_whitespace() {
                                i += 1;
                            } else {
                                break;
                            }
                        }
                        state.chunks.advance(i);
                        break Some((Ok(chunk), state));
                    }
                } else {
                    // end of stream
                    // if there is some data in the buffer (that isn't whitespace), return error
                    if state.chunks.is_empty() {
                        break None;
                    } else {
                        // return error
                        break Some((
                            Err(Error::ParseError(state.chunks.freeze())),
                            StreamState {
                                chunks: BytesMut::new(),
                                ..state
                            },
                        ));
                    }
                }
            }
        });
        Ok(Box::pin(response_stream))
    }
}
