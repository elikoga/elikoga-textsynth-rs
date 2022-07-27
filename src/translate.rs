//! Provides translate api

use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;
use thiserror::Error;

use crate::{IsEngine, TextSynthClient};

/// Enum for the different translation engines available for TextSynth
#[derive(strum::Display)]
pub enum Engine {
    /// M2M100 1.2B is a 1.2 billion parameter language model specialized for
    /// translation. It supports multilingual translation between 100 languages.
    #[strum(serialize = "m2m100_1_2B")]
    M2M10012B,
}

impl IsEngine for Engine {
    fn is_translation(&self) -> bool {
        true
    }
}

/// Struct for a translation request
#[skip_serializing_none]
#[derive(Serialize, Builder)]
#[builder(setter(into))]
#[builder(build_fn(validate = "Self::validate"))]
pub struct Request {
    /// Each string is an independent text to translate. Batches of at most 64
    /// texts can be provided.
    text: Vec<String>,
    /// Two or three character ISO language code for the source language. The
    /// special value "auto" indicates to auto-detect the source language. The
    /// language auto-detection does not support all languages and is based on
    /// heuristics. Hence if you know the source language you should explicitly
    /// indicate it.
    source_lang: String,
    /// Two or three character ISO language code for the target language.
    target_lang: String,
    /// Number of beams used to generate the translated text. The translation is
    /// usually better with a larger number of beams. Each beam requires
    /// generating a separate translated text, hence the number of generated
    /// tokens is multiplied by the number of beams.
    #[builder(setter(strip_option))]
    #[builder(default)]
    num_beams: Option<u32>,
    /// The translation model only translates one sentence at a time. Hence the
    /// input must be split into sentences. When split_sentences = true
    /// (default), each input text is automatically split into sentences using
    /// source language specific heuristics. If you are sure that each input
    /// text contains only one sentence, it is better to disable the automatic
    /// sentence splitting.
    #[builder(setter(strip_option))]
    #[builder(default)]
    split_sentences: Option<bool>,
}

impl RequestBuilder {
    fn validate(&self) -> Result<(), String> {
        // text has length 1 to 64
        match &self.text {
            Some(text) if !(1..=64).contains(&text.len()) => {
                return Err("text has to have 1 to 64 elements".to_string());
            }
            _ => {}
        }
        // source_lang is 2 or 3 characters long or is "auto"
        match &self.source_lang {
            Some(source_lang)
                if !(source_lang.len() == 2 || source_lang.len() == 3 || source_lang == "auto") =>
            {
                return Err(
                    "source_lang has to be a 2 or 3 characters long iso language code or be \"auto\""
                        .to_string(),
                );
            }
            _ => {}
        }
        // target_lang is 2 or 3 characters long
        match &self.target_lang {
            Some(target_lang) if !(target_lang.len() == 2 || target_lang.len() == 3) => {
                return Err(
                    "target_lang has to be a 2 or 3 characters long iso language code".to_string(),
                );
            }
            _ => {}
        }
        // num_beams has range 1 to 5
        match self.num_beams {
            Some(Some(num_beams)) if !(1..=5).contains(&num_beams) => {
                return Err("num_beams has to be in the range 1 to 5".to_string());
            }
            _ => {}
        }
        Ok(())
    }
}

/// Struct for a translation answer
#[derive(Deserialize, Debug)]
pub struct Response {
    /// Array of translation objects.
    pub translations: Vec<Translation>,
    /// Indicate the total number of input tokens. It is useful to estimate the
    /// number of compute resources used by the request.
    pub input_tokens: u32,
    /// Indicate the total number of generated tokens. It is useful to estimate
    /// the number of compute resources used by the request.
    pub output_tokens: u32,
}

/// a single translation result
#[derive(Deserialize, Debug)]
pub struct Translation {
    /// translated text
    pub text: String,
    /// ISO language code corresponding to the detected lang (identical to
    /// source_lang if language auto-detection is not enabled)
    pub detected_source_lang: String,
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
}

impl TextSynthClient {
    /// Perform a completion request
    pub async fn translate(&self, engine: &Engine, request: &Request) -> Result<Response, Error> {
        let request_json = serde_json::to_string(&request)?;
        let url = format!("{}/engines/{}/translate", self.base_url, engine);
        let response = self.client.post(&url).body(request_json).send().await?;
        response.json().await.map_err(|e| e.into())
    }
}
