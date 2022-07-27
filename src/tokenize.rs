//! Provides tokenize api

use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;
use thiserror::Error;

use crate::{IsEngine, TextSynthClient};

/// Struct for a tokenize request
#[skip_serializing_none]
#[derive(Serialize, Builder)]
#[builder(setter(into))]
pub struct Request {
    /// Input text.
    text: String,
}

/// Struct for a tokenization answer
#[derive(Deserialize, Debug)]
pub struct Response {
    /// Token indexes corresponding to the input text.
    pub tokens: Vec<u32>,
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
    /// Perform a tokenization request
    pub async fn tokenize(
        &self,
        engine: &impl IsEngine,
        request: &Request,
    ) -> Result<Response, Error> {
        let request_json = serde_json::to_string(&request)?;
        let url = format!("{}/engines/{}/tokenize", self.base_url, engine);
        let response = self.client.post(&url).body(request_json).send().await?;
        response.json().await.map_err(|e| e.into())
    }
}
