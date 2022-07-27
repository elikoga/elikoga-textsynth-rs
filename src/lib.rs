#![warn(missing_docs)]
//! TextSynth API Crate

pub mod completions;
pub mod tokenize;
pub mod translate;

#[macro_use]
extern crate derive_builder;

use std::fmt::Display;

use reqwest::Client;

/// Engine trait,
pub trait IsEngine: Display {
    /// Returns wether it is a completion engine or not.
    fn is_completion(&self) -> bool {
        false
    }
    /// Returns wether it is a translation engine or not.
    fn is_translation(&self) -> bool {
        false
    }
}

/// TextSynth API Client
pub struct TextSynthClient {
    /// endpoint of TextSynth API
    base_url: String,
    /// Client for making requests to the TextSynth API
    client: Client,
}

impl TextSynthClient {
    /// Create a new TextSynth API Client with a custom endpoint
    pub fn new_with_endpoint(api_key: &str, endpoint: &str) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap(),
        );
        let reqwest_client = Client::builder().default_headers(headers);
        TextSynthClient {
            base_url: endpoint.to_string(),
            client: reqwest_client.build().unwrap(),
        }
    }

    /// Create a new TextSynth API Client
    pub fn new(api_key: &str) -> Self {
        Self::new_with_endpoint(api_key, "https://api.textsynth.com/v1")
    }
}
