use elikoga_textsynth::{completions::Engine, tokenize::RequestBuilder, TextSynthClient};

#[tokio::test]
async fn tokenize() {
    // get API Key from env
    let api_key = std::env::var("TEXT_SYNTH_API_KEY").expect("TEXT_SYNTH_API_KEY not set");
    let client = TextSynthClient::new(&api_key);
    let text = "The quick brown fox jumps over the lazy dog";
    let request = RequestBuilder::default()
        .text(text)
        .build()
        .expect("logprob request should build");
    let response = client
        .tokenize(&Engine::GPTJ6B, &request)
        .await
        .expect("logprob request should succeed");
    assert_eq!(
        response.tokens,
        [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290]
    );
}
