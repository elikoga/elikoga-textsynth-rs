use elikoga_textsynth::{
    completions::{logprob::RequestBuilder, Engine},
    TextSynthClient,
};

#[tokio::test]
async fn logprob() {
    // get API Key from env
    let api_key = std::env::var("TEXT_SYNTH_API_KEY").expect("TEXT_SYNTH_API_KEY not set");
    let client = TextSynthClient::new(&api_key);
    let text = "world!";
    let request = RequestBuilder::default()
        .context("Hello, ")
        .continuation(text)
        .build()
        .expect("logprob request should build");
    let response = client
        .logprob(&Engine::GPTJ6B, &request)
        .await
        .expect("logprob request should succeed");
    assert_eq!(response.logprob, -16.8283226862796);
    assert!(!response.is_greedy);
}
