use elikoga_textsynth::{
    translate::{Engine, RequestBuilder},
    TextSynthClient,
};

#[tokio::test]
async fn translate() {
    // get API Key from env
    let api_key = std::env::var("TEXT_SYNTH_API_KEY").expect("TEXT_SYNTH_API_KEY not set");
    let client = TextSynthClient::new(&api_key);
    let text = "Hello, world!";
    let request = RequestBuilder::default()
        .text([text.into()])
        .source_lang("en")
        .target_lang("de")
        .num_beams(1_u32)
        .build()
        .expect("failed to build completion request");
    let response = client
        .translate(&Engine::M2M10012B, &request)
        .await
        .expect("Request should succeed");
    assert_eq!(response.translations[0].text, "Hallo Welt !");
}
