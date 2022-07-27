use elikoga_textsynth::{
    completions::{Engine, RequestBuilder},
    TextSynthClient,
};
use futures::StreamExt;

#[tokio::test]
async fn completions() {
    // get API Key from env
    let api_key = std::env::var("TEXT_SYNTH_API_KEY").expect("TEXT_SYNTH_API_KEY not set");
    let client = TextSynthClient::new(&api_key);
    let text = r"Ninety-nine bottles of beer on the wall,
ninety-nine bottles of beer.
Take one down, pass it around,
ninety-eight bottles of beer on the wall.
Ninety-eight bottles of beer on the wall,";
    let request = RequestBuilder::default()
        .prompt(text)
        .temperature(0.0)
        .stop(["Ninety-seven bottles of beer on the wall".into()])
        .stream(true)
        .build()
        .expect("failed to build completion request");
    for engine in [Engine::GPTJ6B, Engine::GPTNeoX20B] {
        println!("Engine: {}", engine);
        let mut response = client
            .completions(&engine, &request)
            .await
            .expect("failed to complete");
        let mut result = String::new();
        while let Some(completion) = response.next().await {
            // fill string with completion
            let completion = completion.expect("failed to get completion");
            let completion = completion.text[0].as_str();
            result.push_str(completion);
        }
        assert_eq!(
            result,
            r"
ninety-eight bottles of beer.
Take one down, pass it around,
ninety-seven bottles of beer on the wall.
"
        );
    }
}
