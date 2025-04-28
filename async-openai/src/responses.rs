/// OpenAI's most advanced interface for generating model responses. Supports text and image inputs, and text outputs. Create stateful interactions with the model, using the output of previous responses as input. Extend the model's capabilities with built-in tools for file search, web search, computer use, and more. Allow the model access to external systems and data using function calling.
///
/// Related guide: [Responses](https://platform.openai.com/docs/guides/text)
pub struct Responses<'c, C: Config> {
    client: &'c Client<C>,
}

impl<'c, C: Config> Responses<'c, C> {
    pub fn new(client: &'c Client<C>) -> Self {
        Self { client }
    }

    /// Creates a model response. Provide [text](https://platform.openai.com/docs/guides/text) or
    /// [image](https://platform.openai.com/docs/guides/images) inputs to generate
    /// [text](https://platform.openai.com/docs/guides/text)
    /// or [JSON](https://platform.openai.com/docs/guides/structured-outputs) outputs. Have the model call
    /// your own [custom code](https://platform.openai.com/docs/guides/function-calling) or use built-in
    /// [tools](https://platform.openai.com/docs/guides/tools) like [web search](https://platform.openai.com/docs/guides/tools-web-search)
    /// or [file search](https://platform.openai.com/docs/guides/tools-file-search) to use your own data
    /// as input for the model's response.
    ///
    /// byot: You must ensure "stream: false" in serialized `request`
    #[crate::byot(
        T0 = serde::Serialize,
        R = serde::de::DeserializeOwned
    )]
    pub async fn create(&self, request: CreateResponse) -> Result<Response, OpenAIError> {
        #[cfg(not(feature = "byot"))]
        {
            if request.stream.is_some() && request.stream.unwrap() {
                return Err(OpenAIError::InvalidArgument(
                    "When stream is true, use Responses::create_stream".into(),
                ));
            }
        }
        self.client.post("/responses", request).await
    }
}
