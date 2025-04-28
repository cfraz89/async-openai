use std::fs::File;

use super::{FunctionObject, ReasoningEffort};

#[derive(Clone, Serialize, Default, Debug, Builder, Deserialize, PartialEq)]
#[builder(name = "CreateResponseArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct CreateResponse {
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
    /// while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or `top_p` but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>, // min: 0, max: 2, default: 1,

    /// An alternative to sampling with temperature, called nucleus sampling,
    /// where the model considers the results of the tokens with top_p probability mass.
    /// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    ///
    ///  We generally recommend altering this or `temperature` but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>, // min: 0, max: 1, default: 1

    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// The unique ID of the previous response to the model. Use this to
    /// create multi-turn conversations. Learn more about [conversation state](https://platform.openai.com/docs/guides/conversation-state).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    /// ID of the model to use.
    /// See the [model endpoint compatibility](https://platform.openai.com/docs/models#model-endpoint-compatibility) table for details on which models work with the Chat API.
    pub model: String,

    /// **o-series models only**
    /// Configuration options for
    /// [reasoning models](https://platform.openai.com/docs/guides/reasoning).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,

    /// An upper bound for the number of tokens that can be generated for a
    /// response, including visible output tokens and
    /// [reasoning tokens](https://platform.openai.com/docs/guides/reasoning).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    /// Inserts a system (or developer) message as the first item in the model's context.
    /// When using along with `previous_response_id`, the instructions from a previous
    /// response will be not be carried over to the next response. This makes it simple
    /// to swap out system (or developer) messages in new responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Configuration options for a text response from the model. Can be plain
    /// text or structured JSON data. Learn more:
    /// - [Text inputs and outputs](/docs/guides/text)
    /// - [Structured Outputs](/docs/guides/structured-outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextResponseFormat>,

    /// An array of tools the model may call while generating a response.
    /// You can specify which tool to use by setting the `tool_choice` parameter.
    /// The two categories of tools you can provide the model are:
    /// - **Built-in tools**: Tools that are provided by OpenAI that extend the
    /// model's capabilities, like [web search](https://platform.openai.com/docs/guides/tools-web-search)
    /// or [file search](https://platform.openai.com/docs/guides/tools-file-search). Learn more about
    /// [built-in tools](https://platform.openai.com/docs/guides/tools).
    /// - **Function calls (custom tools)**: Functions that are defined by you,
    /// enabling the model to call your own code. Learn more about
    /// [function calling](https://platform.openai.com/docs/guides/function-calling).
    pub tools: Vec<Tool>,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct Reasoning {
    /// **o-series models only**
    /// Constrains effort on reasoning for
    /// [reasoning models](https://platform.openai.com/docs/guides/reasoning).
    /// Currently supported values are `low`, `medium`, and `high`. Reducing
    /// reasoning effort can result in faster responses and fewer tokens used
    /// on reasoning in a response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,

    ///**computer_use_preview only**
    /// A summary of the reasoning performed by the model. This can be
    /// useful for debugging and understanding the model's reasoning
    /// process.
    /// One of `concise` or `detailed`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate_summary: Option<GenerateSummary>,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum GenerateSummary {
    Concise,
    Detailed,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct TextResponseFormat {
    format: ResponseFormat,
}

// Align the existing type with name in OpenAPI spec
pub type FunctionTool = FunctionObject;

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Tool {
    FileSearch(FileSearchTool),
    Function(FunctionTool),
    Computer(ComputerTool),
    WebSearch(WebSearchTool),
}

/// A tool that searches for relevant content from uploaded files.
/// Learn more about the [file search tool](https://platform.openai.com/docs/guides/tools-file-search).
#[derive(Clone, Serialize, Default, Debug, Builder, Deserialize, PartialEq)]
#[builder(name = "FileSearchToolArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct FileSearchTool {
    /// The type of the file search tool. Always `file_search`.
    #[builder(default = "ToolType::FileSearch")]
    pub r#type: ToolType,

    /// The IDs of the vector stores to search.
    pub vector_store_ids: Vec<String>,

    /// The maximum number of results to return. This number should be
    /// between 1 and 50 inclusive.
    pub max_num_results: Option<u32>,
}

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    /// The type of the file search tool.
    FileSearch,
    /// The type of the function tool.
    Function,
    /// The type of the computer use tool.
    ComputerUsePreview,
    /// The type of the web search tool.
    WebSearch,
}
