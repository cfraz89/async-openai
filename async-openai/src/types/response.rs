use std::fs::File;

use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use super::{
    FunctionObject, MessageStatus, RankingOptions, ReasoningEffort, ResponseFormat,
    VectorStoreSearchFilter,
};
use crate::types::OpenAIError;

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// How the model should select which tool (or tools) to use when generating
    /// a response. See the `tools` parameter to see how to specify which tools
    /// the model can call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// The truncation strategy to use for the model response.
    /// - `auto`: If the context of this response and previous ones exceeds
    /// the model's context window size, the model will truncate the
    /// response to fit the context window by dropping input items in the
    /// middle of the conversation.
    /// - `disabled` (default): If a model response will exceed the context
    /// window size for a model, the request will fail with a 400 error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<ResponseTruncation>,

    pub input: Input,
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
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Tool {
    FileSearch(FileSearchTool),
    Function(FunctionTool),
    ComputerUsePreview(ComputerTool),
    WebSearchPreview(WebSearchTool),
    #[serde(rename = "web_search_preview_2025_03_11")]
    WebSearchPreview2025_03_11(WebSearchTool),
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
    /// The IDs of the vector stores to search.
    pub vector_store_ids: Vec<String>,

    /// The maximum number of results to return. This number should be
    /// between 1 and 50 inclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_num_results: Option<u32>,

    /// A filter to apply based on file attributes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<VectorStoreSearchFilter>,

    /// Ranking options for search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranking_options: Option<RankingOptions>,
}

///  A tool that controls a virtual computer. Learn more about the
/// [computer tool](https://platform.openai.com/docs/guides/tools-computer-use).
#[derive(Clone, Serialize, Default, Debug, Builder, Deserialize, PartialEq)]
#[builder(name = "ComputerToolArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct ComputerTool {
    /// The width of the computer display.
    pub display_width: u32,

    /// The height of the computer display.
    pub display_height: u32,

    /// The type of computer environment to control.
    pub environment: Environment,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Environment {
    Mac,
    Windows,
    Ubuntu,
    Browser,
}

/// This tool searches the web for relevant results to use in a response.
/// Learn more about the [web search tool](https://platform.openai.com/docs/guides/tools-web-search).
#[derive(Clone, Serialize, Default, Debug, Builder, Deserialize, PartialEq)]
#[builder(name = "WebSearchToolArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct WebSearchTool {
    /// The IDs of the vector stores to search.
    pub user_location: Option<WebSearchToolUserLocation>,
}

/// Approximate location parameters for the search.
#[derive(Clone, Serialize, Default, Debug, Builder, Deserialize, PartialEq)]
#[builder(name = "WebSearchToolUserLocationArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct WebSearchToolUserLocation {
    /// The type of location approximation. Always `approximate`.
    pub r#type: LocationType,

    /// The two-letter [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1) of the user,
    /// e.g. `US`.
    pub country: Option<String>,

    /// Free text input for the region of the user, e.g. `California`.
    pub region: Option<String>,

    /// Free text input for the city of the user, e.g. `San Francisco`.
    pub city: Option<String>,

    /// The [IANA timezone](https://timeapi.io/documentation/iana-timezones)
    /// of the user, e.g. `America/Los_Angeles`.
    pub timezone: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LocationType {
    #[default]
    Approximate,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ToolChoice {
    Options(ToolChoiceOptions),
    Types(ToolChoiceTypes),
    Function(ToolChoiceFunction),
}

/// Controls which (if any) tool is called by the model.
/// `none` means the model will not call any tools and instead generates a message.
/// `auto` is the default value and means the model can pick between generating a message or calling one or more tools.
/// `required` means the model must call one or more tools before responding to the user.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceOptions {
    None,
    Auto,
    Required,
}

/// The type of hosted tool the model should to use. Learn more about
/// [built-in tools](https://platform.openai.com/docs/guides/tools).
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceTypes {
    FileSearch,
    ComputerUsePreview,
    WebSearchPreview,
    #[serde(rename = "web_search_preview_2025_03_11")]
    WebSearchPreview2025_03_11,
}

/// Use this option to force the model to call a specific function.
#[derive(Clone, Serialize, Default, Debug, Builder, Deserialize, PartialEq)]
#[builder(name = "ToolChoiceFunctionArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct ToolChoiceFunction {
    /// For function calling, the type is always `function`.
    #[builder(default = "FunctionToolType::Function")]
    pub r#type: FunctionToolType,

    /// The name of the function to call.
    pub name: String,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum FunctionToolType {
    #[default]
    Function,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ResponseTruncation {
    Auto,
    #[default]
    Disabled,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Input {
    /// A text input to the model, equivalent to a text input with
    /// the `user` role.
    String(String),
    /// A list of one or many input items to the model, containing
    /// different content types.
    Array(Vec<InputItem>),
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
#[serde(rename_all = "lowercase")]
pub enum InputItem {
    Message(EasyInputMessage),
    /// An item representing part of the context for the response to be
    /// generated by the model. Can contain text, images, and audio inputs,
    /// as well as previous assistant responses and tool call outputs.
    Item(Item),
    ItemReference(ItemReference),
}

/// A message input to the model with a role indicating instruction following
/// hierarchy. Instructions given with the `developer` or `system` role take
/// precedence over instructions given with the `user` role. Messages with the
/// `assistant` role are presumed to have been generated by the model in previous
/// interactions.
/// Use this option to force the model to call a specific function.
#[derive(Clone, Serialize, Default, Debug, Builder, Deserialize, PartialEq)]
#[builder(name = "EasyInputMessageArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct EasyInputMessage {
    /// The type of the message input. Always `message`.
    #[builder(default = "MessageType::Message")]
    pub r#type: MessageType,

    /// The role of the message input. One of `user`, `assistant`, `system`,
    /// or `developer`.
    pub role: EasyInputMessageRole,

    /// Text, image, or audio input to the model, used to generate a response.
    /// Can also contain previous assistant responses.
    pub content: InputMessageContent,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum MessageType {
    #[default]
    Message,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub enum EasyInputMessageRole {
    User,
    Assistant,
    System,
    Developer,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum InputMessageContent {
    Text(String),
    Array(Vec<InputContent>),
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum InputContent {
    InputText(InputText),
    InputImage(InputImage),
    InpuFile(InputFile),
}

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct InputText {
    /// The text input to the model.
    pub text: String,
}

/// An image input to the model. Learn about
/// [image inputs](https://platform.openai.com/docs/guides/vision).
#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct InputImage {
    /// The URL of the image to be sent to the model. A fully qualified URL or
    /// base64 encoded image in a data URL.
    pub image_url: Option<String>,

    /// The ID of the file to be sent to the model.
    pub file_id: Option<String>,

    /// The detail level of the image to be sent to the model. One of `high`,
    /// `low`, or `auto`. Defaults to `auto`.
    pub detail: ImageDetail,
}

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    High,
    Low,
    #[default]
    Auto,
}

/// An image input to the model. Learn about
/// [image inputs](https://platform.openai.com/docs/guides/vision).
#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct InputFile {
    /// The ID of the file to be sent to the model.
    pub file_id: Option<String>,

    /// The name of the file to be sent to the model.
    pub filename: Option<String>,

    /// The content of the file to be sent to the model.
    pub file_data: Option<String>,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Item {
    Message(InputMessage),
}

#[derive(Clone, Serialize, Default, Debug, Builder, Deserialize, PartialEq)]
#[builder(name = "InputMessageArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct InputMessage {
    /// The role of the message input. One of `user`, `system`, or `developer`.
    pub role: InputMessageRole,

    /// The status of item. One of `in_progress`, `completed`, or
    /// `incomplete`. Populated when items are returned via API.
    pub status: MessageStatus,

    /// A list of one or many input items to the model, containing different content
    /// types.
    pub content: Vec<InputContent>,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub enum InputMessageRole {
    User,
    System,
    Developer,
}

#[derive(Clone, Serialize, Default, Debug, Builder, Deserialize, PartialEq)]
#[builder(name = "OutputMessageArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct OutputMessage {
    /// The unique ID of the output message.
    pub id: String,

    /// The type of the output message. Always `message`.
    #[builder(default = "MessageType::Message")]
    pub r#type: MessageType,

    /// The role of the output message. Always `assistant`.
    #[builder(default = "InputMessageRole::Assistant")]
    pub role: InputMessageRole,

    /// The content of the output message.
    pub content: Vec<InputContent>,
}
