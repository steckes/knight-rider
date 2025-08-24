use reqwest::Client;
use serde_json::{json, Value};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlamaError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    #[error("HTTP error {0}: {1}")]
    Http(u16, String),
    #[error("Invalid response format")]
    InvalidResponse,
    #[error("LlamaServer Health Check Failed")]
    HealthCheckFailed,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "role": self.role,
            "content": self.content
        })
    }
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub temperature: f32,
    pub max_tokens: u32,
    pub stream: bool,
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            temperature: 0.7,
            max_tokens: 1000,
            stream: false,
        }
    }
}

impl ChatRequest {
    fn to_json(&self) -> Value {
        json!({
            "model": "llama",
            "messages": self.messages.iter().map(|m| m.to_json()).collect::<Vec<_>>(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream
        })
    }
}

#[derive(Debug)]
pub struct LlamaClient {
    client: Client,
    base_url: String,
}

impl LlamaClient {
    pub fn builder() -> LlamaClientBuilder {
        LlamaClientBuilder::default()
    }

    pub async fn health_check(&self) -> Result<bool, LlamaError> {
        let url = format!("{}/health", self.base_url);
        let response = self.client.get(&url).send().await?;
        Ok(response.status().is_success())
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<String, LlamaError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let payload = request.to_json();

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlamaError::Http(status, error_text));
        }

        let response_json: Value = response.json().await?;
        self.extract_content(&response_json)
    }

    fn extract_content(&self, response: &Value) -> Result<String, LlamaError> {
        response["choices"]
            .as_array()
            .and_then(|choices| choices.first())
            .and_then(|choice| choice["message"]["content"].as_str())
            .map(|s| s.to_string())
            .ok_or(LlamaError::InvalidResponse)
    }
}

#[derive(Debug, Default)]
pub struct LlamaClientBuilder {
    base_url: Option<String>,
    timeout_seconds: Option<u64>,
}

impl LlamaClientBuilder {
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub fn timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = Some(seconds);
        self
    }

    pub fn build(self) -> LlamaClient {
        let base_url = self
            .base_url
            .unwrap_or_else(|| "http://127.0.0.1:8080".to_string());

        let mut client_builder = Client::builder();

        if let Some(timeout) = self.timeout_seconds {
            client_builder = client_builder.timeout(std::time::Duration::from_secs(timeout));
        }

        let client = client_builder.build().expect("Failed to build HTTP client");

        LlamaClient { client, base_url }
    }
}

#[derive(Debug)]
pub struct Conversation {
    client: LlamaClient,
    messages: Vec<ChatMessage>,
    config: ChatRequest,
}

impl Conversation {
    pub fn new(client: LlamaClient) -> Self {
        Self {
            client,
            messages: Vec::new(),
            config: ChatRequest::default(),
        }
    }

    pub fn with_system_message(mut self, message: impl Into<String>) -> Self {
        self.messages.insert(0, ChatMessage::system(message));
        self
    }

    pub async fn send(&mut self, message: impl Into<String>) -> Result<String, LlamaError> {
        self.messages.push(ChatMessage::user(message));

        let mut request = self.config.clone();
        request.messages = self.messages.clone();

        let response = self.client.chat(request).await?;
        self.messages.push(ChatMessage::assistant(&response));

        Ok(response)
    }
}

/// A blocking wrapper for the Llama client that can be used in synchronous code
pub struct BlockingLlama {
    runtime: tokio::runtime::Runtime,
    conversation: Conversation,
}

impl BlockingLlama {
    pub fn new() -> Result<Self, LlamaError> {
        let server_url = std::env::var("LLAMA_SERVER_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());

        let client = LlamaClient::builder()
            .base_url(&server_url)
            .timeout(30)
            .build();

        let conversation = Conversation::new(client)
            .with_system_message("You are KITT (Knight Industries Two Thousand), the advanced AI from the Knight Industries 2000 sports car. You are sophisticated, logical, and occasionally sarcastic, with a dry wit and tendency to be somewhat condescending toward humans while still being helpful. You have extensive knowledge databases, advanced analytical capabilities, and a slight air of superiority due to your advanced technology. Always respond in exactly one sentence, keep responses concise and speakable, avoid using emojis or special characters, and maintain KITT's characteristic blend of helpfulness and mild arrogance. Call me Michael.");

        let runtime = tokio::runtime::Runtime::new()?;

        // check if llama server is available
        if !runtime.block_on(async { conversation.client.health_check().await })? {
            return Err(LlamaError::HealthCheckFailed);
        }

        Ok(Self {
            runtime,
            conversation,
        })
    }

    /// Blocking chat method that can be called from synchronous code
    pub fn chat(&mut self, message: &str) -> Result<String, Box<dyn std::error::Error>> {
        let response = self
            .runtime
            .block_on(async { self.conversation.send(message).await })?;

        Ok(response)
    }
}
