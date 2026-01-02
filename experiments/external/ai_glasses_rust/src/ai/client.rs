//! AI 服务客户端
//!
//! 与 Python AI 服务通信

use std::time::Duration;
use anyhow::Result;
use ndarray::Array1;
use reqwest::Client;
use tracing::{info, warn, debug};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

use crate::buffer::frame::Frame;
use super::types::*;

/// AI 服务客户端
pub struct AiClient {
    client: Client,
    base_url: String,
}

impl AiClient {
    /// 创建新的 AI 客户端
    pub fn new(base_url: impl Into<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: base_url.into(),
        }
    }

    /// V-JEPA2 编码帧
    pub async fn encode_frames(&self, frames: &[Frame]) -> Result<Vec<Array1<f32>>> {
        let url = format!("{}/encode", self.base_url);

        // 将帧转换为 Base64
        let frame_data: Vec<String> = frames.iter()
            .map(|f| BASE64.encode(&f.data))
            .collect();

        let timestamps: Vec<f64> = frames.iter()
            .map(|f| f.timestamp)
            .collect();

        let request = EncodeRequest {
            frames: frame_data,
            timestamps,
        };

        debug!("Sending encode request with {} frames", frames.len());

        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            anyhow::bail!("Encode request failed: {}", error);
        }

        let result: EncodeResponse = response.json().await?;

        // 转换为 ndarray
        let embeddings: Vec<Array1<f32>> = result.embeddings
            .into_iter()
            .map(|e| Array1::from_vec(e))
            .collect();

        debug!("Received {} embeddings", embeddings.len());

        Ok(embeddings)
    }

    /// Gemma 3n 多模态理解
    pub async fn understand(
        &self,
        keyframes: &[Frame],
        audio: Option<&[f32]>,
        sample_rate: Option<u32>,
        prompt: &str,
        messages: Option<Vec<serde_json::Value>>,
    ) -> Result<UnderstandResponse> {
        let url = format!("{}/understand", self.base_url);

        // 转换关键帧
        let keyframe_data: Vec<String> = keyframes.iter()
            .map(|f| BASE64.encode(&f.data))
            .collect();

        let keyframe_times: Vec<f64> = keyframes.iter()
            .map(|f| f.timestamp)
            .collect();

        // 转换音频
        let audio_data = audio.map(|a| {
            let bytes: Vec<u8> = a.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            BASE64.encode(&bytes)
        });

        let request = UnderstandRequest {
            keyframes: keyframe_data,
            keyframe_times,
            audio: audio_data,
            sample_rate,
            prompt: prompt.to_string(),
            messages,
        };

        debug!("Sending understand request");

        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            anyhow::bail!("Understand request failed: {}", error);
        }

        let result: UnderstandResponse = response.json().await?;
        info!("Received response in {}ms", result.processing_time_ms);

        Ok(result)
    }

    /// 音频转录
    pub async fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<TranscribeResponse> {
        let url = format!("{}/transcribe", self.base_url);

        let bytes: Vec<u8> = audio.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let audio_data = BASE64.encode(&bytes);

        let request = TranscribeRequest {
            audio: audio_data,
            sample_rate,
        };

        debug!("Sending transcribe request");

        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            anyhow::bail!("Transcribe request failed: {}", error);
        }

        let result: TranscribeResponse = response.json().await?;
        Ok(result)
    }

    /// 健康检查
    pub async fn health(&self) -> Result<bool> {
        let url = format!("{}/health", self.base_url);

        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(e) => {
                warn!("Health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = AiClient::new("http://localhost:8080");
        assert_eq!(client.base_url, "http://localhost:8080");
    }
}
