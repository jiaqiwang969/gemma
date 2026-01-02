//! AI 眼镜系统 - 意图-嵌入核心循环演示
//!
//! 核心机制：
//! 用户意图 ←→ V-JEPA2 视频嵌入 双向循环

#![allow(unused_imports)]

use ndarray::Array1;
use tokio::sync::mpsc;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use ai_glasses::{
    IntentEmbeddingEngine, EngineEvent, IntentType, ExtractionStrategy, Frame,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 初始化日志
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("╔══════════════════════════════════════════════════════════╗");
    info!("║     AI 眼镜系统 - 意图-嵌入核心循环                       ║");
    info!("╚══════════════════════════════════════════════════════════╝");

    demo_intent_embedding_loop().await?;

    Ok(())
}

async fn demo_intent_embedding_loop() -> anyhow::Result<()> {
    info!("\n=== 场景1: 用户问「描述场景」===");
    demo_scenario("描述一下这个场景", generate_normal_video()).await;

    info!("\n=== 场景2: 用户问「什么时候出现了人」===");
    demo_scenario("什么时候出现了人", generate_video_with_person()).await;

    info!("\n=== 场景3: 用户问「视频前后有什么变化」===");
    demo_scenario("视频前后有什么变化", generate_changing_video()).await;

    info!("\n=== 场景4: 意图动态演化 ===");
    demo_intent_evolution().await;

    Ok(())
}

async fn demo_scenario(query: &str, video_data: Vec<(Frame, f32)>) {
    info!("用户查询: \"{}\"", query);

    // 创建引擎
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let mut engine = IntentEmbeddingEngine::new(query)
        .with_event_channel(event_tx);

    info!(
        "  初始意图: {:?} (置信度: {:.0}%)",
        engine.intent().primary,
        engine.intent().primary_confidence * 100.0
    );
    info!("  提取策略: {:?}", engine.current_strategy());

    // 处理视频帧
    for (frame, change_score) in video_data {
        let embedding = generate_mock_embedding(&frame);
        engine.process(frame, embedding, change_score);
    }

    // 收集事件
    let mut events = Vec::new();
    while let Ok(event) = event_rx.try_recv() {
        events.push(event);
    }

    let stats = engine.stats();
    info!("  处理结果:");
    info!("    - Embedding 数量: {}", stats.embedding_count);
    info!("    - 关键帧数量: {}", stats.keyframe_count);
    info!("    - 迭代次数: {}", stats.iteration);

    // 显示事件
    let keyframe_events: Vec<_> = events.iter()
        .filter(|e| matches!(e, EngineEvent::KeyframeSelected { .. }))
        .collect();
    info!("    - 选中的关键帧: {} 个", keyframe_events.len());
}

async fn demo_intent_evolution() {
    info!("演示意图如何从 embedding 分析中演化...\n");

    let mut engine = IntentEmbeddingEngine::new("描述场景");
    info!("初始意图: {:?}", engine.intent().primary);

    // 模拟处理有显著变化的视频
    info!("\n[处理有显著变化的视频...]");
    for i in 0..20 {
        let frame = Frame::from_vec(vec![(i * 10 % 256) as u8; 1000], i as f64 * 0.5, 10, 10);
        let embedding = Array1::from_vec(vec![
            if i < 10 { 0.1 } else { 0.9 }; 1024
        ]);
        // 在第10帧有大变化
        let change = if i == 10 { 0.5 } else { 0.02 };
        engine.process(frame, embedding, change);
    }

    info!(
        "处理后意图: {:?} (置信度: {:.0}%)",
        engine.intent().primary,
        engine.intent().primary_confidence * 100.0
    );
    info!("次要意图: {:?}", engine.intent().secondary);

    // 用户反馈
    info!("\n[用户追问: \"什么时候发生了变化?\"]");
    engine.update_from_user("什么时候发生了变化?");

    info!(
        "更新后意图: {:?} (置信度: {:.0}%)",
        engine.intent().primary,
        engine.intent().primary_confidence * 100.0
    );
    info!("策略: {:?}", engine.current_strategy());

    // 获取相关 embedding
    let relevant = engine.get_relevant_embeddings();
    info!(
        "\n根据当前意图，相关 embedding 数量: {}",
        relevant.len()
    );
    for (t, _) in relevant.iter().take(5) {
        info!("  - 时间: {:.1}s", t);
    }
}

// ============================================================
// 模拟数据生成
// ============================================================

fn generate_normal_video() -> Vec<(Frame, f32)> {
    (0..20)
        .map(|i| {
            let frame = Frame::from_vec(vec![100u8; 1000], i as f64 * 0.5, 10, 10);
            let change = 0.02 + (i as f32 * 0.005); // 轻微变化
            (frame, change)
        })
        .collect()
}

fn generate_video_with_person() -> Vec<(Frame, f32)> {
    (0..20)
        .map(|i| {
            let frame = Frame::from_vec(vec![(i * 5 % 256) as u8; 1000], i as f64 * 0.5, 10, 10);
            // 在第8-12帧有人出现
            let change = if (8..=12).contains(&i) { 0.25 } else { 0.03 };
            (frame, change)
        })
        .collect()
}

fn generate_changing_video() -> Vec<(Frame, f32)> {
    (0..20)
        .map(|i| {
            let frame = Frame::from_vec(vec![(i * 10 % 256) as u8; 1000], i as f64 * 0.5, 10, 10);
            // 开始和结束有大变化
            let change = if i < 3 || i > 17 { 0.3 } else { 0.05 };
            (frame, change)
        })
        .collect()
}

fn generate_mock_embedding(frame: &Frame) -> Array1<f32> {
    // 基于帧时间戳生成不同的 embedding
    let base = (frame.timestamp * 10.0) as f32 % 1.0;
    Array1::from_vec(vec![base; 1024])
}
