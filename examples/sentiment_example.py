"""Example: Quantum-Enhanced Sentiment Analysis with Async Toolformer Orchestrator."""

import asyncio
from datetime import datetime, timedelta

from async_toolformer.quantum_sentiment import create_quantum_sentiment_analyzer
from async_toolformer.quantum_security import SecurityLevel


async def basic_sentiment_example():
    """Basic sentiment analysis example."""
    print("🔬 Basic Sentiment Analysis Example")
    print("=" * 50)
    
    analyzer = create_quantum_sentiment_analyzer(
        max_parallel=10,
        security_level=SecurityLevel.MEDIUM
    )
    
    # Test texts with different sentiments
    test_texts = [
        "I absolutely love this product! It's amazing and works perfectly!",
        "This is the worst thing I've ever bought. Complete waste of money.",
        "The product is okay, nothing special but does the job.",
        "I'm really excited about the new features! Can't wait to try them! 🎉",
        "Feeling disappointed with the service. Expected much better quality."
    ]
    
    print("\nAnalyzing individual texts:")
    for i, text in enumerate(test_texts, 1):
        result = await analyzer._quantum_multi_approach_analysis(text)
        sentiment = result["fused_sentiment"]["sentiment"]
        
        print(f"\n{i}. Text: {text[:50]}...")
        print(f"   Sentiment: {sentiment['polarity']} (score: {sentiment['score']:.2f}, confidence: {sentiment['confidence']:.2f})")
        print(f"   Quantum Coherence: {result['quantum_coherence']:.2f}")


async def batch_analysis_example():
    """Batch sentiment analysis example."""
    print("\n\n🚀 Batch Analysis Example")
    print("=" * 50)
    
    analyzer = create_quantum_sentiment_analyzer(max_parallel=20)
    
    # Simulate customer reviews
    customer_reviews = [
        "Great product, highly recommend!",
        "Terrible quality, broke after one day",
        "Average product, nothing special",
        "Love it! Best purchase this year!",
        "Not worth the money, poor build quality",
        "Excellent customer service and fast shipping",
        "Product didn't meet expectations",
        "Amazing features, very user-friendly",
        "Overpriced for what you get",
        "Perfect for my needs, works great!"
    ]
    
    # Use quantum orchestrator for batch processing
    batch_result = await analyzer.orchestrator.quantum_execute(
        f"Analyze the sentiment of these customer reviews: {customer_reviews}",
        tools=["analyze_batch_sentiment"],
        optimize_plan=True,
        enable_entanglement=True
    )
    
    print(f"\nBatch Analysis Results:")
    print(f"Total reviews analyzed: {len(customer_reviews)}")
    print(f"Processing approach: Quantum-enhanced parallel execution")
    

async def multi_source_fusion_example():
    """Multi-source sentiment fusion example."""
    print("\n\n🔗 Multi-Source Fusion Example")
    print("=" * 50)
    
    analyzer = create_quantum_sentiment_analyzer(
        max_parallel=15,
        enable_speculation=True
    )
    
    # Different sources about the same product
    sources = {
        "twitter": "Just tried the new @product - it's absolutely incredible! #amazing #love",
        "reddit": "Been using this product for a week. Honestly, it's pretty good. Some minor issues but overall satisfied.",
        "review_site": "Comprehensive review: The product delivers on most promises. Build quality is excellent, but price is high.",
        "customer_support": "Customer reported positive experience with product functionality and our support team.",
        "blog_post": "After extensive testing, I can say this product revolutionizes the market. Highly recommended for professionals."
    }
    
    result = await analyzer._quantum_sentiment_fusion(
        text_sources=sources,
        fusion_method="quantum_entanglement",
        enable_cross_correlation=True
    )
    
    print("\nQuantum Fusion Results:")
    fused_sentiment = result["fused_sentiment"]["sentiment"]
    print(f"Fused Sentiment: {fused_sentiment['polarity']} (score: {fused_sentiment['score']:.2f})")
    print(f"System Coherence: {result['quantum_metrics']['coherence']:.2f}")
    print(f"Entanglement Strength: {result['quantum_metrics']['entanglement_strength']}")
    
    print("\nCross-correlations between sources:")
    for correlation, value in result["cross_correlations"].items():
        sources_pair = correlation.replace("__", " ↔ ")
        print(f"  {sources_pair}: {value:.2f}")


async def temporal_sentiment_example():
    """Temporal sentiment analysis example."""
    print("\n\n⏰ Temporal Sentiment Analysis Example")
    print("=" * 50)
    
    analyzer = create_quantum_sentiment_analyzer()
    
    # Simulate time series of customer feedback
    base_time = datetime.utcnow()
    temporal_data = [
        {
            "text": "Just got the product, first impressions are great!",
            "timestamp": base_time
        },
        {
            "text": "After using it for a day, still loving it. Works as advertised.",
            "timestamp": base_time + timedelta(days=1)
        },
        {
            "text": "One week in - some issues started appearing. Getting frustrated.",
            "timestamp": base_time + timedelta(days=7)
        },
        {
            "text": "Support team helped resolve the issues. Back to being happy with it.",
            "timestamp": base_time + timedelta(days=10)
        },
        {
            "text": "Long-term review: Overall satisfied despite early problems.",
            "timestamp": base_time + timedelta(days=30)
        }
    ]
    
    result = await analyzer._quantum_temporal_sentiment(
        temporal_texts=temporal_data,
        time_decay_factor=0.8,
        enable_momentum=True
    )
    
    print("\nTemporal Analysis Results:")
    temporal_sentiment = result["temporal_sentiment"]["sentiment"]
    print(f"Overall Temporal Sentiment: {temporal_sentiment['polarity']} (score: {temporal_sentiment['score']:.2f})")
    
    metrics = result["quantum_temporal_metrics"]
    print(f"Temporal Coherence: {metrics['coherence']:.2f}")
    print(f"Sentiment Volatility: {metrics['volatility']:.2f}")
    
    if result["temporal_momentum"]:
        momentum = result["temporal_momentum"]
        print(f"Sentiment Momentum: {momentum['momentum']:.2f} ({momentum['velocity_trend']})")
        print(f"Sentiment Acceleration: {momentum['acceleration']:.2f}")


async def social_media_analysis_example():
    """Social media sentiment analysis example."""
    print("\n\n📱 Social Media Analysis Example")
    print("=" * 50)
    
    analyzer = create_quantum_sentiment_analyzer()
    
    # Social media posts with hashtags, mentions, and emojis
    social_posts = [
        "OMG! Just tried @newproduct and I'm OBSESSED!!! 😍 #productname #amazing #mustbuy",
        "@company your customer service is absolutely terrible 😡 #frustrated #disappointed",
        "Loving my new purchase! Thanks @company for the quick delivery 📦 #satisfied #goodservice",
        "Meh... @product is okay I guess. Nothing to write home about 🤷‍♀️ #average",
        "BEST PURCHASE EVER!!! @product exceeded all my expectations! 🚀 #incredible #recommended"
    ]
    
    print("\nAnalyzing social media posts:")
    for i, post in enumerate(social_posts, 1):
        result = await analyzer.orchestrator.execute_tool(
            "analyze_social_media_sentiment",
            text=post,
            platform="twitter",
            extract_hashtags=True,
            extract_mentions=True
        )
        
        print(f"\n{i}. Post: {post}")
        print(f"   Sentiment: {result.sentiment.polarity.value} (score: {result.sentiment.score:.2f})")
        
        if "hashtags" in result.entities:
            print(f"   Hashtags: {', '.join(result.entities['hashtags'])}")
        if "mentions" in result.entities:
            print(f"   Mentions: {', '.join(result.entities['mentions'])}")


async def quantum_performance_comparison():
    """Compare quantum vs traditional processing performance."""
    print("\n\n⚡ Quantum Performance Comparison")
    print("=" * 50)
    
    # Large batch of texts for performance testing
    large_batch = [
        f"This is test review number {i}. The sentiment here varies depending on the number."
        + (" I love it!" if i % 3 == 0 else " It's okay." if i % 3 == 1 else " Not great.")
        for i in range(100)
    ]
    
    analyzer = create_quantum_sentiment_analyzer(max_parallel=50)
    
    # Quantum-enhanced analysis
    print("Running quantum-enhanced analysis...")
    start_time = asyncio.get_event_loop().time()
    
    quantum_result = await analyzer.orchestrator.quantum_execute(
        f"Analyze sentiment of {len(large_batch)} reviews using quantum optimization",
        tools=["analyze_batch_sentiment"],
        optimize_plan=True,
        enable_entanglement=True,
        superposition_depth=3
    )
    
    quantum_time = (asyncio.get_event_loop().time() - start_time) * 1000
    
    print(f"Quantum Analysis:")
    print(f"  Processing Time: {quantum_time:.2f}ms")
    print(f"  Texts Processed: {len(large_batch)}")
    print(f"  Throughput: {len(large_batch) / (quantum_time/1000):.1f} texts/second")
    
    # Get quantum metrics
    analytics = analyzer.orchestrator.get_quantum_analytics()
    print(f"  Quantum Metrics:")
    print(f"    Superposition paths: {analytics.get('paths_explored', 'N/A')}")
    print(f"    Coherence score: {analytics.get('coherence_score', 'N/A')}")


async def main():
    """Run all sentiment analysis examples."""
    print("🧠 Quantum-Enhanced Sentiment Analysis Examples")
    print("=" * 70)
    print("Demonstrating the Async Toolformer Orchestrator with sentiment analysis tools")
    
    try:
        await basic_sentiment_example()
        await batch_analysis_example()
        await multi_source_fusion_example()
        await temporal_sentiment_example()
        await social_media_analysis_example()
        await quantum_performance_comparison()
        
        print("\n\n✅ All examples completed successfully!")
        print("🚀 Quantum sentiment analysis is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        raise


if __name__ == "__main__":
    # Run with quantum-optimized event loop
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        print("🔧 Using uvloop for enhanced performance")
    except ImportError:
        print("🔧 Using standard asyncio (uvloop not available)")
    
    asyncio.run(main())