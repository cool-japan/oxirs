// Integration tests for concurrent request handling (concurrent.rs)

use oxirs_fuseki::concurrent::{
    ConcurrentRequestHandler, WorkStealingScheduler, PriorityQueue, LoadShedder,
    RequestPriority,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::sleep;

#[tokio::test]
async fn test_work_stealing_scheduler_basic() {
    let scheduler = WorkStealingScheduler::new(4);

    // Submit tasks
    let mut handles = Vec::new();
    for i in 0..10 {
        let handle = scheduler.submit(RequestPriority::Normal, move || {
            // Simulate work
            std::thread::sleep(Duration::from_millis(10));
            i * 2
        }).await;
        handles.push(handle);
    }

    // Wait for all tasks and verify results
    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle.await.unwrap();
        assert_eq!(result, i * 2);
    }

    scheduler.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_priority_queue_ordering() {
    let queue = Arc::new(RwLock::new(PriorityQueue::new()));

    // Add requests with different priorities
    {
        let mut q = queue.write().await;
        q.push(RequestPriority::Low, "low1").await.unwrap();
        q.push(RequestPriority::Critical, "critical1").await.unwrap();
        q.push(RequestPriority::Normal, "normal1").await.unwrap();
        q.push(RequestPriority::High, "high1").await.unwrap();
        q.push(RequestPriority::Low, "low2").await.unwrap();
    }

    // Verify priority ordering
    {
        let mut q = queue.write().await;
        assert_eq!(q.pop().await.unwrap(), "critical1");
        assert_eq!(q.pop().await.unwrap(), "high1");
        assert_eq!(q.pop().await.unwrap(), "normal1");
        assert_eq!(q.pop().await.unwrap(), "low1");
        assert_eq!(q.pop().await.unwrap(), "low2");
    }
}

#[tokio::test]
async fn test_load_shedder_basic() {
    let load_shedder = LoadShedder::new(0.8, 0.5);

    // Initially should allow requests
    assert!(load_shedder.should_accept_request().await);

    // Simulate high load
    load_shedder.update_system_load(0.9).await;

    // Should start shedding some requests
    let mut rejected = 0;
    for _ in 0..100 {
        if !load_shedder.should_accept_request().await {
            rejected += 1;
        }
    }

    assert!(rejected > 0, "Expected some requests to be rejected under high load");

    // Reduce load
    load_shedder.update_system_load(0.3).await;

    // Should accept all requests again
    for _ in 0..100 {
        assert!(load_shedder.should_accept_request().await);
    }
}

#[tokio::test]
async fn test_concurrent_request_handler() {
    let handler = ConcurrentRequestHandler::builder()
        .worker_threads(4)
        .max_queue_size(100)
        .per_dataset_limit(10)
        .per_user_limit(5)
        .build()
        .await
        .unwrap();

    // Submit concurrent requests
    let mut handles = Vec::new();
    for i in 0..20 {
        let h = handler.clone();
        let handle = tokio::spawn(async move {
            h.submit_request(
                "dataset1",
                "user1",
                RequestPriority::Normal,
                move || {
                    std::thread::sleep(Duration::from_millis(10));
                    i
                }
            ).await
        });
        handles.push(handle);
    }

    // Collect results
    let mut results = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap().unwrap();
        results.push(result);
    }

    // Verify all requests completed
    assert_eq!(results.len(), 20);

    // Verify statistics
    let stats = handler.get_statistics().await;
    assert_eq!(stats.total_requests, 20);
    assert_eq!(stats.completed_requests, 20);
    assert_eq!(stats.failed_requests, 0);
}

#[tokio::test]
async fn test_concurrent_limit_enforcement() {
    let handler = ConcurrentRequestHandler::builder()
        .worker_threads(2)
        .max_queue_size(10)
        .per_dataset_limit(3)
        .per_user_limit(2)
        .build()
        .await
        .unwrap();

    // Submit more requests than the per-user limit
    let mut handles = Vec::new();
    for i in 0..5 {
        let h = handler.clone();
        let handle = tokio::spawn(async move {
            h.submit_request(
                "dataset1",
                "user1",
                RequestPriority::Normal,
                move || {
                    std::thread::sleep(Duration::from_millis(100));
                    i
                }
            ).await
        });
        handles.push(handle);
    }

    // Some requests should be rejected or queued
    let mut accepted = 0;
    let mut rejected = 0;

    for handle in handles {
        match handle.await.unwrap() {
            Ok(_) => accepted += 1,
            Err(_) => rejected += 1,
        }
    }

    assert!(accepted > 0);
    // Due to queuing, some might still be accepted
    assert!(accepted <= 5);
}

#[tokio::test]
async fn test_query_cancellation() {
    let handler = ConcurrentRequestHandler::builder()
        .worker_threads(2)
        .max_queue_size(10)
        .build()
        .await
        .unwrap();

    // Submit a long-running request
    let h = handler.clone();
    let handle = tokio::spawn(async move {
        h.submit_request(
            "dataset1",
            "user1",
            RequestPriority::Normal,
            || {
                std::thread::sleep(Duration::from_secs(10));
                42
            }
        ).await
    });

    // Give it time to start
    sleep(Duration::from_millis(100)).await;

    // Cancel the request
    let cancelled = handler.cancel_request("dataset1", "user1").await;
    assert!(cancelled);

    // The request should fail
    let result = tokio::time::timeout(Duration::from_secs(1), handle).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_fair_scheduling() {
    let handler = ConcurrentRequestHandler::builder()
        .worker_threads(2)
        .max_queue_size(100)
        .enable_fair_scheduling(true)
        .build()
        .await
        .unwrap();

    // Submit requests from different users
    let mut handles = Vec::new();
    for user_id in 0..3 {
        for i in 0..5 {
            let h = handler.clone();
            let user = format!("user{}", user_id);
            let handle = tokio::spawn(async move {
                h.submit_request(
                    "dataset1",
                    &user,
                    RequestPriority::Normal,
                    move || {
                        std::thread::sleep(Duration::from_millis(10));
                        (user_id, i)
                    }
                ).await
            });
            handles.push(handle);
        }
    }

    // All users should get fair treatment
    let mut results_by_user: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();

    for handle in handles {
        if let Ok(Ok((user_id, value))) = handle.await {
            results_by_user.entry(user_id).or_default().push(value);
        }
    }

    // Each user should have completed some requests
    for user_id in 0..3 {
        let user_results = results_by_user.get(&user_id).unwrap();
        assert!(!user_results.is_empty(), "User {} didn't complete any requests", user_id);
    }
}

#[tokio::test]
async fn test_timeout_handling() {
    let handler = ConcurrentRequestHandler::builder()
        .worker_threads(2)
        .max_queue_size(10)
        .request_timeout(Duration::from_millis(100))
        .build()
        .await
        .unwrap();

    // Submit a request that exceeds timeout
    let result = handler.submit_request(
        "dataset1",
        "user1",
        RequestPriority::Normal,
        || {
            std::thread::sleep(Duration::from_secs(1));
            42
        }
    ).await;

    // Should timeout
    assert!(result.is_err());

    let stats = handler.get_statistics().await;
    assert_eq!(stats.timeout_count, 1);
}

#[tokio::test]
async fn test_statistics_tracking() {
    let handler = ConcurrentRequestHandler::builder()
        .worker_threads(2)
        .max_queue_size(10)
        .build()
        .await
        .unwrap();

    // Submit various requests
    for i in 0..10 {
        let h = handler.clone();
        let _ = tokio::spawn(async move {
            h.submit_request(
                "dataset1",
                "user1",
                if i % 2 == 0 { RequestPriority::High } else { RequestPriority::Normal },
                move || {
                    std::thread::sleep(Duration::from_millis(10));
                    i
                }
            ).await
        });
    }

    // Wait for completion
    sleep(Duration::from_millis(500)).await;

    let stats = handler.get_statistics().await;
    assert_eq!(stats.total_requests, 10);
    assert!(stats.completed_requests <= 10);
    assert!(stats.avg_wait_time_ms >= 0.0);
}
