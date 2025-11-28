"""
Locust Load Testing for Animal Action Recognition API
Tests the model's ability to handle multiple concurrent requests
"""

from locust import HttpUser, task, between, events
import os
import random
from pathlib import Path


class AnimalActionUser(HttpUser):
    """
    Simulated user making prediction requests
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts"""
        # Get list of test videos
        self.test_videos = self._get_test_videos()
        print(f"User started with {len(self.test_videos)} test videos")
    
    def _get_test_videos(self):
        """Get list of available test videos"""
        video_dir = Path("../dataset/video")
        
        if video_dir.exists():
            videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
            return [str(v) for v in videos[:10]]  # Limit to 10 videos
        
        return []
    
    @task(weight=10)
    def predict_video(self):
        """
        Main task: Send video for prediction
        Weight=10 means this will run 10x more than other tasks
        """
        if not self.test_videos:
            print("No test videos available")
            return
        
        # Select random video
        video_path = random.choice(self.test_videos)
        
        if not os.path.exists(video_path):
            return
        
        # Send prediction request
        with open(video_path, 'rb') as video_file:
            files = {
                'video': (os.path.basename(video_path), video_file, 'video/mp4')
            }
            data = {
                'max_frames': 32,
                'sample_rate': 2
            }
            
            with self.client.post(
                "/predict",
                files=files,
                data=data,
                catch_response=True,
                name="predict_video"
            ) as response:
                if response.status_code == 200:
                    try:
                        result = response.json()
                        if 'animal' in result and 'action' in result:
                            response.success()
                        else:
                            response.failure("Invalid response format")
                    except Exception as e:
                        response.failure(f"JSON parse error: {e}")
                else:
                    response.failure(f"Status code: {response.status_code}")
    
    @task(weight=2)
    def health_check(self):
        """
        Health check task
        Weight=2 means this runs less frequently
        """
        with self.client.get("/health", catch_response=True, name="health_check") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(weight=1)
    def get_model_info(self):
        """
        Get model information
        Weight=1 means this runs least frequently
        """
        with self.client.get("/info", catch_response=True, name="model_info") as response:
            if response.status_code == 200:
                try:
                    info = response.json()
                    if 'num_animals' in info:
                        response.success()
                    else:
                        response.failure("Invalid info response")
                except:
                    response.failure("JSON parse error")
            else:
                response.failure(f"Info request failed: {response.status_code}")


# Event listeners for detailed reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts"""
    print("="*60)
    print("LOAD TEST STARTING")
    print("="*60)
    print(f"Target host: {environment.host}")
    print(f"Users will spawn at rate configured in command line")
    print("="*60)


@events.test_stop.add_listener  
def on_test_stop(environment, **kwargs):
    """Called when the test stops"""
    print("\n" + "="*60)
    print("LOAD TEST COMPLETE")
    print("="*60)
    
    # Print summary
    stats = environment.stats
    print(f"\nTotal Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"Min Response Time: {stats.total.min_response_time}ms")
    print(f"Max Response Time: {stats.total.max_response_time}ms")
    print(f"Requests per Second: {stats.total.total_rps:.2f}")
    
    print("\nDetailed Breakdown:")
    for name, entry in stats.entries.items():
        print(f"\n{name}:")
        print(f"  Requests: {entry.num_requests}")
        print(f"  Failures: {entry.num_failures}")
        print(f"  Avg Time: {entry.avg_response_time:.2f}ms")
        print(f"  RPS: {entry.total_rps:.2f}")
    
    print("="*60)


# Alternative: Quick stress test for specific endpoints
class QuickStressTest(HttpUser):
    """
    Simplified stress test focusing only on prediction endpoint
    """
    wait_time = between(0.5, 1)  # Faster requests
    
    @task
    def rapid_predict(self):
        """Send rapid prediction requests"""
        # Use a small test video for speed
        test_video = "../dataset/video/test.mp4"
        
        if os.path.exists(test_video):
            with open(test_video, 'rb') as f:
                files = {'video': ('test.mp4', f, 'video/mp4')}
                data = {'max_frames': 16, 'sample_rate': 3}  # Faster processing
                
                self.client.post("/predict", files=files, data=data)