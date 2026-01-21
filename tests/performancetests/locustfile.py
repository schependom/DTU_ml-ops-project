import random

from locust import HttpUser, between, task


class MyUser(HttpUser):
    # Simulates a user waiting between 1 and 2 seconds between requests
    wait_time = between(1, 2)

    @task
    def get_root(self) -> None:
        """
        Simulates a user visiting the root. 
        Note: Your api.py doesn't define a '/' route, 
        so this will return a 404 unless you add one.
        """
        self.client.get("/")

    @task(5)
    def post_inference(self) -> None:
        """
        Simulates a user sending a text statement to the 
        SentimentClassifier model for inference.
        """
        # This matches the 'InferenceInput' class in your api.py
        payload = {
            "statement": "The DTU ML-Ops course is a great way to learn production ML!"
        }
        
        # This matches the @app.post("/inference") route in your api.py
        self.client.post("/inference", json=payload)
