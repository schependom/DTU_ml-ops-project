from locust import HttpUser, between, task


class MyUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def get_root(self) -> None:
        # Simulates a user visiting the root. 
        self.client.get("/")

    @task(5)
    def post_inference(self) -> None:
        #Simulates a user sending a text statement to the SentimentClassifier model for inference
        # matches the InferenceInput class in api.py
        payload = {
            "statement": "Succes!"
        }
        # matches the @app.post("/inference") route in api.py
        self.client.post("/inference", json=payload)
