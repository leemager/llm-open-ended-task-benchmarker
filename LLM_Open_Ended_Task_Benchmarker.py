import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import anthropic
from google import genai
from google.genai import types

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# not worth skimping on costs for the grading part, 
# but o1-mini could probably do as good a job
grading_model = "o1"

# Number of times to run each eval
NUM_RUNS = 5

# To edit based on the tasks' marking criteria categories
EVALUATION_CATEGORIES = [
    "basicQAConcepts",
    "deadlineAccuracy",
    "consultationSteps",
    "resourcePedagogy",
    "completenessImplementation"
]

# Handle annoying API model idiocynrasies 
# Temp 0.2 for models that support it given nature of task
MODEL_CONFIGS = {
    "gpt-4o": {
        "provider": "openai",
        "supports_system": True,
        "temperature": 0.2
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "supports_system": True,
        "temperature": 0.2
    },
    "o1": {
        "provider": "openai",
        "supports_system": True
        # defaulting to medium reasoning effort
    },
    "o1-mini": {
        "provider": "openai",
        "supports_system": False
    },
    "claude-3-5-haiku-20241022": {
        "provider": "anthropic",
        "supports_system": True,
        "required_params": {"max_tokens": 4096},
        "temperature": 0.2
    },
    "claude-3-5-sonnet-20241022": {
        "provider": "anthropic",
        "supports_system": True,
        "required_params": {"max_tokens": 4096},
        "temperature": 0.2
    },
    "gemini-2.0-flash-exp": {
        "provider": "google",
        "supports_system": True,
        "temperature": 0.2
    },
}


models_to_test = [
    "gpt-4o-mini",
    "gpt-4o",
    "o1-mini",
    "o1",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "gemini-2.0-flash-exp"
]

class BaseLLMClient(ABC):
    def __init__(self, api_key, model_name, model_config):
        self.api_key = api_key
        self.model_name = model_name
        self.model_config = model_config

    @abstractmethod
    def send_message(self, messages):
        pass

class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key, model_name, model_config):
        super().__init__(api_key, model_name, model_config)
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def send_message(self, messages):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": messages
        }
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=180
            )
            if not response.ok:
                print(f"OpenAI request failed: {response.status_code} - {response.text}")
                return "ERROR"
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with OpenAI: {e}")
            return "ERROR"

class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key, model_name, model_config):
        super().__init__(api_key, model_name, model_config)
        self.anthropic_client = anthropic.Anthropic(api_key=api_key)

    def send_message(self, messages):
        try:
            system_message = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_messages = [m for m in messages if m["role"] != "system"]

            max_tokens = self.model_config.get("required_params", {}).get("max_tokens", 4096)

            response = self.anthropic_client.messages.create(
                model=self.model_name,
                system=system_message,
                messages=user_messages,
                max_tokens=max_tokens
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error communicating with Anthropic: {e}")
            return "ERROR"

class GoogleGeminiClient(BaseLLMClient):
    def __init__(self, api_key, model_name, model_config):
        super().__init__(api_key, model_name, model_config)
        self.client = genai.Client(api_key=api_key)

    def send_message(self, messages):
        system_prompt = ""
        user_prompt = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt += msg["content"] + "\n"
            else:  
                user_prompt += msg["content"] + "\n"
        
        final_prompt = system_prompt + user_prompt

        temperature = self.model_config.get("temperature", 0.2)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=4000,  
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=final_prompt,
                config=config
            )

            if hasattr(response, "text"):
                return response.text
            else:
                print("Unexpected response format from Google Gemini.")
                return "ERROR"

        except Exception as e:
            print(f"Error communicating with Google Gemini: {e}")
            return "ERROR"

class LLMClientFactory:
    @staticmethod
    def create_client(model_name):
        config = MODEL_CONFIGS.get(model_name, {})
        provider = config.get("provider", "openai")  # Default to openai if not specified

        if provider == "anthropic":
            return AnthropicClient(ANTHROPIC_API_KEY, model_name, config)
        elif provider == "google":
            return GoogleGeminiClient(GOOGLE_API_KEY, model_name, config)
        else: 
            return OpenAIClient(OPENAI_API_KEY, model_name, config)

def read_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return ""
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
        return ""

def verify_input_files(files_dict):
    empty_files = [name for name, content in files_dict.items() if not content]
    if empty_files:
        print(f"Warning: The following files are empty: {', '.join(empty_files)}")
        return False
    return True

def build_messages_for_model(model_name, system_prompt, user_prompt):
    config = MODEL_CONFIGS.get(model_name, {"supports_system": True})
    if config["supports_system"]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        return [{"role": "user", "content": combined_prompt}]

def get_model_solution(model_name, system_prompt, user_prompt):
    messages = build_messages_for_model(model_name, system_prompt, user_prompt)
    client = LLMClientFactory.create_client(model_name)
    return client.send_message(messages)

def grade_solution(grading_model_name, model_response, marking_criteria, perfect_answer):
    scores_placeholder = ",\n            ".join([f"\"{cat}\": 0" for cat in EVALUATION_CATEGORIES])
    feedback_placeholder = ",\n            ".join([f"\"{cat}\": \"\"" for cat in EVALUATION_CATEGORIES])

    grading_prompt = f"""
        You have the following marking criteria:

        {marking_criteria}

        Perfect 50/50 solution is here:
        {perfect_answer}

        Student's solution:
        {model_response}

        Please evaluate and output JSON with these exact keys:

        {{
            "scores": {{
                {scores_placeholder}
            }},
            "feedback": {{
                {feedback_placeholder}
            }},
            "overallComments": ""
        }}
        No extra text, no disclaimers, just valid JSON.
    """

    grading_messages = [
        {"role": "system", "content": "You are a strict but fair grader."},
        {"role": "user", "content": grading_prompt},
    ]

    client = LLMClientFactory.create_client(grading_model_name)
    return client.send_message(grading_messages)

def create_benchmark_plot(df, output_path):
    df['model'] = df['model'].replace({
        'claude-3-5-sonnet-20241022': 'sonnet',
        'claude-3-5-haiku-20241022': 'haiku',
        'gemini-2.0-flash-exp': 'gemini 2 flash'
    })
    
    mean_scores = df.groupby('model')['total_score'].mean().sort_values()
    ordered_models = mean_scores.index
    
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    box = ax.boxplot(
        [df[df['model'] == model]['total_score'] for model in ordered_models],
        labels=ordered_models,
        patch_artist=True
    )
    
    for box_element in box['boxes']:
        box_element.set(color="green", linewidth=2, facecolor="green", alpha=0.7)
    for whisker in box['whiskers']:
        whisker.set(color="green", linewidth=2)
    for cap in box['caps']:
        cap.set(color="green", linewidth=2)
    for median in box['medians']:
        median.set(color="lime", linewidth=2)
    
    ax.set_title("LLM Performance Benchmark Results", fontsize=14, color="white")
    ax.set_xlabel("Model", fontsize=12, color="white")
    ax.set_ylabel("Score (%)", fontsize=12, color="white")
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(False)
    
    plt.savefig(output_path, bbox_inches='tight', facecolor='black')
    plt.close()

def main():
    task_text = read_file("Task.txt")
    knowledge_text = read_file("Knowledge.txt")
    marking_criteria_text = read_file("Marking Criteria.txt")
    perfect_answer_text = read_file("Model Answer.txt")

    input_files = {
        'task': task_text,
        'knowledge': knowledge_text,
        'marking_criteria': marking_criteria_text,
        'perfect_answer': perfect_answer_text
    }
    if not verify_input_files(input_files):
        print("One or more input files are missing or empty. Exiting.")
        return

    system_prompt = "You are an effective and diligent knowledge worker."
    user_prompt = f"""Here is the task:

        {task_text}

        ---
        Below is additional knowledge that may be relevant:

        {knowledge_text}
    """

    results = []

    for run in range(1, NUM_RUNS + 1):
        print(f"\n=== Starting Run {run}/{NUM_RUNS} ===")
        
        for model_name in models_to_test:
            print(f"\n--- Getting solution from {model_name} (Run {run}) ---")
            model_solution = get_model_solution(model_name, system_prompt, user_prompt)
            if model_solution == "ERROR":
                print(f"Skipping grading for {model_name} due to API error.")
                continue

            print(f"Grading solution from {model_name} ...")
            grading_output = grade_solution(
                grading_model_name=grading_model,
                model_response=model_solution,
                marking_criteria=marking_criteria_text,
                perfect_answer=perfect_answer_text
            )

            if grading_output == "ERROR":
                print(f"Skipping parsing for {model_name} due to grading API error.")
                continue

            try:
                parsed_output = json.loads(grading_output)
                scores = parsed_output.get("scores", {})
                feedback = parsed_output.get("feedback", {})
                overall_comments = parsed_output.get("overallComments", "")

                result_dict = {
                    "model": model_name,
                    "run": run
                }
                
                total_score = 0
                for cat in EVALUATION_CATEGORIES:
                    score = scores.get(cat, 0)
                    total_score += score
                    result_dict[f"{cat}_score"] = score
                    result_dict[f"{cat}_feedback"] = feedback.get(cat, "")
                
                result_dict["total_score"] = total_score
                result_dict["overallComments"] = overall_comments
                result_dict["modelSolution"] = model_solution

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from grading model for {model_name}: {e}")
                result_dict = {
                    "model": model_name,
                    "run": run,
                    **{f"{cat}_score": None for cat in EVALUATION_CATEGORIES},
                    **{f"{cat}_feedback": "" for cat in EVALUATION_CATEGORIES},
                    "total_score": None,
                    "overallComments": "Parsing error",
                    "modelSolution": model_solution,
                }

            results.append(result_dict)

    if not results:
        print("No results to save. Exiting.")
        return

    df = pd.DataFrame(results)
    base_path = r"C:\your base path"
    csv_path = os.path.join(base_path, "llm_benchmark_results.csv")
    plot_path = os.path.join(base_path, "llm_benchmark_box_plot.png")
    
    try:
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        
        create_benchmark_plot(df, plot_path)
        print(f"Benchmark plot saved to {plot_path}")
        
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()
