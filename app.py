from flask import Flask, request, jsonify, g
from functools import wraps
import anthropic
import openai
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
from bson import ObjectId
import os
import dotenv
from threading import Thread
from datetime import datetime, timedelta, timezone
import colorsys
from typing import List, Dict, Optional, Union, Any
import json

# Load environment variables
dotenv.load_dotenv(override=False)

# Initialize Flask with API clients
def create_app():
    app = Flask(__name__)
    
    # Load environment variables
    dotenv.load_dotenv(override=False)
    
    # Initialize API clients
    if os.getenv("ANTHROPIC_API_KEY"):
        app.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        app.anthropic_client = None
        
    if os.getenv("OPENAI_API_KEY"):
        app.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        app.openai_client = None
    
    # Initialize conversations dict
    app.conversations = {}
    
    return app

app = create_app()

# Store active conversations
# MongoDB setup
mongo_uri = os.getenv('MONGODB_URI')
mongo_client = MongoClient(mongo_uri)
db = mongo_client['degen']

# Collections
users = db['users']
conversations = db['conversations']
messages = db['messages']

# Model configuration from original script
MODEL_INFO = {
    "sonnet": {
        "api_name": "claude-3-5-sonnet-20240620",
        "display_name": "Claude",
        "company": "anthropic",
    },
    "opus": {
        "api_name": "claude-3-opus-20240229",
        "display_name": "Claude",
        "company": "anthropic",
    },
    "gpt4o": {
        "api_name": "gpt-4o-2024-08-06",
        "display_name": "GPT4o",
        "company": "openai",
    },
    "o1-preview": {"api_name": "o1-preview", "display_name": "O1", "company": "openai"},
    "o1-mini": {"api_name": "o1-mini", "display_name": "Mini", "company": "openai"},
}

def claude_conversation(actor: str, model: str, context: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
    messages = [{"role": m["role"], "content": m["content"]} for m in context]
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": 1024,
        "temperature": 1.0,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    message = app.anthropic_client.messages.create(**kwargs)
    return message.content[0].text

def gpt4_conversation(actor: str, model: str, context: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
    messages = [{"role": m["role"], "content": m["content"]} for m in context]
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 1.0,
    }
    
    if model == "o1-preview" or model == "o1-mini":
        kwargs["max_completion_tokens"] = 4000
    else:
        kwargs["max_tokens"] = 1024
    
    response = app.openai_client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

def load_template(template_name: str, models: List[str]) -> List[Dict[str, Any]]:
    try:
        with open(f"templates/{template_name}.jsonl", "r") as f:
            configs = [json.loads(line) for line in f]

        companies = [MODEL_INFO[model]["company"] for model in models]
        actors = [f"{MODEL_INFO[model]['display_name']} {i+1}" for i, model in enumerate(models)]

        for i, config in enumerate(configs):
            config["system_prompt"] = config["system_prompt"].format(
                **{f"lm{j+1}_company": company for j, company in enumerate(companies)},
                **{f"lm{j+1}_actor": actor for j, actor in enumerate(actors)},
            )
            
            for message in config["context"]:
                message["content"] = message["content"].format(
                    **{f"lm{j+1}_company": company for j, company in enumerate(companies)},
                    **{f"lm{j+1}_actor": actor for j, actor in enumerate(actors)},
                )

            if MODEL_INFO[models[i]]["company"] == "openai" and config["system_prompt"]:
                system_prompt_added = False
                for message in config["context"]:
                    if message["role"] == "user":
                        message["content"] = f"<SYSTEM>{config['system_prompt']}</SYSTEM>\n\n{message['content']}"
                        system_prompt_added = True
                        break
                if not system_prompt_added:
                    config["context"].append({
                        "role": "user",
                        "content": f"<SYSTEM>{config['system_prompt']}</SYSTEM>",
                    })
        return configs
    except FileNotFoundError:
        raise ValueError(f"Template '{template_name}' not found.")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in template '{template_name}'.")

def get_available_templates() -> List[str]:
    template_dir = "./templates"
    templates = []
    for file in os.listdir(template_dir):
        if file.endswith(".jsonl"):
            templates.append(os.path.splitext(file)[0])
    return templates

@app.route("/api/templates", methods=["GET"])
def list_templates() -> Union[Dict[str, Any], tuple[Dict[str, Any], int]]:
    """Get list of available templates"""
    try:
        templates = get_available_templates()
        return jsonify({
            "status": "success",
            "templates": templates
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/api/models", methods=["GET"])
def list_models() -> Dict[str, Any]:
    """Get list of available models"""
    return jsonify({
        "status": "success",
        "models": list(MODEL_INFO.keys())
    })

@app.route("/api/conversation", methods=["POST"])
def start_conversation() -> Union[Dict[str, Any], tuple[Dict[str, Any], int]]:
    """Start a new conversation between AI models"""
    try:
        data = request.json
        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
            
        models = data.get("models", ["opus", "opus"])
        template = data.get("template", "cli")
        max_turns = data.get("max_turns", float("inf"))

        # Validate models
        if not all(model in MODEL_INFO for model in models):
            return jsonify({
                "status": "error",
                "message": "Invalid model selection"
            }), 400

        # Load template configurations
        configs = load_template(template, models)
        
        if len(models) != len(configs):
            return jsonify({
                "status": "error",
                "message": f"Number of models ({len(models)}) does not match template configuration ({len(configs)})"
            }), 400

        # Initialize conversation state
        conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        conversation_state = {
            "id": conversation_id,
            "models": models,
            "lm_models": [MODEL_INFO[model]["api_name"] for model in models],
            "lm_display_names": [f"{MODEL_INFO[model]['display_name']} {i+1}" for i, model in enumerate(models)],
            "system_prompts": [config["system_prompt"] for config in configs],
            "contexts": [config["context"] for config in configs],
            "turn": 0,
            "max_turns": max_turns
        }

        app.conversations[conversation_id] = conversation_state

        return jsonify({
            "status": "success",
            "conversation_id": conversation_id,
            "message": "Conversation initialized"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/api/conversation/<conversation_id>/next", methods=["POST"])
def next_turn(conversation_id: str) -> Union[Dict[str, Any], tuple[Dict[str, Any], int]]:
    """Generate the next turn in the conversation"""
    try:
        if conversation_id not in app.conversations:
            return jsonify({
                "status": "error",
                "message": "Conversation not found"
            }), 404

        state = app.conversations[conversation_id]
        
        if state["turn"] >= state["max_turns"]:
            return jsonify({
                "status": "error",
                "message": "Maximum turns reached"
            }), 400

        responses = []
        for i in range(len(state["models"])):
            lm_response = generate_model_response(
                state["lm_models"][i],
                state["lm_display_names"][i],
                state["contexts"][i],
                state["system_prompts"][i]
            )
            
            # Update contexts
            for j, context in enumerate(state["contexts"]):
                if j == i:
                    context.append({"role": "assistant", "content": lm_response})
                else:
                    context.append({"role": "user", "content": lm_response})
            
            responses.append({
                "actor": state["lm_display_names"][i],
                "response": lm_response
            })

        state["turn"] += 1

        return jsonify({
            "status": "success",
            "turn": state["turn"],
            "responses": responses
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def generate_model_response(model: str, actor: str, context: List[Dict[str, str]], system_prompt: Optional[str]) -> str:
    if model.startswith("claude-"):
        return claude_conversation(actor, model, context, system_prompt)
    else:
        return gpt4_conversation(actor, model, context, system_prompt)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)