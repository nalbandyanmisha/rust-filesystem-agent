use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, Write};
use std::ops::Mul;

const MODEL: &str = "qwen3.8:27b";
const OLLAMA_URL: &str = "http://localhost:11434/api/chat";
const MAX_TOOL_ROUNDS: usize = 5;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Message {
    role: String,
    #[serde(default)]
    content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_name: Option<String>,
}

impl Message {
    fn system(content: &str) -> Self {
        Message {
            role: "system".to_string(),
            content: content.to_string(),
            tool_calls: vec![],
            tool_name: None,
        }
    }

    fn user(content: &str) -> Self {
        Message {
            role: "user".to_string(),
            content: content.to_string(),
            tool_calls: vec![],
            tool_name: None,
        }
    }

    fn tool(name: &str, result: String) -> Self {
        Message {
            role: "tool".to_string(),
            content: result,
            tool_calls: vec![],
            tool_name: Some(name.to_string()),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ToolCall {
    function: FunctionCall,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct FunctionCall {
    name: String,
    arguments: Value,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: String,
    messages: &'a [Message],
    tools: Value,
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: Message,
}

fn tools() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two numbers exactly. Use this for any multiplication.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "first number" },
                        "b": {
                            "type": "number",
                            "description": "second number"
                        }
                    },
                    "required": ["a", "b"]
                }
            }
        }
    ])
}

fn run_tool(call: &FunctionCall) -> String {
    match call.name.as_str() {
        "multiply" => {
            let (Some(a), Some(b)) = (call.arguments["a"].as_f64(), call.arguments["b"].as_f64())
            else {
                return format!(
                    "error: multiply needs numbers 'a' and 'b', got {}",
                    call.arguments
                );
            };
            multiply(a, b).to_string()
        }
        _ => format!("unknown tool: {}", call.name),
    }
}

fn multiply<T>(a: T, b: T) -> T
where
    T: Mul<Output = T>,
{
    a * b
}

fn send(
    client: &reqwest::blocking::Client,
    history: &[Message],
) -> Result<Message, Box<dyn std::error::Error>> {
    let request = ChatRequest {
        model: MODEL.to_string(),
        messages: history,
        tools: tools(),
        stream: false,
    };

    let response: ChatResponse = client
        .post(OLLAMA_URL)
        .json(&request)
        .send()?
        .error_for_status()?
        .json()?;

    Ok(response.message)
}

fn run_tool_calls(calls: &[ToolCall], history: &mut Vec<Message>) {
    for call in calls {
        println!("[tool] {}({})", call.function.name, call.function.arguments);
        let result = run_tool(&call.function);
        println!("[tool] result: {}", result);

        history.push(Message::tool(&call.function.name, result));
    }
}

fn agent_turn(
    client: &reqwest::blocking::Client,
    history: &mut Vec<Message>,
) -> Result<(), Box<dyn std::error::Error>> {
    for _ in 0..MAX_TOOL_ROUNDS {
        let msg = send(client, history)?;

        if msg.tool_calls.is_empty() {
            println!("ai> {}", msg.content);
            history.push(msg);
            return Ok(());
        }

        let calls = msg.tool_calls.clone();
        history.push(msg);
        run_tool_calls(&calls, history);
    }
    println!("ai> (stopped: too many tool calls)");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::builder().timeout(None).build()?;

    let mut history: Vec<Message> = vec![];
    history.push(Message::system("You are a helpful, concise assistant."));

    loop {
        print!("you> ");
        io::stdout().flush()?;
        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            println!();
            break;
        }
        let input = input.trim();

        match input {
            "exit" => break,
            "" => continue,
            _ => {
                let len_before = history.len();
                history.push(Message::user(input));
                if let Err(e) = agent_turn(&client, &mut history) {
                    println!("error: {e}");
                    history.truncate(len_before);
                }
            }
        }
    }
    Ok(())
}
