use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, Write};
use std::ops::Mul;

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
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::builder().timeout(None).build()?;

    let mut history: Vec<Message> = vec![];
    history.push(Message {
        role: "system".to_string(),
        content: "You are a helpful, concise assistant.".to_string(),
        tool_calls: vec![],
        tool_name: None,
    });

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
                history.push(Message {
                    role: "user".to_string(),
                    content: input.to_string(),
                    tool_calls: vec![],
                    tool_name: None,
                });
                let mut rounds = 0;
                loop {
                    rounds += 1;
                    if rounds > 5 {
                        println!("ai> (stopped: too many tool calls)");
                        break;
                    }
                    let request = ChatRequest {
                        model: "qwen3.8:27b".to_string(),
                        messages: history.clone(),
                        tools: tools(),
                        stream: false,
                    };

                    let response: ChatResponse = client
                        .post("http://localhost:11434/api/chat")
                        .json(&request)
                        .send()?
                        .error_for_status()?
                        .json()?;

                    let msg = response.message;

                    if msg.tool_calls.is_empty() {
                        println!("ai> {}", msg.content);
                        history.push(msg);
                        break;
                    }

                    let calls = msg.tool_calls.clone();
                    history.push(msg);

                    for call in &calls {
                        println!("[tool] {}({})", call.function.name, call.function.arguments);
                        let result = run_tool(&call.function);
                        println!("[tool] result: {}", result);

                        history.push(Message {
                            role: "tool".to_string(),
                            content: result,
                            tool_calls: vec![],
                            tool_name: Some(call.function.name.clone()),
                        })
                    }
                }
            }
        }
    }
    Ok(())
}
