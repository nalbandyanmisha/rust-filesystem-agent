use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, Write};
use std::ops::Mul;
use std::path::PathBuf;

const MODEL: &str = "qwen3.8:27b";
const OLLAMA_URL: &str = "http://localhost:11434/api/chat";
const MAX_TOOL_ROUNDS: usize = 5;
const WORKSPACE: &str = "workspace";
const MAX_FILE_CHARS: usize = 10_000;

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
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List all files in the user's workspace folder.",
                "parameters": { "type": "object", "properties": {}, "required": [] }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file from the workspace. Use list_files first to see what exists.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "file path relative to the workspace, e.g. 'todo.txt'" }
                    },
                    "required": ["path"]
                }
            }
        }
    ])
}

fn run_tool(call: &FunctionCall) -> String {
    let result = match call.name.as_str() {
        "multiply" => tool_multiply(&call.arguments),
        "list_files" => tool_list_files(),
        "read_file" => tool_read_file(&call.arguments),
        _ => Err(format!("unknown tool: {}", call.name)),
    };
    match result {
        Ok(output) => output,
        Err(e) => format!("error: {e}"),
    }
}

fn tool_multiply(args: &Value) -> Result<String, String> {
    let (Some(a), Some(b)) = (args["a"].as_f64(), args["b"].as_f64()) else {
        return Err(format!("multiply needs numbers 'a' and 'b', got {}", args));
    };
    Ok(multiply(a, b).to_string())
}

fn multiply<T>(a: T, b: T) -> T
where
    T: Mul<Output = T>,
{
    a * b
}

fn tool_list_files() -> Result<String, String> {
    let entries = std::fs::read_dir(WORKSPACE).map_err(|e| e.to_string())?;
    let mut names: Vec<String> = Vec::new();

    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let mut name = entry.file_name().to_string_lossy().to_string();
        let file_type = entry.file_type().map_err(|e| e.to_string())?;
        if file_type.is_dir() {
            name.push('/');
        }
        names.push(name);
    }

    names.sort();

    if names.is_empty() {
        return Ok("the workspace is empty".to_string());
    }

    Ok(names.join("\n"))
}
fn tool_read_file(args: &Value) -> Result<String, String> {
    let requested = args["path"]
        .as_str()
        .ok_or(format!("read_file needs a string 'path', got {args}"))?;
    let path = safe_path(requested)?;

    if !path.is_file() {
        return Err(format!(
            "'{requested}' is not a file (use list_files to see files)"
        ));
    }

    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read '{requested}' as text: {e}"))?;
    if content.chars().count() > MAX_FILE_CHARS {
        let truncated: String = content.chars().take(MAX_FILE_CHARS).collect();
        return Ok(format!(
            "{truncated}\n[truncated: file is longer than {MAX_FILE_CHARS} characters]"
        ));
    }

    Ok(content)
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

fn safe_path(requested: &str) -> Result<PathBuf, String> {
    let workspace = std::fs::canonicalize(WORKSPACE).map_err(|e| e.to_string())?;

    let path = std::fs::canonicalize(workspace.join(requested))
        .map_err(|e| format!("cannot open '{requested}': {e}"))?;

    if !path.starts_with(&workspace) {
        return Err("access denied: path is outside the workspace".to_string());
    }
    Ok(path)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::builder().timeout(None).build()?;

    let mut history: Vec<Message> = vec![];
    history.push(Message::system("You are a helpful, concise assistant. You can read files in the user's workspace with your tools"));

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
