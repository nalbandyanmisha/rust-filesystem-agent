use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, Write};
use std::ops::Mul;
use std::path::Path;
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
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or overwrite a text file in the workspace with the given content. To change an existing file, read it first, then write the COMPLETE new content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "file path relative to the workspace, e.g. 'todo.txt'" },
                        "content": { "type": "string", "description": "the full new content of the file" }
                    },
                    "required": ["path", "content"]
                }
            }
        }
    ])
}

fn run_tool(call: &FunctionCall) -> String {
    // The real program uses the real workspace folder and asks the real user.
    // Tests call the tool functions directly with a temp folder and a fake confirm.
    let workspace = Path::new(WORKSPACE);
    let result = match call.name.as_str() {
        "multiply" => tool_multiply(&call.arguments),
        "list_files" => tool_list_files(workspace),
        "read_file" => tool_read_file(workspace, &call.arguments),
        "write_file" => tool_write_file(workspace, &call.arguments, confirm),
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

fn tool_list_files(workspace: &Path) -> Result<String, String> {
    let entries = std::fs::read_dir(workspace).map_err(|e| e.to_string())?;
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
fn tool_read_file(workspace: &Path, args: &Value) -> Result<String, String> {
    let requested = args["path"]
        .as_str()
        .ok_or(format!("read_file needs a string 'path', got {args}"))?;
    let path = safe_path(workspace, requested)?;

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

// `confirm` is a parameter: "any function that takes a question and returns yes/no".
// The real program passes the `confirm` function below (asks in the terminal);
// tests pass a closure like `|_| true` or `|_| false`.
// Inside this function, the name `confirm` means the parameter, not the global fn.
fn tool_write_file(
    workspace: &Path,
    args: &Value,
    confirm: impl Fn(&str) -> bool,
) -> Result<String, String> {
    let (Some(requested), Some(content)) = (args["path"].as_str(), args["content"].as_str()) else {
        return Err(format!(
            "write_file needs strings 'path' and 'content', got {args}"
        ));
    };

    let path = safe_write_path(workspace, requested)?;

    let action = if path.exists() { "OVERWRITE" } else { "CREATE" };
    println!("\n--- {action} {requested} ---");
    let preview: String = content.chars().take(500).collect();
    println!("{preview}");
    if content.chars().count() > 500 {
        println!("... ({} characters total)", content.chars().count());
    }
    println!("---");

    if !confirm("Allow this write?") {
        return Err("the user denied this write".to_string());
    }

    std::fs::write(&path, content).map_err(|e| format!("cannot write '{requested}': {e}"))?;

    Ok(format!(
        "wrote {} characters to '{requested}'",
        content.chars().count()
    ))
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

fn safe_path(workspace: &Path, requested: &str) -> Result<PathBuf, String> {
    let workspace = std::fs::canonicalize(workspace).map_err(|e| e.to_string())?;

    let path = std::fs::canonicalize(workspace.join(requested))
        .map_err(|e| format!("cannot open '{requested}': {e}"))?;

    if !path.starts_with(&workspace) {
        return Err("access denied: path is outside the workspace".to_string());
    }
    Ok(path)
}

fn safe_write_path(workspace: &Path, requested: &str) -> Result<PathBuf, String> {
    let workspace = std::fs::canonicalize(workspace).map_err(|e| e.to_string())?;
    let requested_path = Path::new(requested);

    let file_name = requested_path
        .file_name()
        .ok_or(format!("'{requested}' is not a valid file path"))?;
    let parent = requested_path.parent().unwrap_or(Path::new(""));
    let folder = std::fs::canonicalize(workspace.join(parent))
        .map_err(|e| format!("folder for '{requested}' does not exist: {e}"))?;

    if !folder.starts_with(&workspace) {
        return Err("access denied: path is outside the workspace".to_string());
    }

    let path = folder.join(file_name);

    if let Ok(meta) = std::fs::symlink_metadata(&path)
        && meta.file_type().is_symlink()
    {
        return Err(format!("access denied: '{requested}' is a symlink"));
    }
    Ok(path)
}

fn confirm(question: &str) -> bool {
    print!("{question} [y/N] ");
    let _ = io::stdout().flush();
    let mut answer = String::new();
    if io::stdin().read_line(&mut answer).is_err() {
        return false;
    }
    answer.trim().eq_ignore_ascii_case("y")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::builder().timeout(None).build()?;

    let mut history: Vec<Message> = vec![];
    history.push(Message::system("You are a helpful, concise assistant. You can list, read and write files in the user's workspace with your tools"));

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn setup() -> (TempDir, PathBuf) {
        let tmp = TempDir::new().unwrap();
        let ws = tmp.path().join("workspace");

        fs::create_dir(&ws).unwrap();
        fs::create_dir(ws.join("notes")).unwrap();
        fs::write(ws.join("todo.txt"), "buy milk").unwrap();
        fs::write(tmp.path().join("secret.txt"), "TOP SECRET").unwrap();

        std::os::unix::fs::symlink("../secret.txt", ws.join("link.txt")).unwrap();

        (tmp, ws)
    }

    #[test]
    fn multiply_works() {
        let result = tool_multiply(&json!({"a": 6, "b": 7}));
        assert_eq!(result, Ok("42".to_string()));
    }

    // --- safe_path (reading) ---
    #[test]
    fn read_allows_file_in_workspace() {
        let (_tmp, ws) = setup();
        assert!(safe_path(&ws, "todo.txt").is_ok());
    }

    #[test]
    fn read_blocks_parent_dir() {
        let (_tmp, ws) = setup();

        let err = safe_path(&ws, "../secret.txt").unwrap_err();
        assert!(
            err.contains("access denied: path is outside the workspace"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn read_blocks_absolute_path() {
        let (_tmp, ws) = setup();
        let err = safe_path(&ws, "/etc/hosts").unwrap_err();
        assert!(
            err.contains("access denied: path is outside the workspace"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn read_blocks_sneaky_dotdot() {
        let (_tmp, ws) = setup();
        let err = safe_path(&ws, "notes/../../secret.txt").unwrap_err();
        assert!(
            err.contains("access denied: path is outside the workspace"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn read_fails_for_missing_file() {
        let (_tmp, ws) = setup();
        let err = safe_path(&ws, "does-not-exist.txt").unwrap_err();
        assert!(
            err.contains("cannot open 'does-not-exist.txt': "),
            "wrong error: {err}"
        );
    }

    #[test]
    fn read_blocks_symlink_pointing_outside() {
        let (_tmp, ws) = setup();
        let err = safe_path(&ws, "link.txt").unwrap_err();
        assert!(
            err.contains("access denied: path is outside the workspace"),
            "wrong error: {err}"
        );
    }

    // --- safe_write_path (writing) ---
    #[test]
    fn write_allows_new_file() {
        let (_tmp, ws) = setup();
        let path = safe_write_path(&ws, "new.txt").unwrap();
        // Compare with the canonical workspace: on macOS the temp folder is under
        // /var, which is a symlink to /private/var, and the guard returns the real path.
        assert_eq!(path, fs::canonicalize(&ws).unwrap().join("new.txt"));
    }

    #[test]
    fn write_allows_new_file_in_subfolder() {
        let (_tmp, ws) = setup();
        assert!(safe_write_path(&ws, "notes/new.md").is_ok());
    }

    #[test]
    fn write_blocks_parent_dir() {
        let (_tmp, ws) = setup();
        let err = safe_write_path(&ws, "../hacked.txt").unwrap_err();
        assert!(
            err.contains("access denied: path is outside the workspace"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn write_blocks_sneaky_dotdot() {
        let (_tmp, ws) = setup();
        let err = safe_write_path(&ws, "notes/../../hacked.txt").unwrap_err();
        assert!(
            err.contains("access denied: path is outside the workspace"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn write_blocks_absolute_path() {
        let (_tmp, ws) = setup();
        let err = safe_write_path(&ws, "/tmp/hacked.txt").unwrap_err();
        assert!(
            err.contains("access denied: path is outside the workspace"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn write_rejects_empty_path() {
        let (_tmp, ws) = setup();

        let err = safe_write_path(&ws, "").unwrap_err();
        assert!(
            err.contains("is not a valid file path"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn write_rejects_dotdot() {
        let (_tmp, ws) = setup();

        let err = safe_write_path(&ws, "..").unwrap_err();
        assert!(
            err.contains("is not a valid file path"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn write_blocks_symlink() {
        let (_tmp, ws) = setup();
        let err = safe_write_path(&ws, "link.txt").unwrap_err();
        assert!(
            err.contains("access denied: 'link.txt' is a symlink"),
            "wrong error: {err}"
        );
    }

    // --- the tools themselves ---
    #[test]
    fn list_files_shows_files_and_marks_folders() {
        let (_tmp, ws) = setup();
        let result = tool_list_files(&ws).unwrap();
        assert!(result.contains("todo.txt"), "got: {result}");
        assert!(result.contains("notes/"), "folders should end with '/', got: {result}");
    }

    #[test]
    fn read_file_returns_content() {
        let (_tmp, ws) = setup();
        let result = tool_read_file(&ws, &json!({"path": "todo.txt"}));
        assert_eq!(result, Ok("buy milk".to_string()));
    }

    #[test]
    fn write_does_nothing_when_user_denies() {
        let (_tmp, ws) = setup();
        let args = json!({"path": "todo.txt", "content": "hacked"});

        // |_| false = a closure that ignores the question and always answers "no"
        let err = tool_write_file(&ws, &args, |_| false).unwrap_err();

        assert!(err.contains("denied"), "wrong error: {err}");
        assert_eq!(fs::read_to_string(ws.join("todo.txt")).unwrap(), "buy milk");
    }

    #[test]
    fn write_changes_file_when_user_allows() {
        let (_tmp, ws) = setup();
        let args = json!({"path": "todo.txt", "content": "buy milk\nbuy bread"});

        let result = tool_write_file(&ws, &args, |_| true);

        assert!(result.is_ok(), "got: {result:?}");
        assert_eq!(
            fs::read_to_string(ws.join("todo.txt")).unwrap(),
            "buy milk\nbuy bread"
        );
    }

    #[test]
    fn write_attack_is_blocked_before_asking_user() {
        let (_tmp, ws) = setup();
        let args = json!({"path": "../secret.txt", "content": "hacked"});

        // If the guard works, confirm is never called. If it IS called, this
        // closure panics, and a panic makes the test fail.
        let err = tool_write_file(&ws, &args, |_| panic!("confirm must not be called")).unwrap_err();

        assert!(err.contains("access denied"), "wrong error: {err}");
        let secret = fs::read_to_string(ws.parent().unwrap().join("secret.txt")).unwrap();
        assert_eq!(secret, "TOP SECRET");
    }
}
