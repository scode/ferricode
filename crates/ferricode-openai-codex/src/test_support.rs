//! Shares test-only fixtures for the Codex implementation modules.
//!
//! These helpers simulate local HTTP peers and construct credential fixtures for
//! tests. They are compiled only for this crate's tests and must not become
//! production auth, network, or storage behavior.

use crate::{AuthFile, OpenAiCodexAuth, TokenSet};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde_json::json;
use std::io::Write;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Clone)]
pub(crate) struct TestResponse {
    status: &'static str,
    content_type: &'static str,
    body: String,
}

impl TestResponse {
    pub(crate) fn new(status: &'static str, body: &str) -> Self {
        Self {
            status,
            content_type: "text/plain",
            body: body.to_string(),
        }
    }

    pub(crate) fn json(body: &str) -> Self {
        Self {
            status: "200 OK",
            content_type: "application/json",
            body: body.to_string(),
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct SharedOutput(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl SharedOutput {
    pub(crate) fn bytes(&self) -> Vec<u8> {
        self.0.lock().unwrap().clone()
    }
}

impl Write for SharedOutput {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(crate) async fn wait_for_state(output: &SharedOutput) -> String {
    for _ in 0..100 {
        let text = String::from_utf8(output.bytes()).unwrap();
        if let Some(url) = text.lines().find(|line| line.starts_with("http")) {
            let parsed = url::Url::parse(url).unwrap();
            return parsed
                .query_pairs()
                .find_map(|(key, value)| (key == "state").then(|| value.into_owned()))
                .unwrap();
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("authorize URL was not written");
}

pub(crate) async fn spawn_test_server(
    responses: Vec<TestResponse>,
) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
    use std::sync::{Arc, Mutex};

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&requests);

    tokio::spawn(async move {
        for response in responses {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_request(&mut stream).await;
            captured
                .lock()
                .unwrap()
                .push(String::from_utf8_lossy(&request).to_string());
            write_response(&mut stream, response).await;
        }
    });

    (format!("http://{addr}"), requests)
}

pub(crate) async fn send_get(addr: std::net::SocketAddr, target: &str) -> String {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let raw = format!("GET {target} HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\n\r\n");
    stream.write_all(raw.as_bytes()).await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    String::from_utf8(response).unwrap()
}

pub(crate) async fn write_response(stream: &mut TcpStream, response: TestResponse) {
    let raw = format!(
        "HTTP/1.1 {}\r\ncontent-type: {}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        response.status,
        response.content_type,
        response.body.len(),
        response.body
    );
    stream.write_all(raw.as_bytes()).await.unwrap();
}

pub(crate) async fn read_request(stream: &mut TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let mut buffer = [0; 1024];
    loop {
        let bytes = stream.read(&mut buffer).await.unwrap();
        if bytes == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..bytes]);
        if request_is_complete(&request) {
            break;
        }
    }
    request
}

fn request_is_complete(request: &[u8]) -> bool {
    let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
        return false;
    };
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);

    request.len() >= header_end + 4 + content_length
}

pub(crate) fn auth_with_tokens(
    access_token: &str,
    refresh_token: &str,
    expires_at_unix_ms: u64,
) -> AuthFile {
    AuthFile {
        openai_codex: Some(OpenAiCodexAuth {
            tokens: Some(TokenSet {
                access_token: access_token.to_string(),
                refresh_token: refresh_token.to_string(),
                id_token: id_token("acct", "plus"),
                expires_at_unix_ms,
                chatgpt_account_id: "acct".to_string(),
                chatgpt_plan_type: Some("plus".to_string()),
            }),
        }),
    }
}

pub(crate) fn token_json(access: &str, refresh: &str, id_token: &str, expires_in: u64) -> String {
    json!({
        "access_token": access,
        "refresh_token": refresh,
        "id_token": id_token,
        "expires_in": expires_in
    })
    .to_string()
}

pub(crate) fn token_json_without_refresh(access: &str, id_token: &str, expires_in: u64) -> String {
    json!({
        "access_token": access,
        "id_token": id_token,
        "expires_in": expires_in
    })
    .to_string()
}

pub(crate) fn id_token(account_id: &str, plan: &str) -> String {
    let claims = json!({
        "https://api.openai.com/auth": {
            "chatgpt_account_id": account_id,
            "chatgpt_plan_type": plan
        }
    });
    format!(
        "header.{}.signature",
        URL_SAFE_NO_PAD.encode(claims.to_string())
    )
}
