# Built-In Tools

Ferricode tools are local capabilities that the harness can run for the model. They are not MCP tools, they do not come
from OpenAI, and they are not a plugin system. Provider crates expose the tool schemas in whatever wire format a model
backend expects, but `ferricode-core` owns execution and policy.

The important bit is the split of responsibility: the model may ask for a tool call, but the harness decides what that
tool is allowed to do. A provider should pass tool calls through, preserve the provider call id, and send the resulting
tool output back to the same model interaction. It should not read files itself or make provider-specific filesystem
policy decisions.

## Names

Built-in tool names use the `ferricode_` prefix because provider APIs commonly put tools into one flat function
namespace. The prefix makes these names clearly owned by Ferricode and leaves room for provider-native tools, MCP tools,
or future extension points without accidental collisions.

Tool names are part of the provider-facing contract. Renaming one is a model-integration change, not just an internal
Rust refactor.

The current built-in tool names are:

- `ferricode_list_directory`
- `ferricode_read_file`

## Arguments

Tool arguments are JSON strings supplied by the model and parsed by the harness. The current filesystem tools use this
shape:

```json
{ "path": "relative/path" }
```

Paths are relative to the request working directory. For `ferric run`, that is the `--cwd` value, or `.` when no `--cwd`
is provided.

Ferricode also caps the size of one tool call before execution. The provider call id and tool name must each fit within
256 bytes, and the argument string must fit within 16 KiB. Calls outside those limits are returned to the model as tool
errors instead of being executed.

## Execution Loop

One request can take multiple model turns. The provider starts the interaction and either returns final assistant text
or returns tool calls. Ferricode executes those calls, returns structured tool outputs to the provider, and repeats
until the provider returns final text.

Ferricode allows at most 32 consecutive tool-call turns for one request. If the model keeps asking for tools after that,
the command fails instead of looping forever.

Ferricode also allows at most 16 tool calls in one model turn. If a model asks for more than that, each requested call
gets a structured error result for that turn. The provider still has to return outputs keyed by the original call ids so
the backend can reconcile the failed calls with its transcript.

Tool-level failures are returned to the model as structured tool output with `"ok": false`. That lets the model recover
from ordinary problems such as a missing file, a rejected path, or unsupported input. Provider, authentication, network,
and protocol failures still fail the command because the model cannot repair them by choosing different tool arguments.

## Filesystem Policy

The first built-in tools are read-only filesystem tools: one directory-listing tool and one UTF-8 file-read tool. They
are intentionally narrow. There is no write, delete, shell, glob, recursive tree, or search tool in this version.

Filesystem paths are confined to the request working directory. Ferricode rejects absolute paths, empty paths, `..`
traversal, and symlink targets that resolve outside the working directory.

Directory listings are sorted by name, include hidden entries, and are capped at 200 entries. A listing entry includes a
name, a type, and a size for regular files.

File reads return up to 64 KiB of valid UTF-8 text. If a file is larger, the output includes `"truncated": true` and the
returned content stops at a valid UTF-8 boundary. Binary data and invalid UTF-8 in the returned read window are reported
as tool errors instead of being passed to the model as lossy text or base64.
