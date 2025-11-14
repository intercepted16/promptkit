const express = require("express");
const { getRandomContents } = require("../utils/randomContents");
const { tokenize } = require("../utils/tokenize");
const delay = require("../utils/delay");
const { requestCounter, requestLatency, payloadSize } = require("../utils/metrics");

const router = express.Router();

/** Generate a single mock tool call */
function generateSingleToolCall(tool, index) {
  const callId = `call_${Math.random().toString(36).substring(2, 15)}${Math.random().toString(36).substring(2, 15)}`;
  if (tool.type === "function") {
    const mockArgs = generateMockArguments(tool.function.parameters);
    return {
      id: callId,
      type: "function",
      function: {
        name: tool.function.name,
        arguments: JSON.stringify(mockArgs),
      },
    };
  }
  return null;
}

/** Generate mock arguments from JSON schema */
function generateMockArguments(parameters) {
  if (!parameters || !parameters.properties) return {};
  const args = {};
  for (const [key, schema] of Object.entries(parameters.properties)) {
    if (schema.type === "string") args[key] = schema.enum ? schema.enum[0] : "mock_value";
    else if (schema.type === "integer" || schema.type === "number") args[key] = schema.enum ? schema.enum[0] : 1;
    else if (schema.type === "boolean") args[key] = true;
    else if (schema.type === "array") args[key] = [];
    else if (schema.type === "object") args[key] = {};
  }
  return args;
}

/** Detect if tool results already provided */
function hasToolResults(messages) {
  return messages.some((m) => m.role === "tool");
}

/** Detect if assistant already issued a tool call */
function lastAssistantAskedForTools(messages) {
  const last = messages[messages.length - 1];
  return last?.role === "assistant" && Array.isArray(last.tool_calls) && last.tool_calls.length > 0;
}

/** Always call tools when permitted */
function generateToolCalls(tools) {
  if (!Array.isArray(tools) || tools.length === 0) return null;
  return [generateSingleToolCall(tools[0], 0)];
}

router.post("/v1/chat/completions", async (req, res) => {
  const start = Date.now();
  const delayHeader = req.headers["x-set-response-delay-ms"];
  const delayTime = parseInt(delayHeader) || parseInt(process.env.RESPONSE_DELAY_MS) || 0;
  await delay(delayTime);

  const {
    messages,
    stream,
    mockType = process.env.MOCK_TYPE || "random",
    mockFixedContents,
    model,
    tools,
    tool_choice = "auto",
  } = req.body;

  const randomResponses = getRandomContents();

  if (!messages || !Array.isArray(messages)) {
    requestCounter.inc({ method: "POST", path: "/v1/chat/completions", status: 400 });
    return res.status(400).json({ error: 'Missing or invalid "messages" in request body' });
  }

  if (stream !== undefined && typeof stream !== "boolean") {
    return res.status(400).json({ error: 'Invalid "stream" in request body' });
  }

  // Basic mock content
  let content;
  switch (mockType) {
    case "echo":
      content = messages[messages.length - 1].content;
      break;
    case "random":
      content = randomResponses[Math.floor(Math.random() * randomResponses.length)];
      break;
    case "fixed":
      content = mockFixedContents;
      break;
  }

  // Handle tool result (end of tool loop)
  const last = messages[messages.length - 1];
  if (last?.role === "tool") {
    return res.json({
      id: "chatcmpl-" + Math.random().toString(36).slice(2),
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [
        {
          index: 0,
          finish_reason: "stop",
          message: {
            role: "assistant",
            content: `✅ Tool ${last.name || last.tool_call_id || "unknown"} finished. Result: ${last.content}`,
          },
        },
      ],
    });
  }

  // Only create tool calls when appropriate
  const shouldGenerateToolCalls =
    Array.isArray(tools) &&
    tools.length > 0 &&
    !hasToolResults(messages) &&
    !lastAssistantAskedForTools(messages);

  const toolCalls = shouldGenerateToolCalls ? generateToolCalls(tools) : null;

  // Handle non-streaming response
  const response = {
    id: "chatcmpl-" + Math.random().toString(36).slice(2),
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    usage: { prompt_tokens: 10, completion_tokens: 50, total_tokens: 60 },
    choices: [
      {
        index: 0,
        finish_reason: toolCalls ? "tool_calls" : "stop",
        message: {
          role: "assistant",
          content: toolCalls ? null : content,
          ...(toolCalls ? { tool_calls: toolCalls } : {}),
        },
      },
    ],
  };

  requestCounter.inc({ method: "POST", path: "/v1/chat/completions", status: 200 });
  requestLatency.observe({ method: "POST", path: "/v1/chat/completions", status: 200 }, Date.now() - start);
  payloadSize.observe({ method: "POST", path: "/v1/chat/completions", status: 200 }, req.socket.bytesRead);

  // Stream mode
  if (stream) {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    if (toolCalls) {
      // Stream tool call delta chunks
      const toolCall = toolCalls[0];
      res.write(`data: ${JSON.stringify({
        id: response.id,
        object: "chat.completion.chunk",
        created: Date.now(),
        model,
        choices: [{ index: 0, delta: { role: "assistant", tool_calls: [{ id: toolCall.id, type: toolCall.type, function: { name: toolCall.function.name, arguments: "" } }] }, finish_reason: null }],
      })}\n\n`);
      res.write(`data: ${JSON.stringify({
        id: response.id,
        object: "chat.completion.chunk",
        created: Date.now(),
        model,
        choices: [{ index: 0, delta: { tool_calls: [{ function: { arguments: toolCall.function.arguments } }] }, finish_reason: null }],
      })}\n\n`);
      res.write(`data: ${JSON.stringify({
        id: response.id,
        object: "chat.completion.chunk",
        created: Date.now(),
        model,
        choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
      })}\n\n`);
      res.write("data: [DONE]\n\n");
      res.end();
    } else {
      // Stream text chunks
      const tokens = tokenize(content);
      for (const token of tokens) {
        res.write(`data: ${JSON.stringify({
          id: response.id,
          object: "chat.completion.chunk",
          created: Date.now(),
          model,
          choices: [{ index: 0, delta: { role: "assistant", content: token }, finish_reason: null }],
        })}\n\n`);
      }
      res.write(`data: ${JSON.stringify({
        id: response.id,
        object: "chat.completion.chunk",
        created: Date.now(),
        model,
        choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      })}\n\n`);
      res.write("data: [DONE]\n\n");
      res.end();
    }
  } else {
    res.json(response);
  }
});

module.exports = router;
