import { Buffer } from 'node:buffer';

export const config = {
  runtime: 'edge', // 使用 Edge Runtime 以支持流式传输
};

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";
const API_CLIENT = "genai-js/0.21.0";
const DEFAULT_MODEL = "gemini-1.5-pro-latest";
const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";

// 允许跨域的配置
const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*", // 建议生产环境改为你的前端域名
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

// -----------------------------------------------------------------------------
// 核心处理函数
// -----------------------------------------------------------------------------

export default async function handler(request) {
  if (request.method === "OPTIONS") {
    return new Response(null, { headers: CORS_HEADERS });
  }

  const url = new URL(request.url);
  const pathname = url.pathname;

  // 简单的错误处理
  const errHandler = (err) => {
    console.error(err);
    return new Response(
      JSON.stringify({ error: { message: err.message, type: err.name } }), 
      fixCors({ status: err.status ?? 500, headers: { "Content-Type": "application/json" } })
    );
  };

  try {
    const auth = request.headers.get("Authorization");
    const apiKey = auth?.split(" ")[1];

    // 如果是 Vercel 部署，静态文件会自动由 public/ 目录托管，不需要在这里处理
    // 这里只处理 API 路由

    // 路由匹配逻辑 (兼容 OpenAI 路径格式)
    if (pathname.endsWith("/chat/completions")) {
      if (request.method !== "POST") throw new HttpError("Method not allowed", 405);
      const body = await request.json();
      return await handleCompletions(body, apiKey).catch(errHandler);
    } 
    
    if (pathname.endsWith("/embeddings")) {
      if (request.method !== "POST") throw new HttpError("Method not allowed", 405);
      const body = await request.json();
      return await handleEmbeddings(body, apiKey).catch(errHandler);
    }
    
    if (pathname.endsWith("/models")) {
      if (request.method !== "GET") throw new HttpError("Method not allowed", 405);
      return await handleModels(apiKey).catch(errHandler);
    }

    // 默认响应
    return new Response(JSON.stringify({ message: "Gemini Proxy running on Vercel" }), {
      status: 200,
      headers: { ...CORS_HEADERS, "Content-Type": "application/json" }
    });

  } catch (err) {
    return errHandler(err);
  }
}

// -----------------------------------------------------------------------------
// 业务逻辑 (从原 Workers 代码移植并清理)
// -----------------------------------------------------------------------------

class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "HttpError";
    this.status = status;
  }
}

function fixCors(responseOptions) {
  const headers = new Headers(responseOptions.headers);
  Object.entries(CORS_HEADERS).forEach(([k, v]) => headers.set(k, v));
  return { ...responseOptions, headers };
}

function makeHeaders(apiKey, more) {
  return {
    "x-goog-api-client": API_CLIENT,
    ...(apiKey && { "x-goog-api-key": apiKey }),
    ...more,
  };
}

async function handleModels(apiKey) {
  const response = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
    headers: makeHeaders(apiKey),
  });
  let body = await response.text();
  if (response.ok) {
    const { models } = JSON.parse(body);
    body = JSON.stringify({
      object: "list",
      data: models.map(({ name }) => ({
        id: name.replace("models/", ""),
        object: "model",
        created: 0,
        owned_by: "google",
      })),
    }, null, 2);
  }
  return new Response(body, fixCors({ status: response.status, headers: { "Content-Type": "application/json" } }));
}

async function handleEmbeddings(req, apiKey) {
  if (typeof req.model !== "string") {
    throw new HttpError("model is not specified", 400);
  }
  if (!Array.isArray(req.input)) {
    req.input = [req.input];
  }
  let model;
  if (req.model.startsWith("models/")) {
    model = req.model;
  } else {
    req.model = DEFAULT_EMBEDDINGS_MODEL;
    model = "models/" + req.model;
  }
  const response = await fetch(`${BASE_URL}/${API_VERSION}/${model}:batchEmbedContents`, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify({
      requests: req.input.map((text) => ({
        model,
        content: { parts: { text } },
        outputDimensionality: req.dimensions,
      })),
    }),
  });
  
  let body = await response.text();
  if (response.ok) {
    const { embeddings } = JSON.parse(body);
    body = JSON.stringify({
      object: "list",
      data: embeddings.map(({ values }, index) => ({
        object: "embedding",
        index,
        embedding: values,
      })),
      model: req.model,
    }, null, 2);
  }
  return new Response(body, fixCors({ status: response.status, headers: { "Content-Type": "application/json" } }));
}

async function handleCompletions(req, apiKey) {
  let model = DEFAULT_MODEL;
  if (req.model && typeof req.model === "string") {
    if (req.model.startsWith("models/")) {
      model = req.model.substring(7);
    } else if (req.model.startsWith("gemini-") || req.model.startsWith("learnlm-")) {
      model = req.model;
    }
  }

  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (req.stream) {
    url += "?alt=sse";
  }

  const transformedBody = await transformRequest(req);
  
  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(transformedBody),
  });

  if (!response.ok) {
    return new Response(response.body, fixCors({ status: response.status, statusText: response.statusText }));
  }

  let body = response.body;
  const id = generateChatcmplId();

  if (req.stream) {
    // 建立流处理管道
    const stream = response.body
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(new TransformStream(createParseStreamTransformer()))
      .pipeThrough(new TransformStream(createToOpenAiStreamTransformer(req.stream_options?.include_usage, model, id)))
      .pipeThrough(new TextEncoderStream());
      
    return new Response(stream, fixCors({ 
      headers: { 
        "Content-Type": "text/event-stream", 
        "Cache-Control": "no-cache", 
        "Connection": "keep-alive" 
      } 
    }));
  } else {
    const text = await response.text();
    const json = JSON.parse(text);
    const processed = processCompletionsResponse(json, model, id);
    return new Response(processed, fixCors({ headers: { "Content-Type": "application/json" } }));
  }
}

// -----------------------------------------------------------------------------
// 辅助函数与转换逻辑
// -----------------------------------------------------------------------------

const fieldsMap = {
  stop: "stopSequences",
  n: "candidateCount",
  max_tokens: "maxOutputTokens",
  max_completion_tokens: "maxOutputTokens",
  temperature: "temperature",
  top_p: "topP",
  top_k: "topK",
  frequency_penalty: "frequencyPenalty",
  presence_penalty: "presencePenalty",
};

const harmCategory = [
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
  "HARM_CATEGORY_HARASSMENT",
  "HARM_CATEGORY_CIVIC_INTEGRITY",
];
const safetySettings = harmCategory.map((category) => ({
  category,
  threshold: "BLOCK_NONE",
}));

function transformConfig(req) {
  let cfg = {};
  for (let key in req) {
    const matchedKey = fieldsMap[key];
    if (matchedKey) {
      cfg[matchedKey] = req[key];
    }
  }
  if (req.response_format) {
      // 简单处理 response_format
      if(req.response_format.type === 'json_object') {
          cfg.responseMimeType = "application/json";
      }
      // 其他复杂 schema 处理省略，可按需添加
  }
  return cfg;
}

async function parseImg(url) {
  let mimeType, data;
  if (url.startsWith("http://") || url.startsWith("https://")) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch image: ${url}`);
    mimeType = response.headers.get("content-type");
    const arrayBuffer = await response.arrayBuffer();
    data = Buffer.from(arrayBuffer).toString("base64");
  } else {
    const match = url.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
    if (!match) throw new Error("Invalid image data");
    ({ mimeType, data } = match.groups);
  }
  return { inlineData: { mimeType, data } };
}

async function transformMsg({ role, content }) {
  const parts = [];
  if (!Array.isArray(content)) {
    parts.push({ text: content });
  } else {
    for (const item of content) {
      if (item.type === "text") {
        parts.push({ text: item.text });
      } else if (item.type === "image_url") {
        parts.push(await parseImg(item.image_url.url));
      }
    }
  }
  return { role, parts };
}

async function transformRequest(req) {
  const contents = [];
  let system_instruction;
  
  if (req.messages) {
    for (const item of req.messages) {
      if (item.role === "system") {
        const sysMsg = await transformMsg(item);
        // Gemini system instruction 格式不同
        system_instruction = { parts: sysMsg.parts };
      } else {
        const transformed = await transformMsg(item);
        transformed.role = transformed.role === "assistant" ? "model" : "user";
        contents.push(transformed);
      }
    }
  }

  // Gemini 要求如果不含 system_instruction，contents 不能为空且格式正确
  // 这里做一个简单的防御性检查
  if (contents.length === 0 && system_instruction) {
      contents.push({ role: "model", parts: [{ text: " " }] }); 
  }

  return {
    contents,
    system_instruction,
    safetySettings,
    generationConfig: transformConfig(req),
  };
}

function generateChatcmplId() {
  return "chatcmpl-" + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
}

// -----------------------------------------------------------------------------
// 流式处理 Transformers
// -----------------------------------------------------------------------------

function createParseStreamTransformer() {
  let buffer = "";
  return {
    transform(chunk, controller) {
      if (!chunk) return;
      buffer += chunk;
      const responseLineRE = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;
      
      while (true) {
        const match = buffer.match(responseLineRE);
        if (!match) break;
        controller.enqueue(match[1]);
        buffer = buffer.substring(match[0].length);
      }
    },
    flush(controller) {
      if (buffer) { 
        // 忽略无法解析的残留数据
        // console.warn("Leftover buffer:", buffer); 
      }
    }
  };
}

function createToOpenAiStreamTransformer(includeUsage, model, id) {
  let lastData = null;
  
  // 转换单条数据
  const transformResponseStream = (data, stop, first) => {
    const candidate = data.candidates?.[0];
    const content = candidate?.content?.parts?.[0]?.text || "";
    const finishReason = candidate?.finishReason;
    
    // 映射 Gemini finishReason 到 OpenAI
    const reasonMap = {
      "STOP": "stop",
      "MAX_TOKENS": "length",
      "SAFETY": "content_filter",
      "RECITATION": "content_filter"
    };

    const choice = {
      index: candidate?.index || 0,
      delta: {},
      finish_reason: null
    };

    if (stop) {
       choice.delta = {};
       choice.finish_reason = reasonMap[finishReason] || finishReason || "stop";
    } else {
       if (first) {
         choice.delta = { role: "assistant", content: "" };
       } else {
         choice.delta = { content: content };
       }
    }

    const output = {
      id: id,
      object: "chat.completion.chunk",
      created: Math.floor(Date.now() / 1e3),
      model: model,
      choices: [choice]
    };

    if (data.usageMetadata && includeUsage && stop) {
      output.usage = {
        prompt_tokens: data.usageMetadata.promptTokenCount,
        completion_tokens: data.usageMetadata.candidatesTokenCount,
        total_tokens: data.usageMetadata.totalTokenCount
      };
    }

    return "data: " + JSON.stringify(output) + "\n\n";
  };

  return {
    async transform(line, controller) {
      if (!line) return;
      let data;
      try {
        data = JSON.parse(line);
      } catch (e) {
        // 忽略解析错误，如心跳包
        return;
      }
      
      if (!data.candidates || data.candidates.length === 0) return;

      const cand = data.candidates[0];
      const isFirst = lastData === null;
      
      if (isFirst) {
        controller.enqueue(transformResponseStream(data, false, true));
      }
      
      // 发送内容
      if (cand.content) {
        controller.enqueue(transformResponseStream(data, false, false));
      }
      
      lastData = data;
    },
    flush(controller) {
      if (lastData) {
        controller.enqueue(transformResponseStream(lastData, true, false));
        controller.enqueue("data: [DONE]\n\n");
      }
    }
  };
}

function processCompletionsResponse(data, model, id) {
  const cand = data.candidates?.[0];
  return JSON.stringify({
    id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1e3),
    model,
    choices: [{
      index: cand?.index || 0,
      message: {
        role: "assistant",
        content: cand?.content?.parts?.map(p => p.text).join("") || ""
      },
      finish_reason: cand?.finishReason === "STOP" ? "stop" : cand?.finishReason
    }],
    usage: data.usageMetadata ? {
      prompt_tokens: data.usageMetadata.promptTokenCount,
      completion_tokens: data.usageMetadata.candidatesTokenCount,
      total_tokens: data.usageMetadata.totalTokenCount
    } : null
  });
}
