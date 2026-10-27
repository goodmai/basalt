enum RejectionGolden {
    static let json = #"""
{
 "chat-bad-format": {
  "body": {
   "error": {
    "message": "response_format json_schema needs a schema",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "chat-bad-temperature": {
  "body": {
   "error": {
    "message": "Invalid request: temperature must be in [0.0, 2.0]",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "chat-empty": {
  "body": {
   "error": {
    "message": "messages must not be empty",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "chat-not-json": {
  "body": {
   "error": {
    "message": "Cannot decode request: The data couldn’t be read because it isn’t in the correct format.",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "chat-path-model": {
  "body": {
   "error": {
    "message": "\"model\" must be a model id, not a filesystem path.",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "chat-undeclared-tool": {
  "body": {
   "error": {
    "message": "tool_choice names b, which is not in tools",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "chat-wrong-type": {
  "body": {
   "error": {
    "message": "Cannot decode request: The data couldn’t be read because it isn’t in the correct format.",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "messages-bad-temperature": {
  "body": {
   "error": {
    "message": "Invalid request: temperature must be in [0.0, 2.0]",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "messages-bad-thinking": {
  "body": {
   "error": {
    "message": "thinking.budget_tokens must be >= 1 and less than max_tokens (64)",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "messages-empty": {
  "body": {
   "error": {
    "message": "messages must not be empty",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "messages-not-json": {
  "body": {
   "error": {
    "message": "Decode error: The data couldn’t be read because it isn’t in the correct format.",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "messages-path-model": {
  "body": {
   "error": {
    "message": "\"model\" must be a model id, not a filesystem path.",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "messages-undeclared-tool": {
  "body": {
   "error": {
    "message": "tool_choice names b, which is not in tools",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "messages-wrong-type": {
  "body": {
   "error": {
    "message": "Decode error: The data couldn’t be read because it isn’t in the correct format.",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "responses-bad-format": {
  "body": {
   "error": {
    "message": "response_format json_schema needs a schema",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "responses-bad-temperature": {
  "body": {
   "error": {
    "message": "Invalid request: temperature must be in [0.0, 2.0]",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "responses-empty": {
  "body": {
   "error": {
    "message": "input or messages must not be empty",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "responses-not-json": {
  "body": {
   "error": {
    "message": "Cannot decode responses request: The data couldn’t be read because it isn’t in the correct format.",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "responses-path-model": {
  "body": {
   "error": {
    "message": "\"model\" must be a model id, not a filesystem path.",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "responses-undeclared-tool": {
  "body": {
   "error": {
    "message": "tool_choice names b, which is not in tools",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 },
 "responses-wrong-type": {
  "body": {
   "error": {
    "message": "Cannot decode responses request: The data couldn’t be read because it isn’t in the correct format.",
    "type": "invalid_request_error"
   },
   "type": "error"
  },
  "contentType": "application/json",
  "status": 400
 }
}
"""#
}
