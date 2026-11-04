enum TemplateGoldens {
    static let byRepo: [String: String] = [
        "mlx-community/gemma-4-12B-it-qat-4bit": gemma4,
        "mlx-community/gemma-4-26B-A4B-it-qat-4bit": gemma4,
        "mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit": nemotron3Nano,
        "mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits": nemotronNanoV2,
        "LiquidAI/LFM2.5-8B-A1B-MLX-4bit": lfm25Dense,
        "LiquidAI/LFM2.5-24B-A2B-MLX-4bit": lfm25,
        "mlx-community/Ministral-3-8B-Instruct-2512-4bit": ministral3,
        "mlx-community/Ministral-3-14B-Reasoning-2512-4bit": ministral3,
        "ornith-ai/Ornith-1.5-9B-MLX-4bit": ornith,
        "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly": qwen38,
    ]

    static let gemma4 = #"""
        <bos><|turn>system
        You are a coding agent.<|tool>declaration:get_weather{description:<|"|>Get the weather for a city<|"|>,parameters:{properties:{city:{description:<|"|>City name<|"|>,type:<|"|>STRING<|"|>},unit:{enum:[<|"|>celsius<|"|>,<|"|>fahrenheit<|"|>],type:<|"|>STRING<|"|>}},required:[<|"|>city<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>
        <|turn>user
        Is it warm in Paris?<turn|>
        <|turn>model
        <think>
        The user asks about Paris; I should check the weather.
        </think>

        Let me check.<turn|>
        <|turn>user
        Please do.<turn|>
        <|turn>model
        <|tool_call>call:get_weather{city:<|"|>Paris<|"|>,unit:<|"|>celsius<|"|>}<tool_call|><|tool_response>response:get_weather{value:<|"|>{"temp": 21}<|"|>}<tool_response|>
        """#

    static let nemotron3Nano = #"""
        <|im_start|>system
        You are a coding agent.

        # Tools

        You have access to the following functions:

        <tools>
        <function>
        <name>get_weather</name>
        <description>Get the weather for a city</description>
        <parameters>
        <parameter>
        <name>city</name>
        <type>string</type>
        <description>City name</description>
        </parameter>
        <parameter>
        <name>unit</name>
        <type>string</type>
        <enum>["celsius", "fahrenheit"]</enum>
        </parameter>
        <required>["city"]</required>
        </parameters>
        </function>
        </tools>

        If you choose to call a function ONLY reply in the following format with NO suffix:

        <tool_call>
        <function=example_function_name>
        <parameter=example_parameter_1>
        value_1
        </parameter>
        <parameter=example_parameter_2>
        This is the value for the second parameter
        that can span
        multiple lines
        </parameter>
        </function>
        </tool_call>

        <IMPORTANT>
        Reminder:
        - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
        - Required parameters MUST be specified
        - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
        - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
        </IMPORTANT><|im_end|>
        <|im_start|>user
        Is it warm in Paris?<|im_end|>
        <|im_start|>assistant
        <think></think>

        Let me check.<|im_end|>
        <|im_start|>user
        Please do.<|im_end|>
        <|im_start|>assistant
        <think></think>
        <tool_call>
        <function=get_weather>
        <parameter=city>
        Paris
        </parameter>
        <parameter=unit>
        celsius
        </parameter>
        </function>
        </tool_call>
        <|im_end|>
        <|im_start|>user
        <tool_response>
        {"temp": 21}
        </tool_response>
        <|im_end|>
        <|im_start|>assistant
        <think>

        """#

    static let nemotronNanoV2 = #"""
        <SPECIAL_10>System
        You are a coding agent.

        You can use the following tools to assist the user if required:
        <AVAILABLE_TOOLS>[{"description": "Get the weather for a city", "name": "get_weather", "parameters": {"properties": {"city": {"description": "City name", "type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"], "type": "string"}}, "required": ["city"], "type": "object"}}]</AVAILABLE_TOOLS>

        If you decide to call any tool(s), use the following format:
        <TOOLCALL>[{{"name": "tool_name1", "arguments": "tool_args1"}}, {{"name": "tool_name2", "arguments": "tool_args2"}}]</TOOLCALL>

        The user will execute tool-calls and return responses from tool(s) in this format:
        <TOOL_RESPONSE>[{{"tool_response1"}}, {{"tool_response2"}}]</TOOL_RESPONSE>

        Based on the tool responses, you can call additional tools if needed, correct tool calls if any errors are found, or just respond to the user.
        <SPECIAL_11>User
        Is it warm in Paris?
        <SPECIAL_11>Assistant
        Let me check.
        <SPECIAL_12>
        <SPECIAL_11>User
        Please do.
        <SPECIAL_11>Assistant
        <TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}]</TOOLCALL>
        <SPECIAL_12>
        <SPECIAL_11>User
        <TOOL_RESPONSE>[{"temp": 21}]</TOOL_RESPONSE>
        <SPECIAL_11>Assistant
        <think>

        """#

    static let lfm25Dense = #"""
        <|startoftext|><|im_start|>system
        You are a coding agent.
        List of tools: [{"function": {"description": "Get the weather for a city", "name": "get_weather", "parameters": {"properties": {"city": {"description": "City name", "type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"], "type": "string"}}, "required": ["city"], "type": "object"}}, "type": "function"}]<|im_end|>
        <|im_start|>user
        Is it warm in Paris?<|im_end|>
        <|im_start|>assistant
        Let me check.<|im_end|>
        <|im_start|>user
        Please do.<|im_end|>
        <|im_start|>assistant
        <|tool_call_start|>[get_weather(city='Paris', unit='celsius')]<|tool_call_end|><|im_end|>
        <|im_start|>tool
        {"temp": 21}<|im_end|>
        <|im_start|>assistant

        """#

    static let lfm25 = #"""
        <|startoftext|><|im_start|>system
        You are a coding agent.
        List of tools: [{"function": {"description": "Get the weather for a city", "name": "get_weather", "parameters": {"properties": {"city": {"description": "City name", "type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"], "type": "string"}}, "required": ["city"], "type": "object"}}, "type": "function"}]<|im_end|>
        <|im_start|>user
        Is it warm in Paris?<|im_end|>
        <|im_start|>assistant
        Let me check.<|im_end|>
        <|im_start|>user
        Please do.<|im_end|>
        <|im_start|>assistant
        <|tool_call_start|>[get_weather(city='Paris', unit='celsius')]<|tool_call_end|><|im_end|>
        <|im_start|>tool
        {"temp": 21}<|im_end|>
        <|im_start|>assistant
        <think>
        """#

    static let ministral3 = #"""
        <s>[SYSTEM_PROMPT]You are a coding agent.[/SYSTEM_PROMPT][AVAILABLE_TOOLS][{"function": {"description": "Get the weather for a city", "name": "get_weather", "parameters": {"properties": {"city": {"description": "City name", "type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"], "type": "string"}}, "required": ["city"], "type": "object"}}, "type": "function"}][/AVAILABLE_TOOLS][INST]Is it warm in Paris?[/INST]<think>
        The user asks about Paris; I should check the weather.
        </think>

        Let me check.</s>[INST]Please do.[/INST][TOOL_CALLS]get_weather[ARGS]{"city": "Paris", "unit": "celsius"}</s>[TOOL_RESULTS]{"temp": 21}[/TOOL_RESULTS]
        """#

    static let ornith = #"""
        <|im_start|>system
        # Tools

        You have access to the following functions:

        <tools>
        {"function": {"description": "Get the weather for a city", "name": "get_weather", "parameters": {"properties": {"city": {"description": "City name", "type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"], "type": "string"}}, "required": ["city"], "type": "object"}}, "type": "function"}
        </tools>

        If you choose to call a function ONLY reply in the following format with NO suffix:

        <tool_call>
        <function=example_function_name>
        <parameter=example_parameter_1>
        value_1
        </parameter>
        <parameter=example_parameter_2>
        This is the value for the second parameter
        that can span
        multiple lines
        </parameter>
        </function>
        </tool_call>

        <IMPORTANT>
        Reminder:
        - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
        - Required parameters MUST be specified
        - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
        - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
        </IMPORTANT>

        You are a coding agent.<|im_end|>
        <|im_start|>user
        Is it warm in Paris?<|im_end|>
        <|im_start|>assistant
        <think>
        The user asks about Paris; I should check the weather.
        </think>

        Let me check.<|im_end|>
        <|im_start|>user
        Please do.<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

        <tool_call>
        <function=get_weather>
        <parameter=city>
        Paris
        </parameter>
        <parameter=unit>
        celsius
        </parameter>
        </function>
        </tool_call><|im_end|>
        <|im_start|>user
        <tool_response>
        {"temp": 21}
        </tool_response><|im_end|>
        <|im_start|>assistant
        <think>

        """#

    static let qwen38 = #"""
        <|im_start|>system
        Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.

        # Tools

        You have access to the following functions:

        <tools>
        {"function": {"description": "Get the weather for a city", "name": "get_weather", "parameters": {"properties": {"city": {"description": "City name", "type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"], "type": "string"}}, "required": ["city"], "type": "object"}}, "type": "function"}
        </tools>

        If you choose to call a function ONLY reply in the following format with NO suffix:

        <tool_call>
        <function=example_function_name>
        <parameter=example_parameter_1>
        value_1
        </parameter>
        <parameter=example_parameter_2>
        This is the value for the second parameter
        that can span
        multiple lines
        </parameter>
        </function>
        </tool_call>

        <IMPORTANT>
        Reminder:
        - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
        - Required parameters MUST be specified
        - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
        - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
        </IMPORTANT>

        You are a coding agent.<|im_end|>
        <|im_start|>user
        Is it warm in Paris?<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

        <think>
        The user asks about Paris; I should check the weather.
        </think>

        Let me check.<|im_end|>
        <|im_start|>user
        Please do.<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

        <tool_call>
        <function=get_weather>
        <parameter=city>
        Paris
        </parameter>
        <parameter=unit>
        celsius
        </parameter>
        </function>
        </tool_call><|im_end|>
        <|im_start|>user
        <tool_response>
        {"temp": 21}
        </tool_response><|im_end|>
        <|im_start|>assistant
        <think>

        """#
}
