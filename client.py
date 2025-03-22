import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import List

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, stdio_client
from openai import OpenAI

load_dotenv(override=True)


class MCPClient:
    def __init__(self):
        """初始化客户端"""
        # 初始化session和exit_stack
        self.session = None
        self.exit_stack = AsyncExitStack()

        # 从环境变量中获取OpenAI API密钥、基础URL和模型信息
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")

        # 检查是否设置了OpenAI API密钥
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        # 使用获取到的信息初始化OpenAI客户端
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)

    async def process_query(self, query: str) -> str:
        messages = [
            {"role": "user", "content": query},
        ]
        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",  # 确保类型为 function
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,  # 确保参数名称正确
                },
            }
            for tool in response.tools
        ]

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1024,
                    tools=available_tools,
                ),
            )

            content = response.choices[0]
            if content.finish_reason == "tool_calls":
                tool_call = content.message.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                result = await self.session.call_tool(tool_name, tool_args)
                print(f"\n\n[Calling tool] {tool_name} with args: {tool_args} \n")

                messages.append(content.message.model_dump())
                messages.append(
                    {
                        "role": "tool",
                        "content": result.content[0].text,
                        "tool_call_id": tool_call.id,
                    }
                )
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages
                )
                return response.choices[0].message.content
            return content.message.content
        except Exception as e:
            print(e)
            return f"调用大模型 API 出错,错误信息:{e}"

    async def connect_to_server(self, cmds: List[str]):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """

        server_params = StdioServerParameters(command=cmds[0], args=cmds[1:], env=None)

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def chat_loop(self):
        """运行交互式聊天"""
        print("\nMCP 客户端已启动！输入'quit'以退出")

        while True:
            try:
                query = input("[我]: ").strip()
                if query == "quit":
                    break
                response = await self.process_query(query)
                print(f"\n[Mock Response]: {response}\n")
            except Exception as e:
                print(f"\n 发生异常: {str(e)}")

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()
        print("\nMCP 客户端已关闭！")


async def main():
    client = MCPClient()
    try:
        await client.connect_to_server(
            ["docker", "run", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "-i", "mcp/github"]
        )
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
