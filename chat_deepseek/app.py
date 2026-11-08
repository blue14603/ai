import chainlit as cl
import ollama

@cl.on_chat_start
async def start_chat():
    cl.user_session.set("interaction",
                            [
                                {"role": "system", "content": "You are a helpful assistant."}, # context for AI
                            ],
    )
    msg = cl.Message(content="") # default starting msg
    start_message = "Welcome Giang To! How can I help you today"
    for token in start_message:
        await msg.stream_token(token)
    await msg.send()

@cl.step(type="tool")
async def tool(input_message):
    interaction = cl.user_session.get("interaction")
    interaction.append({"role": "user", "content": input_message})
    response = ollama.chat('deepseek-r1', messages=interaction)
    interaction.append({"role": "assistant", "content": response.message.content})
    return response 

@cl.on_message
async def on_message(message:cl.Message):
    tool_res = await tool(message.content)
    msg = cl.Message(content="") 
    for token in tool_res.message.content:
        await msg.stream_token(token)
    await msg.send()