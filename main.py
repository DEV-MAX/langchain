
from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import tool

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, ToolMessage


@tool("get_product_price", description="Get the price of a product by its name.")
def get_product_price(product_name):
    product_prices = {
        "laptop": "$999",
        "smartphone": "$499",
        "headphones": "$199"
        }
    price=product_prices.get(product_name.lower(),"Product not found");

    print(f"Getting the price for {product_name} : {price}")

    return price;


@tool("get_discounted_price", description="Calculate the discounted price given the original price and discount percentage.")
def get_discounted_price(price, discount_percentage):
    try:
        print(f"Received price: {price} and discount percentage: {discount_percentage}")
        price_value = float(price.strip('$'))
        discount_value = float(discount_percentage)
        print(f"Converted price to float: {price_value} and discount to float: {discount_value}")

        discounted_price = price_value * (1 - discount_value / 100)
        print(f"Calculating discounted price: Original price {price}, Discount {discount_percentage}%, Discounted price ${discounted_price:.2f}")
        return f"${discounted_price:.2f}"
    except ValueError:
        print(f"Invalid price format: {price}")
        return "Invalid price format"


MODEL="nvidia/nemotron-3-super-120b-a12b:free"
ITERATION=10


def run_agent(question:str):
    tools=[get_product_price, get_discounted_price]
    tools_dict={t.name: t for t in tools}
    llm= init_chat_model(MODEL, temperature=0.1, model_provider="openrouter")
    llm_with_tools = llm.bind_tools(tools)
    
    print(f"Running agent with model {MODEL} and tools {tools} for question: {question}")
    print("="*60)
    messages=[
        SystemMessage(content=
                      "You are a helpful assistant that can provide product prices and calculate discounted prices."
                      "Use the tools provided to answer the user's question accurately and concisely."
                      "If the user asks for a product price, use the get_product_price tool."
                      "If they ask for a discounted price, use the get_discounted_price tool."
                      "always call in order, dont call get_discounted_price before get_product_price, first get the price and then calculate the discounted price."
                      "If you don't know the answer, say you don't know instead of making up an answer."
                      "If the price is not available, say 'Product not found' instead of making up a price."
                      "STRICT RULES — you must follow these exactly:\n"
                        "1. NEVER guess or assume any product price. "
                        "You MUST call get_product_price first to get the real price.\n"
                        "2. Only call apply_discount AFTER you have received "
                        "a price from get_product_price. Pass the exact price "
                        "returned by get_product_price — do NOT pass a made-up number.\n"
                        "3. NEVER calculate discounts yourself using math. "
                        "Always use the apply_discount tool.\n"
                        "4. If the user does not specify a discount tier, "
                        "ask them which tier to use — do NOT assume one."
                      ),
        HumanMessage(content=question)
    ]
    for i in range(1,ITERATION+1):
        print(f"Iteration {i}/{ITERATION}")
        ai_message=llm_with_tools.invoke(messages)
        tool_calls_to_make= ai_message.tool_calls

        if not tool_calls_to_make:
            print("No more tool calls needed. Final response:")
            print(ai_message.content)
            return ai_message.content
        
        tool_to_call=tool_calls_to_make[0]
        tool_name=tool_to_call.get("name")
        tool_args=tool_to_call.get("args",{})
        available_tool=tools_dict.get(tool_name)
        tool_call_id=tool_to_call.get("id")

        print(f"AI wants to call tool: {tool_name} with args: {tool_args}")
        if not available_tool:
            print(f"Tool {tool_name} not found. Skipping tool call.")
            continue

        observation = available_tool.invoke(tool_args)

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
            )


    print("Reached maximum iterations without a final answer. Returning last AI message content.")
    return None



def main():
    run_agent("What is the price of the laptop and what would be the discounted price with a 10% discount?")



if __name__ == "__main__":
    main()
