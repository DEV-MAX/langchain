
import json
import os

from dotenv import load_dotenv
from langsmith import traceable
load_dotenv()

from openrouter import OpenRouter


@traceable(name="get_product_price", run_type="tool")
def get_product_price(product_name):
    product_prices = {
        "laptop": "$999",
        "smartphone": "$499",
        "headphones": "$199"
        }
    price=product_prices.get(product_name.lower(),"Product not found");

    print(f"Getting the price for {product_name} : {price}")

    return price;

@traceable(name="get_discounted_price", run_type="tool")    
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

client=OpenRouter(  api_key=os.environ["OPENROUTER_API_KEY"])


tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Get the price of a product by its name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The name of the product to get the price for."
                    }
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_discounted_price",
            "description": "Calculate the discounted price based on the original price and discount percentage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "string",
                        "description": "The original price of the product (e.g., '$999')."
                    },
                    "discount_percentage": {
                        "type": "number",
                        "description": "The discount percentage to apply (e.g., 10 for 10%)."
                    }
                },
                "required": ["price", "discount_percentage"]
            }
        }
    }
]

class SystemMessage:
    def __init__(self, content):
        self.content = content
        self.role="system"
    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
        }

class HumanMessage:
    def __init__(self, content):
        self.content = content
        self.role="user"
    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
        }

class ToolMessage:
    def __init__(self, content):
        self.content = content
        self.role="tool"
    def to_dict(self, tool_call_id):
        return {
            "role": self.role,
            "content": self.content,
            "tool_call_id": tool_call_id
        }


@traceable(name="invoke_agent")
def invoke_agent(messages_to_send):
    return  client.chat.send(
            model=MODEL,
            messages=messages_to_send,
            tools=tools_schema,
            tool_choice="auto"
        )

@traceable(name="run_agent")
def run_agent(question:str):
    tools_dict={
        "get_product_price": get_product_price,
        "get_discounted_price": get_discounted_price
    }

    print(f"Running agent with model {MODEL} and tools {tools_schema} for question: {question}")
    print("="*60)
    messages=[
        SystemMessage(content=
    "You are a product pricing assistant. "
    "Known products are: laptop, smartphone, headphones. "
    "If the user mentions one of these known products, call get_product_price using that exact product name. "
    "For discounted price questions, first call get_product_price, then call get_discounted_price using the exact price returned by get_product_price. "
    "Never calculate discounts yourself. "
    "If the product is not one of the known products, call get_product_price anyway and use its result. "
        ),
        HumanMessage(content=question)
    ]

    messages_to_send=[m.to_dict() for m in messages]
    for i in range(1,ITERATION+1):
        print(f"Iteration {i}/{ITERATION}")
        ai_message=invoke_agent(messages_to_send)
        print(f"AI response: {ai_message}")
        ai_message_content=ai_message.choices[0].message

        tool_calls_to_make= ai_message.choices[0].message.tool_calls 

        if not tool_calls_to_make:
            print("No more tool calls needed. Final response:")
            print(ai_message.choices[0].message.content)
            return ai_message.choices[0].message.content
        
        tool_to_call=tool_calls_to_make[0]
        tool_name=tool_to_call.function.name
        tool_args=json.loads(tool_to_call.function.arguments)
        available_tool= tools_dict.get(tool_name)
        
        

        print(f"AI wants to call tool: {tool_name} with args: {tool_args}")
        if not available_tool:
            print(f"Tool {tool_name} not found. Skipping tool call.")
            continue

        observation = available_tool(**tool_args)

        messages_to_send.append(ai_message_content)
        messages_to_send.append(
            ToolMessage(content=str(observation)).to_dict(tool_call_id=tool_to_call.id)
            )


    print("Reached maximum iterations without a final answer. Returning last AI message content.")
    return None



def main():
    run_agent("What is the price of the laptop and what would be the discounted price with a 10% discount?")



if __name__ == "__main__":
    main()
