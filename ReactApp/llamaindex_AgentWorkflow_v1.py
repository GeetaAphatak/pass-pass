from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document, BaseNode
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.agent import ReActAgent
from llama_index.core.workflow import draw_all_possible_flows
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms import ChatMessage, ChatResponse
from typing import Any, List, Optional, Sequence, Dict
import requests, json
import pandas as pd
import asyncio
from dotenv import load_dotenv
import logging, os, sys

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event
from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from configuration_workstream.configuration_workstream import ConfigurationWorkStream




# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)
EMBEDDING_URL = os.getenv('embedding_url', '')
API_KEY = os.getenv('embeddings_bearer_token', '')
EMBEDDING_MODEL_NAME = os.getenv('embedding_model_name', '')
API_URL = os.getenv('API_URL', '')
BEARER_TOKEN = os.getenv('BEARER_TOKEN', '')




class CustomEmbeddings(BaseEmbedding):
    """
    Custom embedding class for interacting with a specific embedding API.

    This class provides methods to generate embeddings for text using a custom API.

    Attributes:
        _api_key (str): API key for authentication.
        _embedding_url (str): URL of the embedding API.
        _embedding_model_name (str): Name of the embedding model to use.
    """
    _api_key: str = API_KEY
    _embedding_url: str = EMBEDDING_URL
    _embedding_model_name: str = EMBEDDING_MODEL_NAME
    

    def __init__(self): #, api_key: str, embedding_url: str, embedding_model_name: str):
        """
        Initialize the CustomEmbeddings instance.

        Args:
            api_key (str): API key for authentication.
            embedding_url (str): URL of the embedding API.
            embedding_model_name (str): Name of the embedding model to use.
        """
        super().__init__()
        logger.info("Initializing CustomEmbeddings class.")
        # self._api_key = api_key
        # self._embedding_url = embedding_url
        # self._embedding_model_name = embedding_model_name

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embeddings, where each embedding is a list of floats.

        Raises:
            ValueError: If the API response format is unexpected.
            requests.RequestException: If there's an error in the API request.
        """
        logger.info(f"Embedding {len(texts)} texts.")
        headers = {
            'applicationType': "Developer",
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}',
        }
        payload = {
            "model": self._embedding_model_name,
            "input": texts
        }
        try:
            response = requests.post(self._embedding_url, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            if 'result' in response_data and 'data' in response_data['result']:
                embeddings = [item['embedding'] for item in response_data['result']['data']]
                logger.info(f"Received {len(embeddings)} embeddings successfully.")
                return embeddings
            else:
                raise ValueError(f"Unexpected response format: {response_data}")
        except requests.RequestException as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a single query text."""
        return self._embed([query])[0]
        
    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get the embedding for a single text.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: The embedding of the text.
        """
        return self._embed([text])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Asynchronously get embedding for a single query text."""
        return await asyncio.to_thread(self._get_query_embedding, query)


    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embeddings, where each embedding is a list of floats.
        """
        return self._embed(texts)

    @property
    def embed_dim(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            int: The dimension of the embedding vectors.
        """
        return 1536

    @property
    def query_embed_dim(self) -> int:
        """Return the query embedding dimension."""
        return self.embed_dim


class APILLM(CustomLLM):
    context_window: int = 39000
    num_output: int = 2560
    model_name: str = "gpt-4o"
    api_url: str = API_URL
    bearer_token: str = BEARER_TOKEN

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def _make_api_call(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "applicationType": "Developer",
            "Content-Type": "application/json"
        }
        input_data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.num_output,
            "temperature": 0
        }
        response = requests.post(self.api_url, json=input_data, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response_text = self._make_api_call(prompt)
        return CompletionResponse(text=response_text)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response_text = self._make_api_call(prompt)
        yield CompletionResponse(text=response_text)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        async for response in self.stream_complete(prompt, **kwargs):
            yield response

## Define workflow events:

class PrepEvent(Event):
    pass


class InputEvent(Event):
    input: list[ChatMessage]


class StreamEvent(Event):
    delta: str


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput


class ReActAgent(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: LLM | None = None,
        tools: list[BaseTool] | None = None,
        extra_context: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.llm = llm or APILLM()
        self.formatter = ReActChatFormatter.from_defaults(
            context=extra_context or ""
        )
        self.output_parser = ReActOutputParser()

    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        # clear sources
        await ctx.set("sources", [])

        # init memory if needed
        memory = await ctx.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        # get user input
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)

        # clear current reasoning
        await ctx.set("current_reasoning", [])

        # set memory
        await ctx.set("memory", memory)

        return PrepEvent()

    @step
    async def prepare_chat_history(
        self, ctx: Context, ev: PrepEvent
    ) -> InputEvent:
        # get chat history
        memory = await ctx.get("memory")
        chat_history = memory.get()
        current_reasoning = await ctx.get("current_reasoning", default=[])

        # format the prompt with react instructions
        llm_input = self.formatter.format(
            self.tools, chat_history, current_reasoning=current_reasoning
        )
        return InputEvent(input=llm_input)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        chat_history = ev.input
        current_reasoning = await ctx.get("current_reasoning", default=[])
        memory = await ctx.get("memory")

        response_gen = await self.llm.astream_chat(chat_history)
        async for response in response_gen:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))

        try:
            reasoning_step = self.output_parser.parse(response.message.content)
            current_reasoning.append(reasoning_step)

            if reasoning_step.is_done:
                memory.put(
                    ChatMessage(
                        role="assistant", content=reasoning_step.response
                    )
                )
                await ctx.set("memory", memory)
                await ctx.set("current_reasoning", current_reasoning)

                sources = await ctx.get("sources", default=[])

                return StopEvent(
                    result={
                        "response": reasoning_step.response,
                        "sources": [sources],
                        "reasoning": current_reasoning,
                    }
                )
            elif isinstance(reasoning_step, ActionReasoningStep):
                tool_name = reasoning_step.action
                tool_args = reasoning_step.action_input
                return ToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id="fake",
                            tool_name=tool_name,
                            tool_kwargs=tool_args,
                        )
                    ]
                )
        except Exception as e:
            current_reasoning.append(
                ObservationReasoningStep(
                    observation=f"There was an error in parsing my reasoning: {e}"
                )
            )
            await ctx.set("current_reasoning", current_reasoning)

        # if no tool calls or final response, iterate again
        return PrepEvent()

    @step
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> PrepEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        current_reasoning = await ctx.get("current_reasoning", default=[])
        sources = await ctx.get("sources", default=[])

        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                current_reasoning.append(
                    ObservationReasoningStep(
                        observation=f"Tool {tool_call.tool_name} does not exist"
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                sources.append(tool_output)
                current_reasoning.append(
                    ObservationReasoningStep(observation=tool_output.content)
                )
            except Exception as e:
                current_reasoning.append(
                    ObservationReasoningStep(
                        observation=f"Error calling tool {tool.metadata.get_name()}: {e}"
                    )
                )

        # save new state in context
        await ctx.set("sources", sources)
        await ctx.set("current_reasoning", current_reasoning)

        # prep the next iteraiton
        return PrepEvent()
    

# Load and process the JSON data
def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['apps']

def create_documents_from_json(apps_data: List[Dict[str, Any]]) -> List[Document]:
    documents = []
    for app in apps_data:
        content = f"Name: {app['name']}\nDescription: {app['description']}\nLink: {app['link']}"
        metadata = {"name": app['name'], "img": app['img']}
        documents.append(Document(text=content, metadata=metadata))
    return documents

# Create the RAG system
def create_rag_system(documents: List[Document], custom_embeddings: BaseEmbedding, llm: APILLM):
    # Create nodes from documents
    node_parser = SimpleNodeParser.from_defaults()
    nodes = node_parser.get_nodes_from_documents(documents)

    # Create the index with custom embeddings
    index = VectorStoreIndex(nodes, embed_model=custom_embeddings)

    # Create a query engine
    query_engine = index.as_query_engine(llm=llm)

    return query_engine


# Load csv file into pandas DataFrame
def load_csv_to_dataframe(file_path: str) -> pd.DataFrame:
    # Read the CSV file
    df = pd.read_csv(file_path)
    for col in df.columns:
        df[col] = df[col].astype(str)

    return df

# Function to query the csv DataFrame
def query_csv_dataframe(df: pd.DataFrame, query: str) -> str:
    try:
        # Use eval to execute the query string as a pandas operation
        result = df.query(query)
        if result.empty:
            return "No results found for the given query."
        else:
            return result.to_string()
    except Exception as e:
        return f"Error executing query: {str(e)}"
    
async def generate_csv_description(llm: APILLM, csv_df: pd.DataFrame) -> str:
    # Get a random sample of the data
    sample = csv_df.sample(n=min(10, len(csv_df))).to_string()

    prompt = f"""
    Analyze the following sample of csv data and provide a concise description of its structure and content.
    Include information about the columns, what kind of data they contain, and any patterns you notice.
    
    Sample csv data:
    {sample}
    
    Dataframe info:
    Total rows: {len(csv_df)}
    Columns: {', '.join(csv_df.columns)}
    
    Description:
    """

    response = await llm.acomplete(prompt)
    return response.text

def get_prompt(prompt_path):
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt = file.read()
        return prompt
    except FileNotFoundError:
        print(f"Prompt file not found: {prompt_path}")
        raise
    except Exception as e:
        print(f"Error loading prompt file: {str(e)}")
        raise

def configuration_workstream(data: str):
        config_workstream = ConfigurationWorkStream()
        try:
            logging.debug("Received request at /get_response endpoint")
            if config_workstream.error_message:
                return (f"Error: {config_workstream.error_message}")

            if data is None:
                error_message = "No Query."
                logging.warning(error_message)
                return (f"Error: {error_message}")

            question = config_workstream.fix_text(data)
            filtered_df = config_workstream.load_data_from_db(question)

            response_text = config_workstream.query_llm_with_data(question, filtered_df)

            df_json = filtered_df.to_json(orient='records')
            logging.info("Request processed successfully")
            return ({"response_text": response_text, "filtered_data": df_json})
        except Exception as e:
            error_message = f"An internal error occurred: {str(e)}"
            logging.error(error_message)
            return ({"response_text": error_message, "filtered_data": pd.DataFrame().to_json(orient='records')}), 500



# Create the agent
async def create_agent(gto_query_engine: RetrieverQueryEngine, csv_df: pd.DataFrame, llm: APILLM):
    # Generate csv description
    csv_description = await generate_csv_description(llm, csv_df)
    print("csv_description", csv_description)

    # def csv_query_tool(query: str) -> str:
    #     result = query_csv_dataframe(csv_df, query)
    #     return f"csv query result:\n{result}"
    external_prompt = get_prompt('configuration_workstream_prompt.txt')

    def csv_query_tool(query: str) -> str:
        try:
            if query.lower() == 'columns':
                return f"CSV columns: {', '.join(csv_df.columns)}"
            elif query.lower() == 'head()' or query.lower() == 'head':
                return f"First few rows of the CSV:\n{csv_df.head().to_string()}"
            else:
                # Parse the filter statement
                # column, value = query.split('==')
                column, value = query.split('contains')
                column = column.strip()
                value = value.strip().strip("'")
                
                # Apply the filter
                result = csv_df[csv_df[column].str.contains(value, case=False)]
                # print("results:",result)
                
                if result.empty:
                    return f"No results found for {column} containing {value}"
                else:
                    return result.to_dict(orient='records')
        except Exception as e:
            return f"Error executing query: {str(e)}\nPlease check the column names and query syntax."

    tools = [
        QueryEngineTool.from_defaults(
            query_engine=gto_query_engine,
            name="GTO_Apps_Knowledge_Base",
            description="Use this tool to find information about GTO applications and tools, including onboarding processes."
        ),
        FunctionTool.from_defaults(
            fn=csv_query_tool,
            name="csv_Query_Tool",
            description=f"""
            Use this tool to query csv information from a CSV file. 
            
            CSV Data Description:
            {await generate_csv_description(llm, csv_df)}
            
            How to use:
            1. To see column names, use query: 'columns'
            2. To see the first few rows, use query: 'head()'
            3. For other queries, use the format: "'COLUMN NAME' contains 'VALUE'"
               For example: "SWITCH NAME contains 'DPVPAR'"
            The tool will return the matching rows as a JSON string.
            If you find multiple entries, give the list of asked values and ask user to choose.
            """
            #  If there is a single response, respond based this information: {external_prompt}
            
        ),
        FunctionTool.from_defaults(
            fn=configuration_workstream,
            name="configuration_workstream",
            description=f"""
            Use this tool to fetch information about Configuration application.
            The Configuration application is an AI-powered chatbot specifically designed to support users in understanding and managing the functionality setup of the Impact application. 
            This setup is governed by a system of Taskcommon switches, which control various aspects of functionality within the Impact platform. 
            Through its intuitive interface, the Configuration chatbot provides detailed information about each Taskcommon switch. This includes the switch's current position, its full name, a descriptive summary of its purpose, a shortened identifier for quick reference called Short Name, as well as its associated Work Group and Work Area. These data points help users gain a comprehensive understanding of how each switch influences application behavior. 
            In addition to offering general reference information, the Configuration chatbot also enables users to query specific configurations for individual Impact clients. It can retrieve and display how Taskcommon switches are currently set up for a particular client, thereby assisting administrators and support teams in troubleshooting, auditing, or optimizing system configurations based on client-specific requirements. 
            
            If there is a single response, respond based this information: {external_prompt}
            If the query is not clear, ask for clarification by posing appropriate queries to user.
            """
        )
    ]

    agent = ReActAgent(
        tools=tools,
        llm=llm,
        verbose=True,
        extra_context=(
            "You are a helpful AI assistant that can answer questions about GTO applications and CSv data. "
            "For GTO-related questions, use the GTO_Apps_Knowledge_Base tool. "
            "For CSV-related questions, use the csv_Query_Tool. "
            "When using the csv_Query_Tool, formulate a query in the format 'COLUMN NAME contains 'VALUE''. "
            "Interpret the JSON result and provide a natural language response to the user. "
            "If you don't find enough information to answer the query, say so politely."
        )
    )
    draw_all_possible_flows(ReActAgent, filename="OnboardGPT_agent_flow.html")
    return agent


async def main():
    # Load and process the JSON data
    apps_data = load_json_data('apps.json')
    gto_documents = create_documents_from_json(apps_data)

    # Load csv file into pandas DataFrame
    csv_df = load_csv_to_dataframe('./BroadridgeDictionary.csv')
    print(type(csv_df), csv_df)

    # Create the APILLM
    llm = APILLM()

    # Create custom embeddings
    custom_embeddings = CustomEmbeddings()
    
    # Create the RAG system for GTO apps
    gto_query_engine = create_rag_system(gto_documents, custom_embeddings, llm)

    # Create the agent
    agent = await create_agent(gto_query_engine, csv_df, llm)
    user_history = ""
    
    # Use the agent
    while True:
        user_input = input("Ask a question about GTO apps or csv (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        try:
            user_history = user_input #+ "\n" +  user_history 
            response = await agent.run(input = user_history)
            print("Agent's response:")
            print(response["response"])
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            # print("Agent's thought process:")
            # Instead of trying to access a non-existent method, we'll print the agent's state
            # print(f"Agent state: {agent.state}")
            # print(f"Last tool used: {agent.last_tool_used}")
            # If you want to see the full agent object (which might be very large), you can uncomment the next line
            # print(f"Full agent object: {agent}")

if __name__ == "__main__":
    asyncio.run(main())


