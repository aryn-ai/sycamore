# Using conversation memory with Langchain

[LangChain](https://www.langchain.com/) is a popular framework for building applications with large language models, and you can use Sycamore to power apps created with Langchain. In this example, we will show how to use Sycamore's conversation memory feature set to store chat history for a LangChain app. This tutorial will walk through creating an implementation of chat history that integrates with LangChain primitives, and calls out to Sycamore's conversation memory APIs.

## Prepare OpenSearch client for Sycamore

Sycamore is compatible with the OpenSearch conversation memory APIs, and you can use an OpenSearch client to access them. However, these APIs are not yet available in the various language clients at the time we wrote this tutorial. Luckily, it’s super easy to just create it in Python. First, for dependencies: this requires the `opensearchpy` package, OpenSearch’s python client:

```bash
pip install opensearch-py
```

Then, imports:

```python
from opensearchpy.client.utils import NamespacedClient, _make_path
from opensearchpy import OpenSearch
import json
```

Now, we’ll create our `ConversationMemoryOpensearchClient` as a subclass of `NamespaceClient`, which is an abstract client class that constructs off of a basic `OpenSearch` client, and provides the functionality to hit any OpenSearch endpoint exposed by a cluster. We will define a method for each of the conversation memory APIs:

```python
class ConversationMemoryOpensearchClient(NamespacedClient):

    def create_conversation(self, name: str=None):
        return self.transport.perform_request(
            method="POST",
            url=_make_path("_plugins", "_ml", "memory", "conversation"),
            body=({"name": name} if name is not None else None)
        )

    def create_interaction(self, conversation_id: str, input: str,
                    prompt: str, response: str, origin: str,
                    additional_info: dict):
        return self.transport.perform_request(
            method="POST",
            url=_make_path("_plugins", "_ml", "memory", "conversation", conversation_id),
            body={
                "input": input,
                "prompt": prompt,
                "response": response,
                "origin": origin,
                "additional_info": json.dumps(additional_info)
            }
        )

    def get_conversations(self, max_results: int = None, next_token: int = None):
        params = {}
        if max_results:
            params["max_results"] = max_results
        if next_token:
            params["next_token"] = next_token

        return self.transport.perform_request(
            method="GET",
            url=_make_path("_plugins", "_ml", "memory", "conversation"),
            params=params if len(params) != 0 else None
        )

    def get_interactions(self, conversation_id: str, max_results: int = None,
												next_token: int = None):
        params = {}
        if max_results:
            params["max_results"] = max_results
        if next_token:
            params["next_token"] = next_token

        return self.transport.perform_request(
            method="GET",
            url=_make_path("_plugins", "_ml", "memory", "conversation", conversation_id),
            params=params if len(params) != 0 else None
        )

    def delete_conversation(self, conversation_id: str):
        return self.transport.perform_request(
            method="DELETE",
            url=_make_path("_plugins", "_ml", "memory", "conversation", conversation_id),
        )
```

We can construct one like this:

```python
opensearch_client = OpenSearch(
	hosts = [{'host': 'localhost', 'port': 9200}],
    http_compress = True, # enables gzip compression for request bodies
    use_ssl = True,
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False
)
conversation_client = ConversationMemoryOpensearchClient(opensearch_client)
```

With this implementation, we hit the endpoints of the conversation memory API with the appropriate arguments, using a method provided by `NamespaceClient` that does just that. The return values are all parsed into python dicts and lists from JSON, so all that's required is minimal logic to determine what optional parameters to send.

## LangChain ChatHistory

Now let’s use this client to implement a LangChain ChatHistory management object. This will be a subclass of `BaseChatMessageHistory`, LangChain’s base class for handling chat history over a number of backends. We need to implement a constructor, a `messages` method/property, an `add_message` method, and a `clear` method. First, if you don’t have LangChain installed, then let's install it

```python
pip install langchain
```

Next, imports:

```python
from typing import List

from langchain.schema import BaseChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
```

Now, for the implementation:

```python
class OpenSearchChatMessageHistory(BaseChatMessageHistory):

    def __init__(
        self,
        client: OpenSearch,
        conversation_id: str = None
    ):
        self.conversation_client = ConversationMemoryOpensearchClient(client)
        self.conversation_id = conversation_id
        self.pending = []

        # Validate that this conversation id exists
        if not self._validate_conversation_id():
            self.conversation_id = None

    def _validate_conversation_id(self):
        if self.conversation_id == None:
            return False
        next_token = 0
        conversations = []
        while self.conversation_id not in conversations:
            conversations_response = self.conversational_client.get_conversations(next_token=next_token, max_results=100)
            conversations = [c["conversation_id"] for c in conversations_response.get("conversation_id")]
            if self.conversation_id in conversations:
                return True
            if "next_token" not in conversations_response:
                return False
            next_token = conversations_response.get(next_token)
        return False

    @property
    def messages(self) -> List[BaseMessage]:
        if self.conversation_id is None:
            return []
        messages = []
        next_token = 0
        while True:
            response = self.conversation_client.get_interactions(self.conversation_id, next_token=next_token, max_results=100)
            for interaction in response.get("interactions"):
                messages.insert(0, AIMessage(content=interaction.get("response")))
                messages.insert(0, HumanMessage(content=interaction.get("input")))
            if "next_token" not in response:
                return messages + self.pending
            next_token = response.get("next_token")

    def add_message(self, message: BaseMessage):
        # If no conversation is active, create one
        if self.conversation_id is None:
            self.conversation_id = self.conversation_client.create_conversation(name=message.content).get("conversation_id")

        # If no pending message then no pairs are possible
        if len(self.pending) == 0:
            self.pending.append(message)
        # If the pending messages are the same type as this message, then it's not a pair
        elif type(message) == type(self.pending[0]):
            self.pending.append(message)
        # If the pending messages are different type than this message, it's a pair and we can write
        else:
            for msg in reversed(self.pending[1:]):
                if type(msg) == HumanMessage:
                    self.conversation_client.create_interaction(
                        self.conversation_id, msg.content, "", "", "", {}
                    )
                if type(msg) == AIMessage:
                    self.conversation_client.create_interaction(
                        self.conversation_id, "", "", msg.content, "", {}
                    )
            if type(message) == HumanMessage:
                self.conversation_client.create_interaction(
                    self.conversation_id, message.content, "", self.pending[0].content, "", {}
                )
            if type(message) == AIMessage:
                self.conversation_client.create_interaction(
                    self.conversation_id, self.pending[0].content, "", message.content, "", {}
                )
            self.pending = []

    def clear(self):
        self.conversation_client.delete_conversation(self.conversation_id)
        self.pending = []
```

Let’s go over this method by method.

1. `__init__`: We make a `ConversationMemoryOpensearchClient` to use for all our API calls, and we construct this from the OpenSearch client object to keep complexity to a minimum. We’ll also be holding onto a single `conversation_id` for the duration of this Chat History. Additionally, we introduce a list called `pending`, because LangChain expects to be able to put individual messages into history one at a time. On the other hand, OpenSearch's Conversation API is built around interactions, which represent a pair of messages (e.g. query and response). So, `pending` represents a message that doesn’t yet have a pair. For this exmple, we use a list, because it's possible for a user to submit several `HumanMessages` before adding an `AIMessage`, or vice versa (although we think this scenario is unlikely). Finally, we validate that if you passed in a conversation ID, it’s a conversation that exists. Otherwise, we set the `conversation_id` to None.

2. `_validate_conversation_id`: The job of this method is to determine whether the currently held conversation ID exists. We do this by retrieving the list of conversations, and checking to see if the currently held conversation ID is in that list. Since the GetConversations API is paginated, we iteratively step through it in case the conversation we’re looking for isn’t on the first page. If we run out of pages, the conversation doesn’t exist.

3. `messages`: This method retrieves the list of messages in the conversation and arranges them for LangChain. There are a couple of oddities about this. First, the GetInteractions API is paginated, but LangChain expects to receive all of the messages at once, so we again have to iterate through the pages. Second, the GetInteractions API retrieves interactions sorted from most recent to least recent; so we have to build the list of messages backwards. Third, an interaction consists of two messages, so we have to decompose that and add two messages for every interaction. Lastly, if there are any pending messages, they will not be returned by the conversation memory API, since they are yet to enter the Sycamore indices, so we need to tack those on at the end too.

4. `add_message`: This method adds a message to conversation memory. Because of the mismatch between LangChain messages and conversation memory interactions described above, there is some added complexity:
    1. If nothing is pending, the new message doesn’t have a pair, so we put it in the 'pending' container to wait for a pair.
    2. If the new message is the same kind of message as the pending messages (e.g. 'HumanMessages'), then we still don’t have a pair of Human/AI messages, so we add the new message onto the pending list.
    3. If the new message is a different kind of message from the pending messages, then we *do* have a pair. If there are *unpaired* pending messages still in the list, we assume that the pair comes from the most recent message (this preserves the message order), and iteratively enter the unpaired messages into conversation memory as interactions with null responses or inputs. Finally, we add the paired message to conversation memory, and then empty out the pending list.

5. `clear`: This method resets everything. It deletes the conversation from the memory, un-set the current conversation id, and empties out the pending list.

## LLMChain with Memory

This is an example on how to create a `LLMChain` that uses the Sycamore conversation memory API as a remote conversation memory store.

```python
from opensearchpy import OpenSearch

from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

template = """You are a chatbot having a conversation with a human.

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)

opensearch_client = OpenSearch(
    hosts = [{'host': 'localhost', 'port': 9200}],
    http_compress = True, # enables gzip compression for request bodies
    use_ssl = True,
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False
)

message_history = OpenSearchChatMessageHistory(opensearch_client)

memory = ConversationBufferMemory(
	memory_key="chat_history",
	chat_memory=message_history
)

llm_chain = LLMChain(
	llm=OpenAI(temperature=0),
	prompt=prompt,
	verbose=True,
	memory=memory
)
```

And now, this LLMChain will write its interactions to Sycamore's conversation memory.
