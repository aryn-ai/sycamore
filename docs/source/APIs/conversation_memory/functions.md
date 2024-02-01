# API Spec

## CreateConversation

Creates a new conversation object in memory

```javascript
POST /_plugins/_ml/memory/conversation
{
	"name": "Name for conversation"
}
```

| Field | Type | Optional | Default | Description |
| --- | --- | --- | --- | --- |
| name | string | yes | “” | A user-defined name for the conversation, for better identification than a uuid |

Response

```javascript
{
	"conversation_id": "ureogi540fedf"
}
```

| Field | Type | Optional | Description |
| --- | --- | --- | --- |
| conversation_id | string | no | id of the newly created conversation |

## CreateInteraction

Creates a new interaction object in a conversation (in memory)

```javascript
POST /_plugins/_ml/memory/conversation/{conversation_id}
{
	"input": "What are the sort benchmarks",
	"prompt_template": "Answer the question, but also pay attention to these search results",
	"response": "The Sort Benchmarks are a series of benchmarks for sorting",
	"origin": "RAG Pipeline",
	"additional_info": "[Document1, Document2, etc]"
}
```

| Field | Type | Optional | Default | Description |
| --- | --- | --- | --- | --- |
| conversation_id | string | no |  | id of the conversation to add this interaction to |
| input | string | no |  | human input that started this interaction |
| prompt_template | string | yes | null | prompt template used to construct the prompt for this interaction |
| response | string | no |  | AI’s response for this interaction |
| origin | string | yes | null | Name of the system that generated this interaction |
| additional_info | string | yes | null | Any additional info required for reconstructing the prompt; e.g. search results in RAG |

Response

```javascript
{
	"interaction_id": "049t4ttjh3po"
}
```

| Field | Type | Optional | Description |
| --- | --- | --- | --- |
| interaction_id | string | no | id of the newly created interaction |

## GetConversations

Returns a list of conversations, sorted most recent → least recent

```javascript
GET /_plugins/_ml/memory/conversation?max_results=5&next_token=3
```

| Field | Type | Optional | Default | Description |
| --- | --- | --- | --- | --- |
| next_token | int | yes | 0 | ordered position to start retrieving conversations from |
| max_results | int | yes | 10 | number of conversations to retrieve |

Response

```javascript
{
	"conversations": [
		{
			"conversation_id": "14924u4hjge",
			"create_time": timestamp,
			"name": "Name for conversation"
		}, ...
	],
	"next_token": 8
}
```

| Field | Type | Optional | Description |
| --- | --- | --- | --- |
| conversations | List[Conversation] | no | list of conversation objects |
| next_token | int | yes | next next_token to use to continue retrieving conversations. If no more conversations, next_token does not appear |

Conversation

| Field | Type | Optional | Description |
| --- | --- | --- | --- |
| name | string | yes | A user-defined name for the conversation, for better identification than a uuid |
| conversation_id | string | no | uuid for this conversation |
| create_time | timestamp | no | when this conversation was created |

## GetInteractions

Returns a list of interactions belonging to a conversation, sorted most recent → least recent

```javascript
GET /_plugins/_ml/memory/conversation/{conversation_id}?max_results=5&next_token=3
```

| Field | Type | Optional | Default | Description |
| --- | --- | --- | --- | --- |
| conversation_id | string | no |  | id of the conversation to get interactions from |
| next_token | int | yes | 0 | ordered position to start retrieving conversations from |
| max_results | int | yes | 10 | number of conversations to retrieve |

Response

```javascript
{
	"interactions": [
		{
			"conversation_id": "14924u4hjgenkau4",
			"interaction_id": "l24ng4op5gih88ei",
			"create_time": timestamp,
			"input": "What are the sort benchmarks",
			"prompt_template": "Answer the question, but also pay attention to these search results",
			"response": "The Sort Benchmarks are a series of benchmarks for sorting",
			"origin": "RAG Pipeline",
			"additional_info": "[Document1, Document2, etc]"
		}, ...
	],
	"next_token": 8
}
```

| Field | Type | Optional | Description |
| --- | --- | --- | --- |
| interactions | List[Interaction] | no | list of conversation objects |
| next_token | int | yes | next next_token to use to continue retrieving conversations. If no more conversations, next_token does not appear |

Interaction

| Field | Type | Optional | Description |
| --- | --- | --- | --- |
| conversation_id | string | no | id of the conversation this interaction belongs to |
| interaction_id | string | no | id of this interaction |
| create_time | timestamp | no | When this interaction was created |
| input | string | no | human input that started this interaction |
| prompt_template | string | yes | prompt template used to construct the prompt for this interaction |
| response | string | no | AI’s response for this interaction |
| origin | string | yes | Name of the system that generated this interaction |
| additional_info | string | yes | Any additional info required for reconstructing the prompt; e.g. search results in RAG |

## DeleteConversation

Deletes a conversation and all of its interactions

```javascript
DELETE /_plugins/_ml/memory/conversation/{conversation_id}
```

| Field | Type | Optional | Description |
| --- | --- | --- | --- |
| conversation_id | string | no | id of the conversation to delete |

Response

```javascript
{
	"success": true
}
```

| Field | Type | Optional | Description |
| --- | --- | --- | --- |
| success | boolean | no | whether the deletion was successful |
