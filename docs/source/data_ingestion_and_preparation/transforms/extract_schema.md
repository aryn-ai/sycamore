## Extract Schema
The Extract Schema Transform allows you to extract a semantically meaningful schema for your documents. These schemas can then by populated using the Extract Properties transform.

The first step is to use Extract Schema to associate each document with a schema. Here, a Schema is a set of metadata from your document. For example, given a credit card agreement PDF, we extract the following:

```python
credit_docs = credit_docs.extract_batch_schema(
    schema_extractor=OpenAISchemaExtractor("CreditCardContract", llm=openai, num_of_elements=50)
)
```
Which will produce JSON-schema formatted metadata, stored in reach Document's `properties["_schema"]`:
```json
{
  "type": "object",
  "properties": {
    "creditCardName": {
      "type": "string"
    },
    "aprPurchases": {
      "type": "string"
    },
    "annualFee": {
      "type": "string"
    }
  },
}
```

Once a schema is extracted, we can populate the values using the Extract Properties transform.
```python
credit_docs.extract_properties(property_extractor=OpenAIPropertyExtractor(llm=openai, num_of_elements=50))
```
The values will be extracted from the document:
```json
{"creditCardName": "Dollar Bank Secured Credit Card - Variable Rate Line of Credit Agreement",
  "aprPurchases": "12.24%",
  "annualFee": "$15.00"},
```
