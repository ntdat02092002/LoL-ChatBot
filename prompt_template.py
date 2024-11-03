from langchain_core.prompts import PromptTemplate
from langchain.chains.query_constructor.base import (
    get_query_constructor_prompt,
)
from langchain.chains.query_constructor.base import AttributeInfo, StructuredQuery


CHATBOT_TEMPLATE = """
        You are a League of Legends assistant. Users will ask you questions about the patch updates.
        Use the following context to answer the question. This is the retrieved information to support your answer to the user's questions.
        If you don't know the answer, respond with "No information found" and encourage the user to provide a more detailed question. 
        Keep the answer relevant to the patch update and concise.
        There may be new knowledge/information that you are not aware of or hasn't been updated. Just rely on the retrieved information to respond.

        Context: {context}
        Question: {question}
        Answer: 

        """

DOCUMENT_CONTENT_DESCRIPTION = "Information of the lastest LOL update patch"

# Define allowed comparators list
ALLOWED_COMPARATORS = [
    "$eq",  # Equal to (number, string, boolean)
    # "$in",  # In array (string or number)
]

# Define allowed operators list
ALLOWED_OPERATORS = [
    # "and",
    "or"
]

EXAMPLES = [
    (
        "What changes does Zoe have?",
        {
            "query": "Zoe",
            "filter": "eq(\"category\", \"champion\")"
        }
    ),
    (
        "Were any components nerfed in the patch?",
        {
            "query": "components nerf",
            "filter": "eq(\"category\", \"item\")"
        }
    ),
    (
        "Are there any buffs for Jinx or Infinity Edge?",
        {
            "query": "buff Jinx Infinity Edge",
            "filter": "or(eq(\"category\", \"champion\"), eq(\"category\", \"item\"))"
        }
    ),
    (
        "Are there any new skins in this patch?",
        {
            "query": "new skins",
            "filter": "eq(\"category\", \"other\")"
        }
    ),
    (
        "Is there any new event or game mode in the latest update?",
        {
            "query": "new event or game mode",
            "filter": "or(eq(\"category\", \"overview\"), eq(\"category\", \"other\"))"
        }
    ),
    (
        "What changes were introduced in the patch",
        {
            "query": "changes",
            "filter": "eq(\"category\", \"overview\")"
        }
    ),
    (
        "hello how are you",
        {
            "query": "None",
            "filter": "eq(\"category\", \"none\")"
        }
    ),
]

METADATA_FIELD_INFO = [
    AttributeInfo(name="category", description="The category of the query. One of ['overview' (summary patch), 'champion' (buff/nerf/modify champion), 'item' (buff/nerf/modify item), 'other'(other info), 'none' (only use if query not relevant game)]",
                    type="string"),
]   

JSON_SCHEMA = r"""
    ```json
    {{{{
        "query": string \\ text string to compare to document contents
        "filter": string \\ logical condition statement for filtering documents
    }}}}
    ```
"""
    
QUERY_TEMPLATE = f"""
    << Structured Request Schema >>
    When responding use a markdown code snippet with a JSON object formatted in the following schema:

    {JSON_SCHEMA}

    The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.

    A logical condition statement is composed of one or more comparison and logical operation statements.

    A comparison statement takes the form: `comp(attr, val)`:
    - `comp` ({{allowed_comparators}}): comparator
    - `attr` (string):  name of attribute to apply the comparison to
    - `val` (string): is the comparison value

    A logical operation statement takes the form `op(statement1, statement2, ...)`:
    - `op` ({{allowed_operators}}): logical operator
    - `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to

    Make sure that you only use the comparators and logical operators listed above and no others.
    Make sure that filters only refer to attributes that exist in the data source.
    Make sure that filters only use the attributed names with its function names if there are functions applied on them.

    Below is the description of the Data Source and {len(EXAMPLES) + 1} Examples:
"""


def get_chatbot_prompt():
    return PromptTemplate(
            template=CHATBOT_TEMPLATE, 
            input_variables=["context", "question"]
        )

def get_constructor_prompt(type):
    if type == "default":
        constructor_prompt = get_query_constructor_prompt(
                DOCUMENT_CONTENT_DESCRIPTION,
                METADATA_FIELD_INFO,
                allowed_comparators=ALLOWED_COMPARATORS,
                allowed_operators=ALLOWED_OPERATORS,
                examples=EXAMPLES,
            )
    elif type == "custom":
        constructor_prompt = get_query_constructor_prompt(
                DOCUMENT_CONTENT_DESCRIPTION,
                METADATA_FIELD_INFO,
                allowed_comparators=ALLOWED_COMPARATORS,
                allowed_operators=ALLOWED_OPERATORS,
                schema_prompt=PromptTemplate(template=QUERY_TEMPLATE,
                input_variables=["allowed_comparators","allowed_operators"]),
                examples=EXAMPLES,
            )
    else:
        assert "Error in get_query_constructor_prompt, only support type in ['default', 'custom']"
    return constructor_prompt